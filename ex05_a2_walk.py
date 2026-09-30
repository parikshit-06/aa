"""
A2 MPC runner with named gaits, optional fixed foothold targets, and sensorless
contact estimation (GM observer + fusion) that overrides the gait schedule on
early / late touchdown.

Examples:
  python -m examples.ex05_a2_walk --gait stand --duration 2 --plot
  python -m examples.ex05_a2_walk --gait trot --vx 0.25 --duration 4 --plot
  python -m examples.ex05_a2_walk --gait trot --vx 0.25 --duration 4 --render
  python -m examples.ex05_a2_walk --gait walk --vx 0.15 --foothold FL:0.35,0.20,0.03
"""

import argparse
from dataclasses import dataclass

import mujoco as mj
import numpy as np
import pinocchio as pin

from convex_mpc.centroidal_mpc import CentroidalMPC
from convex_mpc.com_trajectory import ComTraj
from convex_mpc.gait import GAIT_PRESETS
from convex_mpc.adaptive_gait import AdaptiveGait
from convex_mpc.go2_robot_data import (
    A2_STAND_BASE_HEIGHT,
    A2_TORQUE_LIMITS,
    LEG_ORDER,
    PinGo2Model,
)
from convex_mpc.leg_controller import LegController
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from convex_mpc.contact_estimator import GMObserver, ContactFusion
from convex_mpc.contact_debug import ContactLog, find_foot_geoms, ground_truth_normal_force


# =============================================================================
# Simulation result
# =============================================================================

@dataclass
class SimResult:
    final_time: float
    final_base_pos: np.ndarray
    final_rpy: np.ndarray
    max_abs_roll_pitch: float

    time_log_render: np.ndarray
    q_log_render: np.ndarray
    tau_log_render: np.ndarray


# =============================================================================
# Joint indexing
# =============================================================================

LEG_SLICE = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}


# =============================================================================
# Foothold parser
# =============================================================================

def parse_foothold(value):
    leg, coords = value.split(":", 1)
    leg = leg.upper()

    if leg not in LEG_ORDER:
        raise argparse.ArgumentTypeError(f"leg must be one of {LEG_ORDER}")

    xyz = np.fromstring(coords, sep=",", dtype=float)

    if xyz.shape != (3,):
        raise argparse.ArgumentTypeError("foothold must look like FL:x,y,z")

    return leg, xyz


# =============================================================================
# Torque limits
# =============================================================================

def torque_limits(safety):
    per_leg = np.array(
        [
            A2_TORQUE_LIMITS["hip"],
            A2_TORQUE_LIMITS["thigh"],
            A2_TORQUE_LIMITS["calf"],
        ],
        dtype=float,
    )
    return safety * np.tile(per_leg, 4)


# =============================================================================
# Main simulation
# =============================================================================

def run(args):

    # -------------------------------------------------------------------------
    # Simulation rates
    # -------------------------------------------------------------------------

    sim_hz = 1000
    ctrl_hz = 200
    render_hz = 120.0

    sim_dt = 1.0 / sim_hz
    ctrl_dt = 1.0 / ctrl_hz
    render_dt = 1.0 / render_hz

    ctrl_decim = sim_hz // ctrl_hz

    sim_steps = int(args.duration * sim_hz)
    ctrl_steps = int(args.duration * ctrl_hz)

    # -------------------------------------------------------------------------
    # Create robot models
    # -------------------------------------------------------------------------

    print("Loading Pinocchio model...")
    go2 = PinGo2Model()

    # Foot-frame height on flat ground, from the nominal standing pose (A2 foot sphere is ~0.032 m,
    # so the old hardcoded 0.02 would place every touchdown target ~1 cm underground).
    z_flat = float(np.mean([p[2] for p in go2.get_foot_placement_in_world()]))
    print(f"Flat-ground foot-frame height from stand pose: {z_flat:.4f} m")

    print("Loading MuJoCo model...")
    mujoco_go2 = MuJoCo_GO2_Model()

    print("Models loaded successfully.")

    # -------------------------------------------------------------------------
    # Live MuJoCo viewer
    # -------------------------------------------------------------------------

    viewer = None

    if args.render:
        print("Starting MuJoCo viewer...")
        viewer = mj.viewer.launch_passive(mujoco_go2.model, mujoco_go2.data)

    # -------------------------------------------------------------------------
    # Controllers
    # -------------------------------------------------------------------------

    leg_controller = LegController()

    # -------------------------------------------------------------------------
    # Contact estimation: GM observer -> fusion -> adaptive gait
    # -------------------------------------------------------------------------

    gm_observer = GMObserver(dt=ctrl_dt, K_O=args.gm_gain)
    fusion = ContactFusion.for_robot(pin.computeTotalMass(go2.model))

    # -------------------------------------------------------------------------
    # Gait (per-leg phase state, overridden by estimated contact)
    # -------------------------------------------------------------------------

    footholds = dict(args.foothold or [])

    gait = AdaptiveGait(
        args.gait_hz,
        name=args.gait,
        foothold_targets=footholds,
        flat_td_z=z_flat,
    )

    # -------------------------------------------------------------------------
    # COM trajectory
    # -------------------------------------------------------------------------

    traj = ComTraj(go2)

    # -------------------------------------------------------------------------
    # Initial configuration
    # -------------------------------------------------------------------------

    q_init = go2.current_config.get_q()

    q_init[0] = args.x0
    q_init[1] = args.y0
    q_init[2] = args.height

    mujoco_go2.update_with_q_pin(q_init)
    mujoco_go2.model.opt.timestep = sim_dt

    # -------------------------------------------------------------------------
    # MPC timing
    # -------------------------------------------------------------------------

    gait_t = 1.0 / args.gait_hz
    mpc_dt = gait_t / 16
    mpc_hz = 1.0 / mpc_dt
    steps_per_mpc = max(1, int(ctrl_hz // mpc_hz))

    # -------------------------------------------------------------------------
    # Generate initial trajectory
    # -------------------------------------------------------------------------

    traj.generate_traj(
        go2,
        gait,
        0.0,
        args.vx,
        args.vy,
        args.height,
        args.yaw_rate,
        time_step=mpc_dt,
    )

    # -------------------------------------------------------------------------
    # Create MPC
    # -------------------------------------------------------------------------

    mpc = CentroidalMPC(go2, traj)

    u_opt = np.zeros((12, traj.N), dtype=float)

    # -------------------------------------------------------------------------
    # Torque limits / held torque (applied over the current control interval)
    # -------------------------------------------------------------------------

    tau_lim = torque_limits(args.torque_safety)
    tau_hold = np.zeros(12, dtype=float)

    # -------------------------------------------------------------------------
    # Contact logging (MuJoCo truth is for plots only, never used by control)
    # -------------------------------------------------------------------------

    foot_geoms = find_foot_geoms(mujoco_go2.model)
    clog = ContactLog(ctrl_steps)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    ctrl_i = 0
    max_abs_roll_pitch = 0.0
    next_render_t = 0.0

    time_log_render = []
    q_log_render = []
    tau_log_render = []

    # =========================================================================
    # Simulation loop
    # =========================================================================

    try:

        for k in range(sim_steps):

            time_now_s = float(mujoco_go2.data.time)

            # =================================================================
            # Controller update
            # =================================================================

            if (k % ctrl_decim) == 0 and ctrl_i < ctrl_steps:

                # -------------------------------------------------------------
                # Update Pinocchio from MuJoCo
                # -------------------------------------------------------------

                mujoco_go2.update_pin_with_mujoco(go2)

                # -------------------------------------------------------------
                # Contact estimation
                #
                # tau_hold still holds the (clipped) torque that was applied
                # over the previous control interval, which is exactly what
                # the momentum observer needs. It is overwritten further down.
                # -------------------------------------------------------------

                f_hat, v_foot, p_foot = gm_observer.update(go2, tau_hold)

                contact_est = fusion.update(
                    f_hat[:, 2],
                    np.linalg.norm(v_foot, axis=1),
                    gait.phase,
                    gait.gait_duty,
                )

                gait.update(contact_est, ctrl_dt, foot_z=p_foot[:, 2])
                events = gait.consume_events()   # early / late touchdowns this tick

                # -------------------------------------------------------------
                # Get current orientation
                # -------------------------------------------------------------

                rpy = go2.current_config.compute_euler_angle_world()

                max_abs_roll_pitch = max(
                    max_abs_roll_pitch,
                    float(np.max(np.abs(rpy[:2]))),
                )

                # -------------------------------------------------------------
                # Re-solve MPC (on schedule, or immediately after an override)
                # -------------------------------------------------------------

                if (ctrl_i % steps_per_mpc) == 0 or events:

                    # Terrain-adjusted height reference (mean learned touchdown height)
                    h_ref = args.height + float(np.mean(gait.td_z) - gait.flat_td_z)

                    traj.generate_traj(
                        go2,
                        gait,
                        time_now_s,
                        args.vx,
                        args.vy,
                        h_ref,
                        args.yaw_rate,
                        time_step=mpc_dt,
                    )

                    sol = mpc.solve_QP(go2, traj, False)

                    n = traj.N
                    w_opt = sol["x"].full().flatten()

                    # Extract force solution
                    u_opt = w_opt[12 * n:].reshape((12, n), order="F")

                # -------------------------------------------------------------
                # Leg controllers
                # -------------------------------------------------------------

                tau_raw = np.zeros(12, dtype=float)

                for leg in LEG_ORDER:
                    out = leg_controller.compute_leg_torque(
                        leg,
                        go2,
                        gait,
                        u_opt[LEG_SLICE[leg], 0],
                        time_now_s,
                    )
                    tau_raw[LEG_SLICE[leg]] = out.tau

                # -------------------------------------------------------------
                # Torque saturation
                # -------------------------------------------------------------

                tau_hold = np.clip(tau_raw, -tau_lim, tau_lim)

                # -------------------------------------------------------------
                # Contact log (ground truth from MuJoCo, plots only)
                # -------------------------------------------------------------

                clog.record(
                    time_now_s,
                    f_hat,
                    fusion,
                    gait,
                    ground_truth_normal_force(
                        mujoco_go2.model, mujoco_go2.data, foot_geoms
                    ),
                    events,
                )

                ctrl_i += 1

            # =================================================================
            # MuJoCo physics
            # =================================================================

            mj.mj_step1(mujoco_go2.model, mujoco_go2.data)

            # Apply actuator torques
            mujoco_go2.set_joint_torque(tau_hold)

            mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

            # =================================================================
            # Live viewer update
            # =================================================================

            if viewer is not None:
                viewer.sync()

            # =================================================================
            # Logging
            # =================================================================

            t_after = float(mujoco_go2.data.time)

            if t_after + 1e-12 >= next_render_t:
                time_log_render.append(t_after)
                q_log_render.append(mujoco_go2.data.qpos.copy())
                tau_log_render.append(tau_hold.copy())
                next_render_t += render_dt

            # =================================================================
            # Numerical safety check
            # =================================================================

            if not np.all(np.isfinite(mujoco_go2.data.qpos)):
                raise FloatingPointError(
                    f"Simulation produced non-finite qpos at t={time_now_s:.3f}s"
                )

    finally:

        if viewer is not None:
            viewer.close()

    # =========================================================================
    # Final state
    # =========================================================================

    mujoco_go2.update_pin_with_mujoco(go2)

    if clog.i > 0:
        if args.plot:
            clog.plot()
        else:
            clog.summary()

    return SimResult(
        final_time=float(mujoco_go2.data.time),
        final_base_pos=mujoco_go2.data.qpos[:3].copy(),
        final_rpy=go2.current_config.compute_euler_angle_world(),
        max_abs_roll_pitch=max_abs_roll_pitch,
        time_log_render=np.asarray(time_log_render, dtype=float),
        q_log_render=np.asarray(q_log_render, dtype=float),
        tau_log_render=np.asarray(tau_log_render, dtype=float),
    )


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    # Gait
    parser.add_argument("--gait", choices=sorted(GAIT_PRESETS), default="trot")
    parser.add_argument("--gait-hz", type=float, default=4.0)

    # Simulation
    parser.add_argument("--duration", type=float, default=15.0)

    # Desired velocity
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)

    # Initial pose
    parser.add_argument("--height", type=float, default=A2_STAND_BASE_HEIGHT)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--y0", type=float, default=0.0)

    # Torque safety
    parser.add_argument("--torque-safety", type=float, default=0.75)

    # Footholds
    parser.add_argument("--foothold", action="append", type=parse_foothold, default=[])

    # Contact estimation
    parser.add_argument(
        "--gm-gain",
        type=float,
        default=80.0,
        help="GM observer gain K_O [rad/s]; higher = faster but noisier.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot estimated vs MuJoCo-truth contact after the run.",
    )

    # Rendering
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the MuJoCo viewer during simulation.",
    )
    parser.add_argument("--realtime-factor", type=float, default=1.0)

    args = parser.parse_args()

    result = run(args)

    print(f"final_time_s={result.final_time:.3f}")
    print(f"final_base_pos={np.array2string(result.final_base_pos, precision=4)}")
    print(f"final_rpy={np.array2string(result.final_rpy, precision=4)}")
    print(f"max_abs_roll_pitch_rad={result.max_abs_roll_pitch:.4f}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
