# A2 sensorless contact patch

Copy into `src/convex_mpc/`:
- `contact_estimator.py`  (GMObserver + ContactFusion)
- `adaptive_gait.py`      (AdaptiveGait, subclass of Gait)
- `contact_debug.py`      (ground-truth + log/plot helper, plots only)
- `leg_controller.py`     (replaces yours: LATE search descent + stance hold fallback)

## gait.py
Add to `Gait`:
```python
def td_height(self, leg):
    return 0.02
```
In both `compute_touchdown_world_for_traj_purpose_only` and `compute_swing_traj_and_touchdown`, replace
`pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], 0.02]` with
`pos_norminal_term = [hip_pos_world[0], hip_pos_world[1], self.td_height(leg)]`.

## examples/ex05_a2_walk.py
1. Imports: remove `from convex_mpc.gm_observer import GMObserver`, add
```python
import pinocchio as pin
from convex_mpc.adaptive_gait import AdaptiveGait
from convex_mpc.contact_estimator import GMObserver, ContactFusion
from convex_mpc.contact_debug import ContactLog, find_foot_geoms, ground_truth_normal_force
```
2. Observer:
```python
gm_observer = GMObserver(dt=1.0 / ctrl_hz, K_O=80.0)
fusion = ContactFusion.for_robot(pin.computeTotalMass(go2.model))
```
3. `gait = AdaptiveGait(args.gait_hz, name=args.gait, foothold_targets=footholds)`
4. Delete `gm_force_log` / `gm_tau_dist_log`; add
```python
foot_geoms = find_foot_geoms(mujoco_go2.model)
clog = ContactLog(ctrl_steps)
```
5. In the control block, delete the `joint_torque_to_generalized` lines. Right after
   `mujoco_go2.update_pin_with_mujoco(go2)` and before `tau_hold` is overwritten:
```python
f_hat, v_foot, p_foot = gm_observer.update(go2, tau_hold)
contact_est = fusion.update(f_hat[:, 2], np.linalg.norm(v_foot, axis=1),
                            gait.phase, gait.gait_duty)
gait.update(contact_est, 1.0 / ctrl_hz, foot_z=p_foot[:, 2])
events = gait.consume_events()
```
6. MPC block: `if (ctrl_i % steps_per_mpc) == 0 or events:` and before `generate_traj`
```python
h_ref = args.height + float(np.mean(gait.td_z) - 0.02)
```
   then pass `h_ref` instead of `args.height`.
7. After the leg-controller loop, before `ctrl_i += 1`:
```python
clog.record(time_now_s, f_hat, fusion, gait,
            ground_truth_normal_force(mujoco_go2.model, mujoco_go2.data, foot_geoms), events)
```
   Add a `--plot` flag and call `clog.plot()` before `return SimResult(...)` when set.

## Note on foot height
A2 foot spheres are ~0.032 m radius (Go2 ~0.022), so the old hardcoded touchdown z of 0.02 sits below the
ground. `ex05` now measures the flat foot-frame height from the stand pose (`z_flat`) and passes it as
`flat_td_z`; check the printed value is ~0.03 (sphere-centre frame) or ~0 (sole frame).

## Test order
stand (Fz ~ mg/4, no drift) -> flat trot (few events, 15-30 ms delay) -> step-up / walk-off.
