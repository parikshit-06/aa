"""
src/convex_mpc/contact_debug.py

Validation helpers. MuJoCo ground-truth contact is used ONLY for logging / plots,
never fed back into the controller.
"""
import numpy as np
from .go2_robot_data import LEG_ORDER


def find_foot_geoms(mj_model):
    """Geom ids for FL, FR, RL, RR.

    The A2 MJCF has unnamed foot spheres (class="foot") inside the <leg>_calf bodies, so
    look them up by body + geom type; fall back to geom names if that finds nothing.
    """
    import mujoco as mj
    ids = []
    for leg in LEG_ORDER:
        gid = -1
        for body_name in (f"{leg}_calf", f"{leg}_foot"):
            bid = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                continue
            cands = [g for g in range(mj_model.ngeom)
                     if mj_model.geom_bodyid[g] == bid
                     and mj_model.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE
                     and mj_model.geom_contype[g] + mj_model.geom_conaffinity[g] > 0]
            if cands:
                gid = min(cands, key=lambda g: mj_model.geom_pos[g][2])   # lowest sphere in the body
                break
        if gid < 0:
            for pat in ("{leg}", "{leg}_foot", "{leg}_foot_geom"):
                gid = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_GEOM, pat.format(leg=leg))
                if gid >= 0:
                    break
        if gid < 0:
            raise RuntimeError(f"No foot sphere geom found for {leg}; check the MJCF foot class "
                               f"(needs contype/conaffinity > 0).")
        ids.append(gid)
    return ids


def ground_truth_normal_force(mj_model, mj_data, foot_geom_ids):
    """Normal contact force [N] per foot (0 if not touching)."""
    import mujoco as mj
    fz = np.zeros(4)
    buf = np.zeros(6)
    for c in range(mj_data.ncon):
        con = mj_data.contact[c]
        for i, gid in enumerate(foot_geom_ids):
            if con.geom1 == gid or con.geom2 == gid:
                mj.mj_contactForce(mj_model, mj_data, c, buf)
                fz[i] += abs(buf[0])       # component 0 = normal force in the contact frame
    return fz


class ContactLog:
    def __init__(self, n):
        self.n = n
        self.i = 0
        self.t = np.zeros(n)
        self.fz_est = np.zeros((n, 4))
        self.fz_gt = np.zeros((n, 4))
        self.P = np.zeros((n, 4))
        self.contact = np.zeros((n, 4))
        self.state = np.zeros((n, 4))
        self.phase = np.zeros((n, 4))
        self.td_z = np.zeros((n, 4))
        self.events = []                       # (t, leg, kind)

    def record(self, t, f_hat, fusion, gait, fz_gt, events=()):
        k = min(self.i, self.n - 1)
        self.t[k] = t
        self.fz_est[k] = f_hat[:, 2]
        self.fz_gt[k] = fz_gt
        self.P[k] = fusion.P
        self.contact[k] = fusion.contact
        self.state[k] = gait.state
        self.phase[k] = gait.phase
        self.td_z[k] = gait.td_z
        self.events += [(t, leg, kind) for leg, kind in events]
        self.i += 1

    def summary(self, thresh=10.0):
        """Estimated-vs-true contact agreement, and mean detection delay of true contact onsets."""
        m = self.i
        gt = self.fz_gt[:m] > thresh
        est = self.contact[:m] > 0.5
        acc = (gt == est).mean(axis=0)
        delays = []
        for leg in range(4):
            on_gt = np.where(~gt[:-1, leg] & gt[1:, leg])[0] + 1
            for k in on_gt:
                later = np.where(est[k:, leg])[0]
                if len(later) and later[0] < 40:
                    delays.append(self.t[k + later[0]] - self.t[k])
        print("contact agreement per leg (FL FR RL RR):", np.round(acc, 3))
        if delays:
            print(f"touchdown detection delay: mean {1e3*np.mean(delays):.1f} ms, "
                  f"max {1e3*np.max(delays):.1f} ms over {len(delays)} touchdowns")
        print("override events:", len(self.events), "  early:",
              sum(e[2] == 'early' for e in self.events), " late:", sum(e[2] == 'late' for e in self.events))

    def plot(self):
        import matplotlib.pyplot as plt
        m = self.i
        t = self.t[:m]
        fig, ax = plt.subplots(4, 2, figsize=(14, 9), sharex=True)
        for i, leg in enumerate(LEG_ORDER):
            ax[i, 0].plot(t, self.fz_gt[:m, i], label="MuJoCo truth", lw=1)
            ax[i, 0].plot(t, self.fz_est[:m, i], label="GM estimate", lw=1)
            ax[i, 0].set_ylabel(f"{leg} Fz [N]")
            ax[i, 1].plot(t, self.P[:m, i], lw=1, label="fused P(contact)")
            ax[i, 1].step(t, self.contact[:m, i], where="post", lw=1, label="est contact")
            ax[i, 1].step(t, (self.fz_gt[:m, i] > 10) * 0.9, where="post", lw=1, ls="--", label="true contact")
            ax[i, 1].step(t, (self.state[:m, i] == 0) * 0.5, where="post", lw=1, ls=":", label="gait stance")
            for (te, le, kind) in self.events:
                if le == i:
                    ax[i, 1].axvline(te, color="r" if kind == "late" else "g", alpha=0.5)
        ax[0, 0].legend(loc="upper right", fontsize=7)
        ax[0, 1].legend(loc="upper right", fontsize=7)
        ax[-1, 0].set_xlabel("t [s]")
        ax[-1, 1].set_xlabel("t [s]  (green=early, red=late)")
        fig.tight_layout()
        self.summary()
        plt.show()
