"""
src/convex_mpc/contact_estimator.py

Sensorless contact estimation for the A2 convex-MPC controller.

    GMObserver     generalized-momentum observer -> per-foot GRF estimate
    ContactFusion  GRF + foot speed + gait-phase prior -> debounced contact flags

Reuses PinGo2Model: go2.data.{M,C,g} (from computeAllTerms in update_model),
go2.compute_3x3_foot_Jacobian_world(leg), go2.get_single_foot_state_in_world(leg).
Call order every control tick:  go2.update_model(q, dq)  ->  obs.update(go2, tau_prev)
"""
import numpy as np
from .go2_robot_data import LEG_ORDER


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


class GMObserver:
    """
    p = M v ;  p_dot = C^T v - g + S^T tau + sum_i J_i^T f_i
    r = K (p - iota) ;  iota_dot = C^T v - g + S^T tau + r    =>   r_dot = K (sum J^T f - r)
    Leg i joint rows:  r_leg_i = Jleg_i^T f_i   ->   f_i = (Jleg_i^T)^-1 r_leg_i  (damped)

    IMPORTANT: `tau` must be the joint torques applied over the *previous* control
    interval (post-clipping), because p(t_k) - p(t_{k-1}) is driven by tau_{k-1}.
    """

    def __init__(self, dt=0.005, K_O=80.0, damping=1e-3):
        self.dt, self.K, self.lam = dt, K_O, damping
        self.iota = None
        self.r = np.zeros(18)
        self.f_hat = np.zeros((4, 3))    # GRF acting on the robot, world frame
        self.v_foot = np.zeros((4, 3))   # world-frame foot velocity
        self.p_foot = np.zeros((4, 3))   # world-frame foot position

    def reset(self):
        self.iota = None

    def update(self, go2, tau):
        g, C, M = go2.compute_dynamcis_terms()
        M = np.triu(M) + np.triu(M, 1).T            # computeAllTerms/crba fill the upper triangle only
        v = go2.current_config.get_dq()
        p = M @ v
        beta = C.T @ v - g + np.concatenate([np.zeros(6), np.asarray(tau).reshape(12)])

        if self.iota is None:
            self.iota = p.copy()
        self.r = self.K * (p - self.iota)
        self.iota = self.iota + (beta + self.r) * self.dt

        for i, leg in enumerate(LEG_ORDER):
            J = go2.compute_3x3_foot_Jacobian_world(leg)
            r_leg = self.r[6 + 3 * i: 9 + 3 * i]
            self.f_hat[i] = np.linalg.solve(J @ J.T + self.lam * np.eye(3), J @ r_leg)
            self.p_foot[i], self.v_foot[i] = go2.get_single_foot_state_in_world(leg)
        return self.f_hat, self.v_foot, self.p_foot


class ContactFusion:
    """Log-odds fusion of phase prior, GRF and foot-speed evidence, then hysteresis + debounce."""

    def __init__(self, f_on=15.0, f_w=5.0, v_on=0.20, v_w=0.06,
                 w_prior=1.0, w_f=1.0, w_v=0.7,
                 p_hi=0.75, p_lo=0.35, n_debounce=2, prior_ramp=0.10):
        self.f_on, self.f_w, self.v_on, self.v_w = f_on, f_w, v_on, v_w
        self.w_prior, self.w_f, self.w_v = w_prior, w_f, w_v
        self.p_hi, self.p_lo, self.n_deb, self.ramp = p_hi, p_lo, n_debounce, prior_ramp
        self.contact = np.zeros(4, bool)
        self.cnt = np.zeros(4, int)
        self.P = np.full(4, 0.5)

    @classmethod
    def for_robot(cls, total_mass, **kw):
        """Force threshold = 20% of nominal per-foot static load."""
        return cls(f_on=0.2 * total_mass * 9.81 / 4.0, **kw)

    def phase_prior(self, phase, duty):
        stance = phase < duty
        edge = np.where(stance, np.minimum(phase, duty - phase),
                        np.minimum(phase - duty, 1.0 - phase))
        conf = np.clip(edge / self.ramp, 0.0, 1.0)
        return 0.5 + np.where(stance, 0.4, -0.4) * conf

    def update(self, fz, v_norm, phase, duty):
        p_f = sigmoid((fz - self.f_on) / self.f_w)
        p_v = sigmoid((self.v_on - v_norm) / self.v_w)
        L = (self.w_prior * logit(self.phase_prior(phase, duty))
             + self.w_f * logit(p_f) + self.w_v * logit(p_v))
        self.P = sigmoid(L)
        want = np.where(self.contact, self.P > self.p_lo, self.P > self.p_hi)
        flip = want != self.contact
        self.cnt = np.where(flip, self.cnt + 1, 0)
        self.contact = np.where(self.cnt >= self.n_deb, want, self.contact)
        return self.contact.copy()
