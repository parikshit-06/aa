"""
src/convex_mpc/adaptive_gait.py

Drop-in replacement for Gait. Same constructor, same swing/touchdown planning,
but the contact schedule comes from a per-leg phase STATE that is overridden by the
contact estimator instead of from absolute time.

Phase convention is identical to gait.py:  phase = offset + t*hz (mod 1),
stance for phase < duty, swing otherwise, wrap 1 -> 0 is nominal touchdown.

    STANCE -> SWING  phase reaches duty            (inhibited while any leg is LATE)
    SWING  -> STANCE early:   contact seen after `min_swing_progress`; phase reset to 0
    SWING  -> LATE   phase reached 1.0, no contact; phase held, foot keeps searching down
    LATE   -> STANCE contact; phase reset to 0
"""
import numpy as np
from .gait import Gait
from .go2_robot_data import LEG_ORDER


class AdaptiveGait(Gait):
    STANCE, SWING, LATE = 0, 1, 2

    def __init__(self, *args, min_swing_progress=0.3, early_max_progress=0.9,
                 k_sync=0.5, v_search=0.25, max_search_depth=0.10, late_grace=0.03, flat_td_z=0.02,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.flat_td_z = float(flat_td_z)       # foot-frame height on flat ground (measure it, see ex05)
        self.td_z = np.full(4, self.flat_td_z)  # learned foot-frame height at last touchdown, per leg
        self._foot_z = None
        self.min_prog = min_swing_progress      # ignore contact right after liftoff
        self.early_max = early_max_progress     # beyond this, contact is just "nominal"
        self.k_sync = k_sync                    # Hz, pull back toward nominal offsets
        self.v_search = v_search                # m/s, foot descent while LATE
        self.max_depth = max_search_depth       # m
        self.late_grace = late_grace            # s, estimator lag below this is not "late"
        self.reset(0.0)

    # ------------------------------------------------------------------
    def reset(self, t0=0.0):
        """Call when the gait starts (e.g. when switching stand -> trot)."""
        self.phi_g = (t0 * self.gait_hz) % 1.0
        self.phase = (self.phase_offsets + self.phi_g) % 1.0
        self.state = np.where(self.phase < self.gait_duty, self.STANCE, self.SWING)
        self.late_time = np.zeros(4)
        self.search_depth = np.zeros(4)         # LegController lowers the swing target by this
        self.hold_stance = np.zeros(4, bool)    # liftoff inhibited this tick
        self.events = []                        # [(leg_idx, "early"|"late"|"nominal")]

    def td_height(self, leg):
        """Touchdown height used by the swing planner (replaces the hardcoded 0.02 in gait.py)."""
        return float(self.td_z[LEG_ORDER.index(leg.upper())])

    def consume_events(self):
        """Non-nominal touchdowns since last call -> use to trigger an immediate MPC re-solve."""
        ev = [e for e in self.events if e[1] != "nominal"]
        self.events = []
        return ev

    # ------------------------------------------------------------------
    def _touchdown(self, i, kind):
        self.state[i] = self.STANCE
        self.phase[i] = 0.0
        self.search_depth[i] = 0.0
        self.late_time[i] = 0.0
        self.events.append((i, kind))
        if self._foot_z is not None:
            self.td_z[i] = np.clip(self._foot_z[i], self.flat_td_z - 0.15, self.flat_td_z + 0.25)
        if kind != "nominal":
            self.phi_g = (-self.phase_offsets[i]) % 1.0   # re-anchor nominal clock to this leg

    def update(self, contact, dt, foot_z=None):
        """Advance the per-leg phases one control tick.
        contact: bool[4] from ContactFusion; foot_z: measured world z of the 4 feet (learns terrain)."""
        self._foot_z = foot_z
        if self.gait_duty >= 1.0:
            return
        duty, hz = self.gait_duty, self.gait_hz
        self.hold_stance[:] = False
        any_late = bool(np.any(self.state == self.LATE))
        if not any_late:
            self.phi_g = (self.phi_g + dt * hz) % 1.0

        for i in range(4):
            s = self.state[i]
            if s == self.LATE:
                self.late_time[i] += dt
                self.search_depth[i] = min(self.search_depth[i] + self.v_search * dt, self.max_depth)
                if contact[i]:
                    self._touchdown(i, "late" if self.late_time[i] > self.late_grace else "nominal")
                continue

            rate = hz
            if not any_late:
                e = ((self.phi_g + self.phase_offsets[i] - self.phase[i] + 0.5) % 1.0) - 0.5
                rate += self.k_sync * np.clip(e, -0.15, 0.15)
            new = self.phase[i] + rate * dt

            if s == self.SWING:
                prog = (self.phase[i] - duty) / (1.0 - duty)
                if contact[i] and prog > self.min_prog:
                    self._touchdown(i, "early" if prog < self.early_max else "nominal")
                elif new >= 1.0:
                    if contact[i]:
                        self._touchdown(i, "nominal")
                    else:
                        self.state[i] = self.LATE
                        self.phase[i] = 1.0 - 1e-6
                else:
                    self.phase[i] = new
            else:  # STANCE
                if new >= duty:
                    if any_late:
                        self.phase[i] = duty - 1e-6      # extend stance to keep support
                        self.hold_stance[i] = True
                    else:
                        self.state[i] = self.SWING
                        self.phase[i] = new
                else:
                    self.phase[i] = new

    # ------------------------------------------------------------------
    def compute_contact_table(self, t0: float, dt: float, N: int) -> np.ndarray:
        """(4,N) 1=stance. t0 is ignored: the table is projected from the current per-leg phases.
        Same half-step sampling as the original Gait. compute_current_mask() (dt=0,N=1) still works."""
        if self.gait_duty >= 1.0:
            return np.ones((4, N), dtype=np.int32)
        t_ahead = np.arange(N) * dt + dt / 2.0
        phases = np.mod(self.phase[:, None] + t_ahead[None, :] * self.gait_hz, 1.0)
        table = (phases < self.gait_duty).astype(np.int32)
        if N > 0:
            table[self.state == self.LATE, 0] = 0   # not touching yet, expected soon (k>=1 wraps to stance)
            table[self.hold_stance, 0] = 1          # still standing although schedule says liftoff
        return table
