import time
import statistics
from collections import deque

# ── Config ───────────────────────────────────────────────────────────────────
BASELINE_ALPHA = 0.01   # very slow EMA — tracks natural postural drift (mirror
                         # checks, seat shifts), not the momentary head nod itself

WINDOW_SECONDS = 60      # rolling window for the head-down dwell ratio

DROP_ENTER_DEG = 12.0    # |pitch - baseline| above this => head counted as "dropped"
DROP_EXIT_DEG  = 6.0     # must fall back below this to be counted "recovered"
                         # (hysteresis band so borderline pitch doesn't chatter)

RATIO_MIN = 0.05   # dwell ratio at/below this contributes 0 to F_imu
RATIO_MAX = 0.40   # dwell ratio at/above this contributes 100 to F_imu

NOD_HZ_MAX = 0.15   # nod-recovery events/sec treated as fully drowsy-like

SEVERE_GYRO_DPS = 90.0   # a single fast pitch-rate spike this large = a head
                         # already falling (microsleep jerk), not a slow drift
LATCH_SECONDS   = 2.0    # how long a severe spike forces F_imu to 100, so the
                         # <1s alert-response target isn't gated on the window


class ImuFatigueMonitor:
    """
    Scores head-nod behaviour as a fatigue indicator from headband pitch.

    Baseline "neutral" head pitch is tracked with a slow EMA so gradual
    postural changes don't get flagged as drowsiness. The primary signal is
    the fraction of the rolling window spent with the head pitched away from
    that baseline (a PERCLOS-style dwell ratio, analogous to eye-closure
    percentage). A secondary signal is the rate of distinct nod-then-recover
    events, since repeated short nods are more characteristic of microsleep
    than one sustained tilt.

    A separate fast path latches F_imu to 100 for a couple of seconds
    whenever the gyro reports a rapid pitch-rate spike — a sudden head drop
    reads as drowsy immediately rather than waiting for the window average
    to catch up, since the alert has to fire in under a second.
    """

    def __init__(self):
        self.baseline = None
        self.states = deque()   # (timestamp, dropped: bool)
        self.dropped = False
        self.nod_events = 0
        self.severe_until = None

    def _update_baseline(self, pitch_deg):
        if self.baseline is None:
            self.baseline = pitch_deg
        else:
            self.baseline = BASELINE_ALPHA * pitch_deg + (1 - BASELINE_ALPHA) * self.baseline
        return self.baseline

    def _prune_window(self, now):
        while self.states and now - self.states[0][0] > WINDOW_SECONDS:
            self.states.popleft()

    def update(self, pitch_deg, gyro_rate_deg_s, now):
        """Feed in one IMU sample. Returns F_imu (0-100)."""

        baseline = self._update_baseline(pitch_deg)
        residual = abs(pitch_deg - baseline)

        if not self.dropped and residual > DROP_ENTER_DEG:
            self.dropped = True
        elif self.dropped and residual < DROP_EXIT_DEG:
            self.dropped = False
            self.nod_events += 1

        self.states.append((now, self.dropped))
        self._prune_window(now)

        if abs(gyro_rate_deg_s) > SEVERE_GYRO_DPS:
            self.severe_until = now + LATCH_SECONDS

        if len(self.states) < 5:
            windowed_score = 0.0
        else:
            dwell_ratio = sum(1 for _, d in self.states if d) / len(self.states)
            ratio_score = (dwell_ratio - RATIO_MIN) / (RATIO_MAX - RATIO_MIN) * 100
            ratio_score = max(0.0, min(100.0, ratio_score))

            window_span = max(now - self.states[0][0], 1.0)
            nod_rate = self.nod_events / window_span
            nod_score = min(nod_rate / NOD_HZ_MAX, 1.0) * 100

            windowed_score = 0.85 * ratio_score + 0.15 * nod_score

        if self.severe_until is not None and now < self.severe_until:
            return 100.0

        return round(windowed_score, 1)


if __name__ == "__main__":
    # Standalone bench test — feeds pitch/gyro from stdin as "pitch_deg,gyro_dps"
    # per line. Real integration point: once the headband firmware exposes pitch
    # and gyro-Y over a BLE GATT characteristic (it currently only Serial-prints
    # them for debugging), replace this loop with a BLE notification handler
    # (e.g. via `bleak`) feeding the same monitor.update(...) call.
    import sys

    monitor = ImuFatigueMonitor()
    print("Reading 'pitch_deg,gyro_dps' lines from stdin...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            pitch_str, gyro_str = line.split(",")
            pitch_deg = float(pitch_str)
            gyro_dps = float(gyro_str)
        except ValueError:
            continue

        now = time.monotonic()
        f_imu = monitor.update(pitch_deg, gyro_dps, now)
        print(f"Pitch: {pitch_deg:6.2f} deg | Gyro: {gyro_dps:7.2f} dps | F_imu: {f_imu:5.1f}")
