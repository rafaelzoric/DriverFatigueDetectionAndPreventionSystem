import serial
import pynmea2
import time
import statistics
from collections import deque

# ── Config ───────────────────────────────────────────────────────────────────
GPS_PORT = "/dev/ttyAMA0"   # UART0 on Pi 5 (see Readme: dtoverlay=uart0-pi5)
GPS_BAUD = 9600

SPEED_GATE_ON_KMH  = 60.0   # algorithm only activates above this speed (highway driving)
SPEED_GATE_OFF_KMH = 55.0   # hysteresis: stays active until speed drops below this

EMA_ALPHA      = 0.1        # smoothing factor for the speed baseline (lower = smoother/slower)
WINDOW_SECONDS = 45         # rolling window for the residual standard deviation

SD_MIN = 1.0   # km/h — residual SD at/below this contributes 0 to F_gps
SD_MAX = 8.0   # km/h — residual SD at/above this contributes 100 to F_gps

ZERO_CROSS_HZ_MAX = 0.3   # oscillation rate ~0.3 Hz treated as fully drowsy-like


def knots_to_kmh(knots):
    return knots * 1.852


class GpsFatigueMonitor:
    """
    Scores erratic highway speed-holding as a fatigue indicator.

    Baseline speed is tracked with an EMA (cheap, O(1), low lag). The fatigue
    signal is the rolling standard deviation of the residual (speed - baseline)
    over WINDOW_SECONDS, since drowsy driving shows up as oscillation around a
    speed rather than a change in average speed. A secondary, smaller signal
    is how often the residual crosses zero (oscillation frequency).

    Only runs above SPEED_GATE_ON_KMH, with hysteresis down to
    SPEED_GATE_OFF_KMH so it doesn't flicker on/off near the threshold.
    """

    def __init__(self):
        self.ema = None
        self.active = False
        self.residuals = deque()   # (timestamp, residual_kmh)
        self.last_residual_sign = None
        self.zero_crossings = 0

    def _update_ema(self, speed_kmh):
        if self.ema is None:
            self.ema = speed_kmh
        else:
            self.ema = EMA_ALPHA * speed_kmh + (1 - EMA_ALPHA) * self.ema
        return self.ema

    def _prune_window(self, now):
        while self.residuals and now - self.residuals[0][0] > WINDOW_SECONDS:
            self.residuals.popleft()

    def update(self, speed_kmh, now):
        """Feed in a new GPS speed reading. Returns F_gps (0-100), or None if the gate is inactive."""

        if not self.active and speed_kmh >= SPEED_GATE_ON_KMH:
            self.active = True
            self.ema = speed_kmh
            self.residuals.clear()
            self.zero_crossings = 0
            self.last_residual_sign = None
        elif self.active and speed_kmh < SPEED_GATE_OFF_KMH:
            self.active = False

        if not self.active:
            return None

        baseline = self._update_ema(speed_kmh)
        residual = speed_kmh - baseline

        self.residuals.append((now, residual))
        self._prune_window(now)

        sign = 1 if residual > 0 else (-1 if residual < 0 else 0)
        if self.last_residual_sign is not None and sign != 0 and sign != self.last_residual_sign:
            self.zero_crossings += 1
        if sign != 0:
            self.last_residual_sign = sign

        if len(self.residuals) < 5:
            return 0.0   # not enough samples yet to trust the SD

        sd = statistics.pstdev(r for _, r in self.residuals)
        sd_score = (sd - SD_MIN) / (SD_MAX - SD_MIN) * 100
        sd_score = max(0.0, min(100.0, sd_score))

        window_span = max(now - self.residuals[0][0], 1.0)
        crossing_rate = self.zero_crossings / window_span
        crossing_score = min(crossing_rate / ZERO_CROSS_HZ_MAX, 1.0) * 100

        f_gps = 0.85 * sd_score + 0.15 * crossing_score
        return round(f_gps, 1)


def read_gps_speed(line):
    """Parse one NMEA sentence and return speed in km/h, or None if it carries no speed fix."""
    try:
        msg = pynmea2.parse(line)
    except pynmea2.ParseError:
        return None

    if isinstance(msg, pynmea2.types.talker.RMC) and msg.spd_over_grnd is not None:
        return knots_to_kmh(float(msg.spd_over_grnd))
    if isinstance(msg, pynmea2.types.talker.VTG) and msg.spd_over_grnd_kmph is not None:
        return float(msg.spd_over_grnd_kmph)

    return None


if __name__ == "__main__":
    monitor = GpsFatigueMonitor()

    with serial.Serial(GPS_PORT, GPS_BAUD, timeout=1) as ser:
        print("Listening for GPS fixes...")
        while True:
            try:
                line = ser.readline().decode("ascii", errors="replace").strip()
            except serial.SerialException:
                continue

            if not line:
                continue

            speed_kmh = read_gps_speed(line)
            if speed_kmh is None:
                continue

            now = time.monotonic()
            f_gps = monitor.update(speed_kmh, now)

            if f_gps is None:
                print(f"Speed: {speed_kmh:5.1f} km/h | monitor inactive (<{SPEED_GATE_OFF_KMH} km/h)")
            else:
                print(f"Speed: {speed_kmh:5.1f} km/h | F_gps: {f_gps:5.1f}")
