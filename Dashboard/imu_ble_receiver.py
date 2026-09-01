"""
SleepAway — IMU BLE Receiver
=============================
Receives calibrated IMU data from the ESP32-C3 headband module
over Bluetooth Low Energy.

Run standalone to verify the link:
    python3 imu_ble_receiver.py

Or import into main.py:
    from imu_ble_receiver import IMUReceiver
    imu = IMUReceiver()
    imu.start()                    # spawns background thread
    ...
    score = imu.get_score()        # call from your main loop

Install:
    pip install bleak --break-system-packages
"""

import asyncio
import struct
import threading
import time
import math
from collections import deque

from bleak import BleakScanner, BleakClient


# ── Must match the ESP32 sketch exactly ───────────────────────────
DEVICE_NAME         = "SleepAway-Headband"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

ACCEL_SCALE = 400.0
GYRO_SCALE  = 3000.0

# ── Nod detection parameters — tune these during testing ──────────
NOD_SENSITIVITY   = 20.0    # degrees below baseline to count as dropped
NOD_HOLD_TIME     = 1.0     # seconds head must stay down
NOD_WINDOW        = 10.0    # seconds within which repeated nods count
NOD_THRESHOLD     = 3       # nods needed for a full-score alert
BASELINE_SAMPLES  = 40      # readings averaged to establish upright pose

STALE_AFTER_SEC   = 5.0     # no data for this long -> score drops to 0
BUFFER_SIZE       = 100     # rolling window kept for future ML use


class IMUReceiver:
    """
    Owns the BLE connection in a background thread and exposes
    thread-safe accessors for the main loop to poll.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Latest reading
        self.ax = self.ay = self.az = 0.0
        self.gx = self.gy = self.gz = 0.0
        self.pitch = 0.0
        self.roll  = 0.0

        self.connected   = False
        self.last_update = 0.0

        # Rolling buffer — useful later for ML windowing
        self.buffer = deque(maxlen=BUFFER_SIZE)

        # Baseline (upright head pose), established on first N samples
        self._baseline_pitch   = None
        self._baseline_samples = []

        # Nod detection state
        self._head_down      = False
        self._nod_start      = 0.0
        self._last_nod_time  = 0.0
        self._nod_count      = 0

        self._thread = None
        self._stop   = threading.Event()

    # ── Called by the BLE thread on each notification ─────────────
    def _handle_packet(self, sender, data):
        if len(data) != 12:
            return   # ignore malformed packets rather than crashing

        raw = struct.unpack("<6h", data)   # little-endian, 6 signed shorts

        ax = raw[0] / ACCEL_SCALE
        ay = raw[1] / ACCEL_SCALE
        az = raw[2] / ACCEL_SCALE
        gx = raw[3] / GYRO_SCALE
        gy = raw[4] / GYRO_SCALE
        gz = raw[5] / GYRO_SCALE

        # Derive pitch and roll from the accelerometer
        pitch = math.degrees(math.atan2(ax, math.sqrt(ay * ay + az * az)))
        roll  = math.degrees(math.atan2(ay, math.sqrt(ax * ax + az * az)))

        with self._lock:
            self.ax, self.ay, self.az = ax, ay, az
            self.gx, self.gy, self.gz = gx, gy, gz
            self.pitch, self.roll     = pitch, roll
            self.last_update          = time.time()
            self.buffer.append((ax, ay, az, gx, gy, gz))

            # Establish baseline from the first N readings
            if self._baseline_pitch is None:
                self._baseline_samples.append(pitch)
                if len(self._baseline_samples) >= BASELINE_SAMPLES:
                    self._baseline_pitch = (sum(self._baseline_samples)
                                            / len(self._baseline_samples))
                    print(f"[IMU] Baseline pitch: {self._baseline_pitch:.1f} deg")
                return

            self._update_nod_state(pitch)

    # ── Nod state machine — mirrors the ESP32-side logic ──────────
    def _update_nod_state(self, pitch):
        now = time.time()
        delta = pitch - self._baseline_pitch
        dropped = delta < -NOD_SENSITIVITY

        if dropped:
            if not self._head_down:
                self._head_down = True
                self._nod_start = now
            elif now - self._nod_start >= NOD_HOLD_TIME:
                # Confirmed nod
                if now - self._last_nod_time < NOD_WINDOW:
                    self._nod_count += 1
                else:
                    self._nod_count = 1
                self._last_nod_time = now
                self._head_down     = False
                print(f"[IMU] Nod detected (count: {self._nod_count})")
        else:
            self._head_down = False

        # Expire the nod count once the window lapses
        if self._nod_count > 0 and now - self._last_nod_time > NOD_WINDOW:
            self._nod_count = 0

    # ── Public accessors, safe to call from the main loop ─────────
    def get_score(self):
        """0-100 IMU fatigue score. Returns 0 if data is stale."""
        with self._lock:
            if time.time() - self.last_update > STALE_AFTER_SEC:
                return 0
            if self._nod_count >= NOD_THRESHOLD:     return 100
            if self._nod_count >= NOD_THRESHOLD - 1: return 75
            if self._nod_count >= 1:                 return 25
            return 0

    def get_reading(self):
        """Latest calibrated values as a dict."""
        with self._lock:
            return {
                "ax": self.ax, "ay": self.ay, "az": self.az,
                "gx": self.gx, "gy": self.gy, "gz": self.gz,
                "pitch": self.pitch, "roll": self.roll,
                "connected": self.connected,
                "age": time.time() - self.last_update,
            }

    def get_window(self):
        """Copy of the rolling buffer — for future ML classification."""
        with self._lock:
            return list(self.buffer)

    def is_connected(self):
        with self._lock:
            return self.connected

    def reset_baseline(self):
        """Re-establish the upright reference, e.g. after refitting."""
        with self._lock:
            self._baseline_pitch   = None
            self._baseline_samples = []
            self._nod_count        = 0
        print("[IMU] Baseline reset")

    # ── BLE loop ──────────────────────────────────────────────────
    async def _ble_loop(self):
        while not self._stop.is_set():
            try:
                print("[IMU] Scanning for headband...")
                devices = await BleakScanner.discover(timeout=8.0)
                target = next((d for d in devices if d.name == DEVICE_NAME), None)

                if target is None:
                    print("[IMU] Headband not found — retrying in 5 s")
                    with self._lock:
                        self.connected = False
                    await asyncio.sleep(5)
                    continue

                print(f"[IMU] Connecting to {target.address}")
                async with BleakClient(target.address) as client:
                    await client.start_notify(CHARACTERISTIC_UUID,
                                              self._handle_packet)
                    with self._lock:
                        self.connected = True
                    print("[IMU] Connected — receiving data")

                    while client.is_connected and not self._stop.is_set():
                        await asyncio.sleep(0.5)

                print("[IMU] Disconnected")
                with self._lock:
                    self.connected = False

            except Exception as e:
                print(f"[IMU] Error: {e} — retrying in 5 s")
                with self._lock:
                    self.connected = False
                await asyncio.sleep(5)

    def _thread_entry(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ble_loop())

    def start(self):
        """Spawn the BLE thread. Non-blocking."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_entry, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════
#  Standalone test — verifies the BLE link independently
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    imu = IMUReceiver()
    imu.start()

    print("Receiving — press Ctrl+C to stop")
    print("Hold the sensor upright and still while the baseline is set.\n")

    try:
        while True:
            r = imu.get_reading()
            score = imu.get_score()

            if r["connected"]:
                print(f"A: {r['ax']:6.2f} {r['ay']:6.2f} {r['az']:6.2f}  "
                      f"G: {r['gx']:6.2f} {r['gy']:6.2f} {r['gz']:6.2f}  "
                      f"Pitch: {r['pitch']:6.1f}  Roll: {r['roll']:6.1f}  "
                      f"Score: {score:3d}",
                      end="\r")
            else:
                print("Waiting for headband...", end="\r")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping")
        imu.stop()
