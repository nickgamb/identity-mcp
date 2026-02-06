"""
================================================================================
EMOTIV EPOC X READER
================================================================================

PURPOSE:
    Device abstraction for reading EEG from an EMOTIV Epoc X headset via the
    USB wireless dongle.  Includes a synthetic mode for testing without hardware.

CONNECTION:
    The Epoc X streams 32-byte packets at 256 Hz through its USB wireless
    dongle (VID 0x1234, PID 0xED02).  Packets alternate between EEG data
    and motion/extra data, yielding an effective 128 Hz EEG rate.

    Encryption: raw bytes are XOR'd with 0x55, then AES-128-ECB decrypted.
    The AES key is derived from the dongle's serial number (NOT the headset
    serial printed on the device).

    Direct USB cable only charges — it does NOT stream EEG data.
    Bluetooth LE is not supported here (Windows blocks GATT notifications).

CHANNELS (14 EEG, 10-20 system):
    AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

SPECIFICATIONS:
    - Sampling rate: 128 Hz EEG (256 Hz raw with interleaved motion packets)
    - Resolution: ~0.51 µV / LSB (16-bit two-byte pairs)
    - Packet size: 32 bytes (XOR 0x55 + AES-128-ECB encrypted)

USAGE:
    from emotiv_reader import EmotivReader

    reader = EmotivReader(mode="hid")            # auto-detect dongle
    reader = EmotivReader(mode="synthetic")       # no hardware

    reader.connect()
    data = reader.read_seconds(10)                # (14, N) numpy array, µV
    reader.disconnect()

ACKNOWLEDGEMENT:
    EPOC X protocol based on vtr0n/emotiv-lsl (Apache-2.0), which built on
    CyKit by CymatiCorp.

================================================================================
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Generator
from enum import Enum

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EMOTIV_VENDOR_ID  = 0x1234  # 4660
EMOTIV_PRODUCT_ID = 0xED02  # EPOC X

# Channel names in the order they are parsed from the decrypted packet
# (byte pairs at offsets 2-15 then 18-31, with swaps applied).
EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2",  "P8", "T8", "FC6", "F4", "F8", "AF4",
]

NUM_CHANNELS = len(EEG_CHANNELS)  # 14
SAMPLE_RATE  = 128  # Hz (effective EEG rate; dongle streams at 256 Hz interleaved)
PACKET_SIZE  = 32   # bytes

# Frequency bands (Hz) for spectral analysis
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 64.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# Channel byte positions in the decrypted 32-byte EPOC X packet.
#
# EEG data is stored as 14 two-byte pairs:
#   Bytes  2-15  → 7 channels (first half)
#   Bytes 16-17  → quality / reserved
#   Bytes 18-31  → 7 channels (second half)
#
# Raw parsing order:  F3, FC5, AF3, F7, T7, P7, O1, O2, P8, T8, F8, AF4, FC6, F4
# After swaps:        AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
# ─────────────────────────────────────────────────────────────────────────────

# (byte_lo, byte_hi) for each channel in raw parsing order
_RAW_BYTE_PAIRS = [
    (2, 3),    # → F3   (raw idx 0)
    (4, 5),    # → FC5  (raw idx 1)
    (6, 7),    # → AF3  (raw idx 2)
    (8, 9),    # → F7   (raw idx 3)
    (10, 11),  # → T7   (raw idx 4)
    (12, 13),  # → P7   (raw idx 5)
    (14, 15),  # → O1   (raw idx 6)
    (18, 19),  # → O2   (raw idx 7)  [skip bytes 16-17]
    (20, 21),  # → P8   (raw idx 8)
    (22, 23),  # → T8   (raw idx 9)
    (24, 25),  # → F8   (raw idx 10)
    (26, 27),  # → AF4  (raw idx 11)
    (28, 29),  # → FC6  (raw idx 12)
    (30, 31),  # → F4   (raw idx 13)
]

# Swap indices to reorder from raw to standard 10-20 channel order:
#   swap 0 ↔ 2  (F3 ↔ AF3)
#   swap 11 ↔ 13 (AF4 ↔ F4)
#   swap 1 ↔ 3  (FC5 ↔ F7)
#   swap 10 ↔ 12 (F8 ↔ FC6)
_SWAP_PAIRS = [(0, 2), (11, 13), (1, 3), (10, 12)]


class ConnectionMode(Enum):
    HID = "hid"
    SYNTHETIC = "synthetic"


# ─────────────────────────────────────────────────────────────────────────────
# AES key derivation  (EPOC X — different from original EPOC)
# ─────────────────────────────────────────────────────────────────────────────

def generate_aes_key(serial_number: str) -> bytes:
    """Derive AES-128 key from the **dongle** serial number.

    The EPOC X uses a key arrangement that differs from the original EPOC.
    Source: vtr0n/emotiv-lsl (based on CyKit).

    Args:
        serial_number: Dongle serial (16 chars, e.g. ``UD202311230073B5``).

    Returns:
        16-byte AES key.
    """
    sn = [ord(c) for c in serial_number]
    if len(sn) < 4:
        raise ValueError(f"Serial number too short: need ≥4 chars, got {len(sn)}")

    return bytes([
        sn[-1], sn[-2], sn[-4], sn[-4],
        sn[-2], sn[-1], sn[-2], sn[-4],
        sn[-1], sn[-4], sn[-3], sn[-2],
        sn[-1], sn[-2], sn[-2], sn[-3],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Packet parsing
# ─────────────────────────────────────────────────────────────────────────────

def _convert_uv(lo_byte: int, hi_byte: int) -> float:
    """Convert a two-byte EPOC X channel pair to microvolts.

    Formula from EMOTIV EDK / emotiv-lsl::

        µV = (lo * 0.128205128205129 + 4201.02564096001)
           + ((hi - 128) * 32.82051289)
    """
    return (lo_byte * 0.128205128205129 + 4201.02564096001) + \
           ((hi_byte - 128) * 32.82051289)


def parse_eeg_packet(decrypted: bytes) -> np.ndarray:
    """Parse a decrypted 32-byte EPOC X EEG packet into 14 channel values (µV).

    Returns a ``(14,)`` numpy array ordered per ``EEG_CHANNELS``.
    """
    raw_vals = []
    for lo_idx, hi_idx in _RAW_BYTE_PAIRS:
        raw_vals.append(_convert_uv(decrypted[lo_idx], decrypted[hi_idx]))

    # Apply channel-order swaps
    for a, b in _SWAP_PAIRS:
        raw_vals[a], raw_vals[b] = raw_vals[b], raw_vals[a]

    return np.array(raw_vals, dtype=np.float64)


def is_motion_packet(decrypted: bytes) -> bool:
    """Return True if this is a motion/extra-data packet (not EEG)."""
    # Motion packets have byte[1] == 0x20 (32)
    return decrypted[1] == 0x20


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticGenerator:
    """Generates realistic synthetic EEG for testing without hardware.

    Each *identity_seed* produces a unique, repeatable "person" with
    distinctive alpha frequency, amplitudes, and channel gains.
    """

    def __init__(self, identity_seed: int = 42):
        self.rng = np.random.RandomState(identity_seed)
        self.sample_counter = 0

        self._alpha_freq  = self.rng.uniform(9.0, 11.0)
        self._alpha_amp   = self.rng.uniform(15.0, 40.0)
        self._beta_freq   = self.rng.uniform(15.0, 25.0)
        self._beta_amp    = self.rng.uniform(5.0, 15.0)
        self._noise_level = self.rng.uniform(3.0, 8.0)

        self._channel_gains  = self.rng.uniform(0.7, 1.3, size=NUM_CHANNELS)
        self._channel_phases = self.rng.uniform(0, 2 * np.pi, size=NUM_CHANNELS)

        posterior = [EEG_CHANNELS.index(ch) for ch in ["O1", "O2", "P7", "P8"]]
        frontal  = [EEG_CHANNELS.index(ch) for ch in ["AF3", "AF4", "F3", "F4"]]

        self._alpha_weights = np.ones(NUM_CHANNELS) * 0.3
        for idx in posterior:
            self._alpha_weights[idx] = 1.0

        self._beta_weights = np.ones(NUM_CHANNELS) * 0.3
        for idx in frontal:
            self._beta_weights[idx] = 1.0

    def generate_samples(self, n_samples: int, task_label: str = "baseline") -> np.ndarray:
        """Return ``(14, n_samples)`` array of synthetic EEG in µV."""
        t = np.arange(self.sample_counter, self.sample_counter + n_samples) / SAMPLE_RATE
        self.sample_counter += n_samples

        alpha_mod, beta_mod, theta_mod = 1.0, 1.0, 0.3
        if task_label == "meditation":
            alpha_mod, theta_mod, beta_mod = 1.5, 0.8, 0.5
        elif task_label.startswith("word_"):
            alpha_mod, beta_mod = 0.6, 1.5
        elif task_label in ("blink", "fingers_right", "fingers_left"):
            beta_mod = 1.3
        elif task_label in ("face_happy", "face_sad", "face_neutral"):
            alpha_mod = 0.8

        data = np.zeros((NUM_CHANNELS, n_samples))
        for ch in range(NUM_CHANNELS):
            phase = self._channel_phases[ch]
            gain  = self._channel_gains[ch]
            alpha = self._alpha_amp * alpha_mod * self._alpha_weights[ch] * np.sin(2 * np.pi * self._alpha_freq * t + phase)
            beta  = self._beta_amp  * beta_mod  * self._beta_weights[ch]  * np.sin(2 * np.pi * self._beta_freq  * t + phase * 1.3)
            theta = self._alpha_amp * 0.5 * theta_mod * np.sin(2 * np.pi * 6.0 * t + phase * 0.7)
            white = self.rng.randn(n_samples) * self._noise_level
            pink  = np.cumsum(white) * 0.02
            pink -= np.mean(pink)
            data[ch] = gain * (alpha + beta + theta + pink)
        return data


# ─────────────────────────────────────────────────────────────────────────────
# Main reader class
# ─────────────────────────────────────────────────────────────────────────────

class EmotivReader:
    """Unified interface for reading EEG from the EPOC X dongle or synthetic.

    All data is returned as numpy arrays of shape ``(14, N)`` in microvolts.
    """

    def __init__(
        self,
        mode: str = "hid",
        serial_number: Optional[str] = None,
        is_research: bool = False,
        synthetic_seed: int = 42,
    ):
        """
        Args:
            mode:           ``"hid"`` for hardware dongle, ``"synthetic"`` for testing.
            serial_number:  Ignored for EPOC X (dongle serial auto-detected).
                            Kept for API compatibility.
            is_research:    Ignored for EPOC X. Kept for API compatibility.
            synthetic_seed: Random seed for a repeatable synthetic identity.
        """
        self.mode = ConnectionMode(mode)
        self.serial_number = serial_number
        self.is_research = is_research
        self.connected = False
        self.battery = 0

        # HID state
        self._hid_device = None
        self._cipher = None

        # Synthetic state
        self._synthetic = None
        self._synthetic_seed = synthetic_seed

        # Contact quality cache
        self.quality: Dict[str, float] = {ch: 0.0 for ch in EEG_CHANNELS}

        if self.mode == ConnectionMode.SYNTHETIC:
            self._synthetic = SyntheticGenerator(identity_seed=synthetic_seed)

    # ── Connection ───────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if self.mode == ConnectionMode.SYNTHETIC:
            logger.info("Synthetic mode — no hardware needed")
            self.connected = True
            return True
        return self._connect_dongle()

    def _connect_dongle(self) -> bool:
        """Open the EMOTIV USB dongle's EEG Signals endpoint."""
        try:
            import hid
        except ImportError:
            raise ImportError("hidapi is required.  Install with: pip install hidapi")

        devices = hid.enumerate(EMOTIV_VENDOR_ID, EMOTIV_PRODUCT_ID)
        if not devices:
            logger.error("No EMOTIV dongle found.  Is the USB receiver plugged in?")
            return False

        # Pick the "EEG Signals" interface (usage_page 0xFFFF, usage 0x0002)
        eeg_dev = None
        for d in devices:
            if d.get("usage_page") == 0xFFFF:
                eeg_dev = d
                break
        if eeg_dev is None:
            eeg_dev = max(devices, key=lambda d: d.get("interface_number", -1))

        dongle_serial = eeg_dev.get("serial_number", "")
        product_name  = eeg_dev.get("product_string", "EMOTIV")
        logger.info(f"Found dongle: {product_name}  (SN: {dongle_serial})")

        if not dongle_serial or len(dongle_serial) < 4:
            logger.error("Dongle serial number not readable.")
            return False

        self.serial_number = dongle_serial

        # AES setup — EPOC X uses dongle serial + XOR pre-processing
        from Crypto.Cipher import AES as _AES
        aes_key = generate_aes_key(dongle_serial)
        self._cipher = _AES.new(aes_key, _AES.MODE_ECB)
        logger.info(f"AES key ready (EPOC X, dongle SN: {dongle_serial})")

        # Open HID
        try:
            self._hid_device = hid.device()
            self._hid_device.open_path(eeg_dev["path"])
            self._hid_device.set_nonblocking(False)
            self.connected = True
            logger.info("Dongle connected — ready to read EEG")
            return True
        except Exception as e:
            logger.error(f"Failed to open dongle HID: {e}")
            return False

    def disconnect(self):
        if self._hid_device:
            try:
                self._hid_device.close()
            except Exception:
                pass
            self._hid_device = None
        self.connected = False
        logger.info("Disconnected")

    # ── Packet I/O ───────────────────────────────────────────────────────────

    def read_packet(self) -> Optional[np.ndarray]:
        """Read and decrypt one EEG sample (14 channels, µV).

        Motion packets are consumed silently and ``None`` is returned.
        """
        if not self.connected:
            return None

        if self.mode == ConnectionMode.SYNTHETIC:
            return self._synthetic.generate_samples(1)[:, 0]

        return self._read_dongle_packet()

    def _read_dongle_packet(self) -> Optional[np.ndarray]:
        try:
            raw = self._hid_device.read(PACKET_SIZE, timeout_ms=1000)
            if not raw:
                return None

            # EPOC X: XOR with 0x55 before AES decryption
            xored = bytes([b ^ 0x55 for b in raw])
            decrypted = self._cipher.decrypt(xored)

            # Skip motion / extra-data packets
            if is_motion_packet(decrypted):
                return None

            return parse_eeg_packet(decrypted)

        except Exception as e:
            logger.debug(f"Read error: {e}")
            return None

    # ── High-level reads ─────────────────────────────────────────────────────

    def read_seconds(
        self,
        duration: float,
        task_label: str = "baseline",
        callback=None,
    ) -> np.ndarray:
        """Capture *duration* seconds of EEG.

        Returns ``(14, N)`` numpy array in µV.

        Args:
            duration:   Recording length in seconds.
            task_label: Affects synthetic data patterns; ignored for hardware.
            callback:   Optional ``callback(elapsed, total)`` for progress.
        """
        if not self.connected:
            raise RuntimeError("Not connected")

        n_samples = int(duration * SAMPLE_RATE)

        # ── Synthetic path ──
        if self.mode == ConnectionMode.SYNTHETIC:
            if callback:
                chunk_size = SAMPLE_RATE
                chunks = []
                for i in range(0, n_samples, chunk_size):
                    remaining = min(chunk_size, n_samples - i)
                    chunks.append(self._synthetic.generate_samples(remaining, task_label))
                    callback(min(i + remaining, n_samples) / SAMPLE_RATE, duration)
                    time.sleep(0.01)
                return np.hstack(chunks)
            return self._synthetic.generate_samples(n_samples, task_label)

        # ── Hardware path ──
        collected = []
        start = time.time()
        last_cb = start

        while time.time() - start < duration:
            sample = self.read_packet()
            if sample is not None:
                collected.append(sample)

            if callback and time.time() - last_cb >= 1.0:
                callback(time.time() - start, duration)
                last_cb = time.time()

        if callback:
            callback(duration, duration)

        if not collected:
            logger.warning("No EEG data collected — is the headset turned on?")
            return np.zeros((NUM_CHANNELS, 0))

        return np.column_stack(collected)

    def stream(
        self,
        window_seconds: float = 2.0,
        task_label: str = "baseline",
    ) -> Generator[np.ndarray, None, None]:
        """Yield successive windows of EEG as ``(14, window_samples)`` arrays."""
        if not self.connected:
            raise RuntimeError("Not connected")

        window_samples = int(window_seconds * SAMPLE_RATE)

        while self.connected:
            if self.mode == ConnectionMode.SYNTHETIC:
                yield self._synthetic.generate_samples(window_samples, task_label)
                time.sleep(window_seconds * 0.9)
            else:
                buf = []
                t0 = time.time()
                while len(buf) < window_samples and time.time() - t0 < window_seconds * 2:
                    s = self.read_packet()
                    if s is not None:
                        buf.append(s)
                if buf:
                    yield np.column_stack(buf)

    def get_info(self) -> dict:
        return {
            "mode": self.mode.value,
            "connected": self.connected,
            "channels": EEG_CHANNELS,
            "num_channels": NUM_CHANNELS,
            "sample_rate": SAMPLE_RATE,
            "serial_number": self.serial_number,
            "battery": self.battery,
            "is_research": self.is_research,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing utilities
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(data: np.ndarray, low: float, high: float,
                    fs: int = SAMPLE_RATE, order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter for ``(channels, samples)`` or ``(samples,)``."""
    from scipy.signal import butter, filtfilt

    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, min(high / nyq, 0.99)], btype="band")

    if data.ndim == 1:
        return filtfilt(b, a, data)
    return np.array([filtfilt(b, a, data[ch]) for ch in range(data.shape[0])])


def notch_filter(data: np.ndarray, freq: float, bandwidth: float = 2.0,
                 fs: int = SAMPLE_RATE) -> np.ndarray:
    """IIR notch filter to remove line noise."""
    from scipy.signal import iirnotch, filtfilt

    w0 = freq / (fs / 2.0)
    if w0 >= 1.0:
        return data
    b, a = iirnotch(w0, freq / bandwidth)

    if data.ndim == 1:
        return filtfilt(b, a, data)
    return np.array([filtfilt(b, a, data[ch]) for ch in range(data.shape[0])])


def preprocess_eeg(data: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Standard preprocessing: bandpass 0.5–45 Hz + notch at 50 & 60 Hz."""
    if data.size == 0:
        return data
    if data.shape[-1] < 50:
        logger.warning(f"Too few samples ({data.shape[-1]}) for filtering")
        return data

    filtered = bandpass_filter(data, 0.5, 45.0, fs)
    if 50.0 < fs / 2.0:
        filtered = notch_filter(filtered, 50.0, fs=fs)
    if 60.0 < fs / 2.0:
        filtered = notch_filter(filtered, 60.0, fs=fs)
    return filtered
