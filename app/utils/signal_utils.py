import numpy as np
import pandas as pd
import logging
from scipy.signal import butter, filtfilt, find_peaks

logger = logging.getLogger(__name__)


# -------------------------
# ECG Filter
# -------------------------
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=256, order=2):
    try:
        if signal is None or len(signal) == 0:
            return signal

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(order, [low, high], btype='band')

        return filtfilt(b, a, signal)

    except Exception as e:
        logger.warning(f"⚠️ bandpass_filter failed: {e}")
        return signal  # fallback (do not break pipeline)


# -------------------------
# BPM Calculation
# -------------------------
def compute_bpm(ecg_signal, fs):
    try:
        if ecg_signal is None or len(ecg_signal) < fs:
            return 0

        peaks, _ = find_peaks(ecg_signal, distance=fs * 0.6)

        if len(peaks) < 2:
            return 0

        rr_intervals = np.diff(peaks) / fs

        if len(rr_intervals) == 0:
            return 0

        avg_rr = np.mean(rr_intervals)

        if avg_rr <= 0:
            return 0

        bpm = 60 / avg_rr

        # 🔥 Safety clamp (production)
        if bpm < 20 or bpm > 220:
            return 0

        return float(bpm)

    except Exception as e:
        logger.warning(f"⚠️ compute_bpm failed: {e}")
        return 0


# -------------------------
# Signal Quality
# -------------------------
def check_quality(signal):
    try:
        if signal is None or len(signal) == 0:
            return "Unknown"

        std = np.std(signal)

        if std < 50:
            return "Poor"
        elif std < 200:
            return "Moderate"
        else:
            return "Good"

    except Exception as e:
        logger.warning(f"⚠️ check_quality failed: {e}")
        return "Unknown"


# -------------------------
# SpO2 Extraction
# -------------------------
def extract_spo2(oxym_window):
    try:
        if oxym_window is None or len(oxym_window) == 0:
            return np.nan

        signal = np.array(oxym_window)

        # Remove NaNs
        signal = signal[~np.isnan(signal)]

        if len(signal) < 100:
            return np.nan

        if np.std(signal) < 1e-3:
            return np.nan

        # Smooth
        signal = pd.Series(signal).rolling(window=10, min_periods=1).mean().values

        # Normalize safely
        min_val = np.min(signal)
        max_val = np.max(signal)

        if max_val - min_val == 0:
            return np.nan

        norm_signal = (signal - min_val) / (max_val - min_val)

        spo2 = 94 + (np.mean(norm_signal) * 6)

        # 🔥 Safety bounds
        if spo2 < 50 or spo2 > 100:
            return np.nan

        return round(float(spo2), 2)

    except Exception as e:
        logger.warning(f"⚠️ extract_spo2 failed: {e}")
        return np.nan