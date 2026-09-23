# -*- coding: utf-8 -*-
"""
formant_filter.py: Vowel Formant Filters & Antiformant Notch modules.
Implements:
  - Parallel Biquad Formant Filter Bank (F1, F2, F3) with peak gain normalization (~0 dB)
  - Nasal Antiformant Notch Filter (zero pole cancellation)
  - Lip Radiation Filter (+6 dB/octave tilt)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.signal import lfilter


def biquad_resonator(x: np.ndarray, f_res: float, bandwidth: float, sr: int = 44100) -> np.ndarray:
    """
    2nd-order digital resonator with normalized ~0 dB peak gain at resonant frequency.
    """
    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2.0 * np.pi * f_res / sr

    b0 = 1.0 - r
    a1 = -2.0 * r * np.cos(theta)
    a2 = r ** 2

    y = np.zeros_like(x)
    y1, y2 = 0.0, 0.0
    for j in range(len(x)):
        out = b0 * x[j] - a1 * y1 - a2 * y2
        y[j] = out
        y2 = y1
        y1 = out
    return y


def parallel_vowel_filter(
    x: np.ndarray,
    f1: float,
    f2: float,
    f3: float = 2600.0,
    bw1: float = 80.0,
    bw2: float = 100.0,
    bw3: float = 120.0,
    sr: int = 44100,
) -> np.ndarray:
    """
    Parallel VCF bank: blends F1 (body), F2 (vowel color), and F3 (clarity).
    """
    s_f1 = biquad_resonator(x, f1, bw1, sr=sr)
    s_f2 = biquad_resonator(x, f2, bw2, sr=sr)
    s_f3 = biquad_resonator(x, f3, bw3, sr=sr)
    vowel_out = 1.0 * s_f1 + 0.65 * s_f2 + 0.30 * s_f3
    return vowel_out.astype(np.float32)


def nasal_notch_filter(x: np.ndarray, f_zero: float = 1400.0, bw: float = 150.0, sr: int = 44100) -> np.ndarray:
    """
    2nd-order notch filter simulating nasal antiformant zero attenuation.
    """
    r = np.exp(-np.pi * bw / sr)
    theta = 2.0 * np.pi * f_zero / sr

    b = [1.0, -2.0 * np.cos(theta), 1.0]
    a = [1.0, -2.0 * r * np.cos(theta), r ** 2]

    # DC normalization
    dc_gain = (2.0 - 2.0 * np.cos(theta)) / (1.0 - 2.0 * r * np.cos(theta) + r ** 2)
    if abs(dc_gain) > 1e-4:
        b = [val / dc_gain for val in b]

    return lfilter(b, a, x).astype(np.float32)


def lip_radiation_filter(x: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """
    First-order high-frequency pre-emphasis simulating open-air lip radiation.
    """
    y = np.zeros_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    # Blend with direct signal to preserve deep fundamental bass
    return (0.70 * y + 0.55 * x).astype(np.float32)
