# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Small image helpers (no matplotlib): hillshade, colour ramps, block averaging."""
import numpy as np
from PIL import Image

RAMPS = {
    # value 0..1 -> colour.  "sun": night to full sun; "earth": no link to always linked
    "sun":   ["#0b1026", "#2c2463", "#7a2f6b", "#d0563c", "#f6b443", "#fff7d6"],
    "earth": ["#07121e", "#0e3550", "#13708a", "#2eb3b0", "#8fe8cf", "#effff9"],
    "both":  ["#0c0a18", "#3a1d55", "#7f2f7e", "#c95b8c", "#f6a38c", "#fff0e4"],
    "night": ["#fff7d6", "#f6b443", "#d0563c", "#7a2f6b", "#2c2463", "#0b1026"],  # short nights bright
}


def _hex(c):
    return [int(c[i:i + 2], 16) for i in (1, 3, 5)]


def colorize(v, ramp, alpha=None):
    stops = np.array([_hex(c) for c in RAMPS[ramp]], dtype=np.float32)
    x = np.linspace(0, 1, len(stops))
    v = np.clip(np.nan_to_num(v, nan=0.0), 0, 1)
    rgb = np.stack([np.interp(v, x, stops[:, k]) for k in range(3)], axis=-1)
    if alpha is not None:
        rgb = np.concatenate([rgb, (np.clip(alpha, 0, 1) * 255)[..., None]], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))


def hillshade(z, pixel_m, az_deg=315.0, alt_deg=35.0):
    gy, gx = np.gradient(z.astype(np.float32), pixel_m)
    slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    s = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(s, 0, 1)


def block_mean(a, f):
    if f == 1:
        return a
    H, W = a.shape[:2]
    H2, W2 = H // f, W // f
    a = a[:H2 * f, :W2 * f]
    return a.reshape(H2, f, W2, f, *a.shape[2:]).mean(axis=(1, 3))
