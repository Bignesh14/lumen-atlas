# Lumen Atlas - NASA Space Apps Challenge 2026 (Lunar Landing Site Navigator)
# AI-assisted code: written with Claude (Anthropic), reviewed and run by Bignesh Das.
# License: Apache-2.0
"""Readers for NASA LRO LOLA gridded elevation products (PDS3 IMG + LBL).

PolarDEM : LDEM_80S_20M / LDEM_80S_40M  (south polar stereographic, spherical)
CylDEM   : LDEM_64                       (global simple-cylindrical)

Heights are returned in metres relative to the 1737.4 km reference sphere
(HEIGHT = DN * SCALING_FACTOR, from the label).  Both grids are pixel-registered and
the *_PROJECTION_OFFSET values are the 0-based pixel-centre coordinate of the origin.
"""
import re
from pathlib import Path

import numpy as np

from .config import R_MOON_M


def parse_lbl(path):
    txt = Path(path).read_text(errors="replace")

    def num(key):
        m = re.search(rf"^\s*{key}\s*=\s*\"?\s*([-+0-9.eE]+)", txt, re.M)
        return float(m.group(1)) if m else None

    def word(key):
        m = re.search(rf"^\s*{key}\s*=\s*\"?([A-Z_ ]+?)\"?\s*$", txt, re.M)
        return m.group(1).strip() if m else None

    return {
        "lines": int(num("LINES")), "samples": int(num("LINE_SAMPLES")),
        "sample_type": word("SAMPLE_TYPE"), "sample_bits": int(num("SAMPLE_BITS")),
        "scaling": num("SCALING_FACTOR") or 1.0, "offset_radius_m": num("OFFSET"),
        "projection": word("MAP_PROJECTION_TYPE"),
        "map_scale": num("MAP_SCALE"), "map_resolution": num("MAP_RESOLUTION"),
        "line_off": num("LINE_PROJECTION_OFFSET"), "sample_off": num("SAMPLE_PROJECTION_OFFSET"),
        "center_lat": num("CENTER_LATITUDE"), "center_lon": num("CENTER_LONGITUDE"),
        "unit": word("UNIT"),
    }


def _open_img(img_path, meta):
    if meta["sample_bits"] != 16 or "LSB" not in (meta["sample_type"] or ""):
        raise ValueError(f"expected 16-bit LSB integers, got {meta['sample_type']} {meta['sample_bits']}")
    return np.memmap(img_path, dtype="<i2", mode="r", shape=(meta["lines"], meta["samples"]))


def ll_to_unit(lat_deg, lon_deg):
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def unit_to_ll(u):
    u = np.asarray(u, dtype=np.float64)
    lat = np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(u[..., 1], u[..., 0]))
    return lat, lon


def local_frame(u):
    """East and North unit vectors at surface direction u (shape (...,3)).
    Same formulas as the Metal kernel: E = norm(-uy, ux, 0), N = u x E."""
    u = np.asarray(u, dtype=np.float64)
    e = np.stack([-u[..., 1], u[..., 0], np.zeros_like(u[..., 0])], axis=-1)
    e /= np.linalg.norm(e, axis=-1, keepdims=True)
    n = np.cross(u, e)
    return e, n


def _bilinear(arr, row, col, scale):
    row = np.asarray(row, dtype=np.float64); col = np.asarray(col, dtype=np.float64)
    H, W = arr.shape
    ok = (row >= 0) & (col >= 0) & (row < H - 1) & (col < W - 1)
    r0 = np.where(ok, np.floor(row), 0).astype(np.int64)
    c0 = np.where(ok, np.floor(col), 0).astype(np.int64)
    fr = np.where(ok, row - r0, 0.0); fc = np.where(ok, col - c0, 0.0)
    h00 = arr[r0, c0].astype(np.float64); h01 = arr[r0, c0 + 1].astype(np.float64)
    h10 = arr[r0 + 1, c0].astype(np.float64); h11 = arr[r0 + 1, c0 + 1].astype(np.float64)
    h = ((h00 * (1 - fc) + h01 * fc) * (1 - fr) + (h10 * (1 - fc) + h11 * fc) * fr) * scale
    return np.where(ok, h, np.nan)


class PolarDEM:
    """South polar stereographic LOLA DEM (spherical, true scale at the pole).

    Forward:  rho = 2R tan(pi/4 + lat/2);  x = rho sin(lon);  y = rho cos(lon)
              col = sample_off + sx * a / scale ;  row = line_off - sy * b / scale
              where (a, b) = (x, y), or (y, x) if swap.
    The orientation flags (sx, sy, swap) are verified against known craters by
    scripts/00_selftest_dem.py and stored in out/orientation.json.
    """

    def __init__(self, img_path, lbl_path=None, orientation=(1.0, 1.0, False)):
        img_path = Path(img_path)
        lbl_path = Path(lbl_path) if lbl_path else img_path.with_suffix(".LBL")
        self.meta = m = parse_lbl(lbl_path)
        if "POLAR" not in (m["projection"] or ""):
            raise ValueError("not a polar stereographic product")
        if m["center_lat"] != -90:
            raise ValueError("this reader assumes the south pole product")
        self.R = (m["offset_radius_m"] or R_MOON_M)
        self.scale_m = m["map_scale"]           # m / pixel
        self.factor = m["scaling"]              # DN -> m
        self.line_off = m["line_off"]; self.sample_off = m["sample_off"]
        self.data = _open_img(img_path, m)
        self.H, self.W = self.data.shape
        self.set_orientation(*orientation)

    def set_orientation(self, sx, sy, swap):
        self.sx, self.sy, self.swap = float(sx), float(sy), bool(swap)

    # --- projections -------------------------------------------------------
    def ll_to_xy(self, lat, lon):
        lat = np.radians(np.asarray(lat, dtype=np.float64)); lon = np.radians(np.asarray(lon, dtype=np.float64))
        rho = 2 * self.R * np.tan(np.pi / 4 + lat / 2)
        return rho * np.sin(lon), rho * np.cos(lon)

    def xy_to_ll(self, x, y):
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        rho = np.hypot(x, y)
        lat = np.degrees(2 * np.arctan(rho / (2 * self.R)) - np.pi / 2)
        lon = np.degrees(np.arctan2(x, y))
        return lat, lon

    def xy_to_rc(self, x, y):
        a, b = (y, x) if self.swap else (x, y)
        col = self.sample_off + self.sx * np.asarray(a) / self.scale_m
        row = self.line_off - self.sy * np.asarray(b) / self.scale_m
        return row, col

    def rc_to_xy(self, row, col):
        a = (np.asarray(col, dtype=np.float64) - self.sample_off) * self.scale_m / self.sx
        b = (self.line_off - np.asarray(row, dtype=np.float64)) * self.scale_m / self.sy
        return (b, a) if self.swap else (a, b)

    def ll_to_rc(self, lat, lon):
        return self.xy_to_rc(*self.ll_to_xy(lat, lon))

    def rc_to_ll(self, row, col):
        return self.xy_to_ll(*self.rc_to_xy(row, col))

    # --- sampling ----------------------------------------------------------
    def height_rc(self, row, col):
        return _bilinear(self.data, row, col, self.factor)

    def height_ll(self, lat, lon):
        return self.height_rc(*self.ll_to_rc(lat, lon))

    def kernel_params(self):
        """Float parameters consumed by the Metal kernel (see horizon.py)."""
        return np.array([self.R, 1.0 / self.scale_m, self.line_off, self.sample_off,
                         self.sx, self.sy, 1.0 if self.swap else 0.0, 2.0 * self.R, self.factor],
                        dtype=np.float32)


class CylDEM:
    """Global simple-cylindrical LOLA DEM (e.g. LDEM_64, ~474 m/pixel at the equator).
    row = line_off - lat * res ;  col = sample_off + (lon - center_lon) * res  (0-based)."""

    def __init__(self, img_path, lbl_path=None):
        img_path = Path(img_path)
        lbl_path = Path(lbl_path) if lbl_path else img_path.with_suffix(".LBL")
        self.meta = m = parse_lbl(lbl_path)
        self.R = (m["offset_radius_m"] or R_MOON_M)
        self.res = m["map_resolution"]          # pixels / degree
        self.factor = m["scaling"]
        self.line_off = m["line_off"]; self.sample_off = m["sample_off"]
        self.clon = m["center_lon"] or 0.0
        self.data = _open_img(img_path, m)
        self.H, self.W = self.data.shape

    def ll_to_rc(self, lat, lon):
        lon = np.asarray(lon, dtype=np.float64)
        dlon = (lon - self.clon + 180.0) % 360.0 - 180.0   # wrap into [-180, 180) about the centre
        return self.line_off - np.asarray(lat, dtype=np.float64) * self.res, self.sample_off + dlon * self.res

    def height_ll(self, lat, lon):
        row, col = self.ll_to_rc(lat, lon)
        col = np.mod(col, self.W)      # wrap around the date line
        return _bilinear(self.data, row, np.minimum(col, self.W - 1.000001), self.factor)
