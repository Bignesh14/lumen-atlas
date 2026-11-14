# Lumen Atlas - NASA Space Apps Challenge 2026 (Lunar Landing Site Navigator)
# AI-assisted code: written with Claude (Anthropic), reviewed and run by Bignesh Das.
# License: Apache-2.0
"""Sun and Earth directions in the Moon's Mean-Earth/Polar-Axis frame (MOON_ME_DE421).

This is the same body-fixed frame the LOLA DEMs use, so terrain and sky line up.
Positions are *apparent* (light-time + aberration), like JPL Horizons' airless
apparent azimuth/elevation.
"""
import numpy as np
from skyfield.api import Loader, PlanetaryConstants

from .config import KERNEL_DIR, R_MOON_M, SUN_RADIUS_KM
from .dem import ll_to_unit, local_frame


class Ephem:
    def __init__(self, kernel_dir=KERNEL_DIR):
        self.load = Loader(str(kernel_dir), verbose=False)
        self.ts = self.load.timescale(builtin=True)
        self.eph = self.load("de421.bsp")
        pc = PlanetaryConstants()
        pc.read_text(self.load("moon_080317.tf"))
        pc.read_text(self.load("pck00008.tpc"))
        pc.read_binary(self.load("moon_pa_de421_1900-2050.bpc"))
        self.frame = pc.build_frame_named("MOON_ME_DE421")
        self.moon, self.sun, self.earth = self.eph["moon"], self.eph["sun"], self.eph["earth"]

    def vectors_km(self, t):
        """Apparent Sun and Earth position vectors from the Moon's centre, in MOON_ME (km).
        Returns two arrays of shape (n, 3)."""
        at = self.moon.at(t)
        s = at.observe(self.sun).apparent().frame_xyz(self.frame).km
        e = at.observe(self.earth).apparent().frame_xyz(self.frame).km
        return np.asarray(s, dtype=np.float64).T.reshape(-1, 3), np.asarray(e, dtype=np.float64).T.reshape(-1, 3)


def topo_altaz(lat, lon, h_m, vec_km):
    """Azimuth (deg, clockwise from north) and elevation (deg, above the local
    horizontal plane of the reference sphere) of target vectors seen from a surface point."""
    u = ll_to_unit(lat, lon)
    e_hat, n_hat = local_frame(u)
    site_km = u * (R_MOON_M + h_m) / 1000.0
    v = np.asarray(vec_km) - site_km
    ve, vn, vu = v @ e_hat, v @ n_hat, v @ u
    az = np.degrees(np.arctan2(ve, vn)) % 360.0
    el = np.degrees(np.arctan2(vu, np.hypot(ve, vn)))
    dist = np.linalg.norm(v, axis=-1)
    return az, el, dist


def sun_angular_radius_deg(dist_km):
    return np.degrees(np.arcsin(SUN_RADIUS_KM / np.asarray(dist_km)))
