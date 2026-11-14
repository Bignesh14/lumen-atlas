# Lumen Atlas - AI-assisted code (Claude, Anthropic); reviewed and run by Bignesh Das. Apache-2.0
"""Geometry self-test on a synthetic DEM (runs anywhere, no NASA data needed):
  1. projection round trips
  2. a float32 NumPy copy of the Metal kernel math agrees with the float64 reference
  3. the horizon peak points at the true great-circle bearing of a synthetic mountain
Run:  python -m tests.test_geometry
"""
import numpy as np

from lumen.dem import PolarDEM, _bilinear, ll_to_unit
from lumen.horizon import make_steps, azimuths, horizon_ref, observer_table


class FakePolar(PolarDEM):
    def __init__(self, n=3000, scale=20.0, orientation=(1.0, 1.0, False)):
        self.R = 1737400.0; self.scale_m = scale; self.factor = 0.5
        self.line_off = self.sample_off = n / 2 - 0.5
        self.H = self.W = n
        self.set_orientation(*orientation)
        rows, cols = np.mgrid[0:n, 0:n]
        lat, lon = self.rc_to_ll(rows, cols)
        # gaussian mountain, 3 km high, sigma 2 km
        mlat, mlon = -89.25, 40.0
        u = ll_to_unit(lat, lon); m = ll_to_unit(mlat, mlon)
        dist = self.R * np.arccos(np.clip(u @ m, -1, 1))
        h = 3000.0 * np.exp(-0.5 * (dist / 2000.0) ** 2)
        self.data = np.round(h / self.factor).astype(np.int16)
        self.mountain = (mlat, mlon)


def kernel_emulation(dem, obs, n_az=360, max_km=200.0):
    """Line-by-line float32 copy of _KERNEL_SRC (az_major layout)."""
    f = np.float32
    _, st = make_steps(dem.R, max_km=max_km); st = st.astype(f)
    _, az = azimuths(n_az); az = az.astype(f)
    p = dem.kernel_params()
    R, inv_scale, off_r, off_c, sx, sy, swap, twoR, factor = p
    out = np.empty((obs.shape[0], n_az), dtype=f)
    for i, (ux, uy, uz, ho) in enumerate(obs.astype(f)):
        ex, ey = -uy, ux
        en = f(1) / np.sqrt(ex * ex + ey * ey); ex *= en; ey *= en
        nx, ny, nz = -uz * ey, uz * ex, ux * ey - uy * ex
        for j in range(n_az):
            ca, sa = az[j]
            dx, dy, dz = ca * nx + sa * ex, ca * ny + sa * ey, ca * nz
            best = f(-1e30)
            for c, s, omc in st:
                px, py, pz = c * ux + s * dx, c * uy + s * dy, c * uz + s * dz
                q = twoR / (f(1) - pz)
                X, Y = py * q, px * q
                a, b = (Y, X) if swap > 0.5 else (X, Y)
                col = off_c + sx * a * inv_scale; row = off_r - sy * b * inv_scale
                if col < 0 or row < 0 or col >= dem.W - 1 or row >= dem.H - 1:
                    break
                h = f(_bilinear(dem.data, row, col, factor))
                rt = R + h
                best = max(best, ((h - ho) - rt * omc) / (rt * s))
            out[i, j] = np.degrees(np.arctan(best))
    return out


def main():
    for orient in [(1, 1, False), (-1, 1, True)]:
        dem = FakePolar(orientation=orient)
        # 1. round trips
        lat = np.array([-89.9, -89.0, -88.2]); lon = np.array([0.0, 123.4, -77.7])
        r, c = dem.ll_to_rc(lat, lon); lat2, lon2 = dem.rc_to_ll(r, c)
        assert np.allclose(lat, lat2, atol=1e-9) and np.allclose(((lon - lon2 + 180) % 360) - 180, 0, atol=1e-9)

        # 2/3. horizon from an observer 0.4 deg (~12 km) away from the mountain
        olat, olon = -89.55, 20.0
        obs = observer_table(dem, [olat], [olon])
        n_az = 72
        ref = horizon_ref(dem, olat, olon, obs[0, 3], n_az=n_az, max_km=40)
        emu = kernel_emulation(dem, obs, n_az=n_az, max_km=40)[0]
        diff = np.nanmax(np.abs(ref - emu))
        # true initial bearing to the mountain
        p1, l1 = np.radians([olat, olon]); p2, l2 = np.radians(dem.mountain)
        brg = np.degrees(np.arctan2(np.sin(l2 - l1) * np.cos(p2),
                                    np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(l2 - l1))) % 360
        peak = azimuths(n_az)[0][np.argmax(ref)]
        dpk = abs((peak - brg + 180) % 360 - 180)
        print(f"orientation={orient}: kernel-vs-ref max |diff| = {diff:.5f} deg ; "
              f"peak az {peak:.1f} vs bearing {brg:.1f} (|d|={dpk:.1f}) ; max horizon {ref.max():.2f} deg")
        assert diff < 0.02, diff
        assert dpk <= 360 / n_az, dpk
    print("geometry self-test PASSED")


if __name__ == "__main__":
    main()
