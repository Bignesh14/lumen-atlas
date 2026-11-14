# Lumen Atlas - NASA Space Apps Challenge 2026 (Lunar Landing Site Navigator)
# AI-assisted code: written with Claude (Anthropic), reviewed and run by Bignesh Das.
# License: Apache-2.0
"""From horizons to answers: per-pixel sunlight and Earth-visibility statistics.

For every observer and every hour of the analysis window:
  * Sun:   fraction of the solar disk above the terrain horizon (0..1)
  * Earth: Earth's centre above the terrain horizon (direct-to-Earth comms possible)
Accumulated per observer (8 numbers):
  0 mean solar-disk fraction        (= "average illumination", as in Mazarico 2011 / Barker 2021)
  1 fraction of hours sunlit         (disk >= 50 %)
  2 fraction of hours Earth visible
  3 fraction of hours sunlit AND Earth visible
  4 longest continuous darkness      (hours, disk < 50 %)
  5 longest continuous Earth outage  (hours)
  6 longest continuous sun+Earth window (hours)
  7 spare (0)
GPU: Metal kernel, one thread per observer, looping over time (no big temporaries).
CPU: float64 NumPy reference used by the referee.
"""
import numpy as np

from .config import R_MOON_M
from .dem import local_frame

N_STATS = 8

_STATS_SRC = r"""
    uint i = thread_position_in_grid.x;
    uint N = (uint)p_i[0];
    uint A = (uint)p_i[1];
    uint T = (uint)p_i[2];
    if (i >= N) return;
    float Rkm = p_f[0];
    float ux = obs[4*i], uy = obs[4*i+1], uz = obs[4*i+2];
    float rsite = Rkm + obs[4*i+3] * 0.001f;
    float ex = -uy, ey = ux;
    float en = metal::rsqrt(ex*ex + ey*ey);
    ex *= en; ey *= en;
    float nx = -uz*ey, ny = uz*ex, nz = ux*ey - uy*ex;
    const float RAD = 57.29577951308232f;
    const float PI = 3.14159265358979f;
    float binw = (float)A / 360.0f;

    float sumf = 0.0f; uint nsun = 0, nearth = 0, nboth = 0;
    uint rdark = 0, rnoe = 0, rboth = 0, mdark = 0, mnoe = 0, mboth = 0;
    for (uint t = 0; t < T; t++) {
        // --- Sun (direction is the same for every pixel: parallax < 0.001 deg) ---
        float sx = sun[4*t], sy = sun[4*t+1], sz = sun[4*t+2], srad = sun[4*t+3];
        float ve = sx*ex + sy*ey, vn = sx*nx + sy*ny + sz*nz, vu = sx*ux + sy*uy + sz*uz;
        float el = metal::atan2(vu, metal::sqrt(ve*ve + vn*vn)) * RAD;
        float az = metal::atan2(ve, vn) * RAD; if (az < 0.0f) az += 360.0f;
        float x = az * binw; float x0 = metal::floor(x); float fx = x - x0;
        uint j0 = ((uint)x0) % A, j1 = (j0 + 1) % A;
        float hz = hz_t[j0*N + i] * (1.0f - fx) + hz_t[j1*N + i] * fx;
        float d = (el - hz) / srad;
        float frac;
        if (d >= 1.0f) frac = 1.0f;
        else if (d <= -1.0f) frac = 0.0f;
        else frac = 1.0f - (metal::acos(d) - d*metal::sqrt(1.0f - d*d)) / PI;
        sumf += frac;
        bool lit = frac >= 0.5f;
        // --- Earth (topocentric: subtract the site position) ---
        float gx = earth[3*t] - rsite*ux, gy = earth[3*t+1] - rsite*uy, gz = earth[3*t+2] - rsite*uz;
        float we = gx*ex + gy*ey, wn = gx*nx + gy*ny + gz*nz, wu = gx*ux + gy*uy + gz*uz;
        float el2 = metal::atan2(wu, metal::sqrt(we*we + wn*wn)) * RAD;
        float az2 = metal::atan2(we, wn) * RAD; if (az2 < 0.0f) az2 += 360.0f;
        float y = az2 * binw; float y0 = metal::floor(y); float fy = y - y0;
        uint k0 = ((uint)y0) % A, k1 = (k0 + 1) % A;
        float hz2 = hz_t[k0*N + i] * (1.0f - fy) + hz_t[k1*N + i] * fy;
        bool see = el2 > hz2;
        bool both = lit && see;
        nsun += lit ? 1 : 0; nearth += see ? 1 : 0; nboth += both ? 1 : 0;
        rdark = lit ? 0 : rdark + 1;  mdark = metal::max(mdark, rdark);
        rnoe = see ? 0 : rnoe + 1;    mnoe = metal::max(mnoe, rnoe);
        rboth = both ? rboth + 1 : 0; mboth = metal::max(mboth, rboth);
    }
    float fT = (float)T;
    out[8*i+0] = sumf / fT;
    out[8*i+1] = (float)nsun / fT;
    out[8*i+2] = (float)nearth / fT;
    out[8*i+3] = (float)nboth / fT;
    out[8*i+4] = (float)mdark;
    out[8*i+5] = (float)mnoe;
    out[8*i+6] = (float)mboth;
    out[8*i+7] = 0.0f;
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import mlx.core as mx
        _kernel = mx.fast.metal_kernel(name="lumen_visibility",
                                       input_names=["hz_t", "obs", "sun", "earth", "p_f", "p_i"],
                                       output_names=["out"], source=_STATS_SRC)
    return _kernel


def sky_arrays(sun_km, earth_km, sun_radius_deg):
    """Pack the per-hour sky into float32 arrays for the kernel."""
    su = sun_km / np.linalg.norm(sun_km, axis=1, keepdims=True)
    sun = np.concatenate([su, np.asarray(sun_radius_deg)[:, None]], axis=1).astype(np.float32)
    return sun, np.asarray(earth_km, dtype=np.float32)


def stats_gpu(hz_t_mx, obs_tab, sun, earth, n_az=360):
    """hz_t_mx: MLX float32 array (n_az * N), azimuth-major (as produced by horizon_mx)."""
    import mlx.core as mx
    N = obs_tab.shape[0]
    k = _get_kernel()
    (res,) = k(inputs=[hz_t_mx, mx.array(obs_tab.astype(np.float32)), mx.array(sun), mx.array(earth),
                       mx.array(np.array([R_MOON_M / 1000.0], dtype=np.float32)),
                       mx.array(np.array([N, n_az, sun.shape[0]], dtype=np.int32))],
               grid=(N, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(N * N_STATS,)], output_dtypes=[mx.float32])
    mx.eval(res)
    return np.array(res).reshape(N, N_STATS)


def _longest_run(mask):
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def stats_ref(hz, obs_row, sun_km, earth_km, sun_radius_deg):
    """Float64 reference for ONE observer.  hz: (n_az,) horizon in degrees."""
    n_az = hz.shape[0]
    u = obs_row[:3] / np.linalg.norm(obs_row[:3])
    e_hat, n_hat = local_frame(u)
    rsite = (R_MOON_M + obs_row[3]) / 1000.0

    def altaz(v):
        ve, vn, vu = v @ e_hat, v @ n_hat, v @ u
        return np.degrees(np.arctan2(vu, np.hypot(ve, vn))), np.degrees(np.arctan2(ve, vn)) % 360.0

    def hz_at(az):
        x = az * n_az / 360.0; j0 = np.floor(x).astype(int) % n_az; f = x - np.floor(x)
        return hz[j0] * (1 - f) + hz[(j0 + 1) % n_az] * f

    su = sun_km / np.linalg.norm(sun_km, axis=1, keepdims=True)
    el, az = altaz(su)
    d = np.clip((el - hz_at(az)) / sun_radius_deg, -1, 1)
    frac = 1 - (np.arccos(d) - d * np.sqrt(1 - d * d)) / np.pi
    lit = frac >= 0.5
    el2, az2 = altaz(earth_km - rsite * u)
    see = el2 > hz_at(az2)
    both = lit & see
    return np.array([frac.mean(), lit.mean(), see.mean(), both.mean(),
                     _longest_run(~lit), _longest_run(~see), _longest_run(both), 0.0])
