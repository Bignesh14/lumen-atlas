# Lumen Atlas - NASA Space Apps Challenge 2026 (Lunar Landing Site Navigator)
# AI-assisted code: written with Claude (Anthropic), reviewed and run by Bignesh Das.
# License: Apache-2.0
"""Terrain horizon: for every observer pixel and every azimuth, the elevation angle of
the highest terrain seen along that direction (Moon curvature included, no atmosphere).

Two independent implementations:
  * horizon_gpu  - custom Metal kernel through MLX (mx.fast.metal_kernel), float32
  * horizon_ref  - NumPy float64 reference using a different math path
                   (great-circle destination formula + exact vector elevation angle)
The referee script checks they agree before any result is used.
"""
import numpy as np

from .config import MAST_M
from .dem import ll_to_unit


def make_steps(R, max_km=200.0, near_m=20.0, near_until_m=1000.0, growth=1.015):
    """Ray-march distances: every 20 m out to 1 km, then growing 1.5 % per step to max_km.
    Returns d (m) and the per-step table (cos t, sin t, 1-cos t) with t = d / R."""
    d = []
    x = near_m
    while x <= near_until_m + 1e-9:
        d.append(x); x += near_m
    while x <= max_km * 1000.0:
        d.append(x); x *= growth
    d = np.array(d, dtype=np.float64)
    t = d / R
    table = np.stack([np.cos(t), np.sin(t), 2.0 * np.sin(t / 2.0) ** 2], axis=1)
    return d, table


def azimuths(n_az=360):
    az = np.arange(n_az, dtype=np.float64) * (360.0 / n_az)
    return az, np.stack([np.cos(np.radians(az)), np.sin(np.radians(az))], axis=1)


def observer_table(dem, lat, lon, mast_m=MAST_M):
    """(ux, uy, uz, h_obs) per observer; h_obs = ground height + mast (m)."""
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64)); lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    u = ll_to_unit(lat, lon)
    h = dem.height_ll(lat, lon) + mast_m
    return np.concatenate([u, h[:, None]], axis=1)


# --------------------------------------------------------------------------------------
# Metal kernel.  One thread = one (observer, azimuth) ray.
#   layout "az_major":    neighbouring threads = neighbouring observers, same azimuth
#                         -> parallel rays, coherent DEM reads (cache friendly)
#   layout "pixel_major": neighbouring threads = same observer, neighbouring azimuths
#                         -> rays fan out, scattered DEM reads
# Both write out[az * N + obs]; the referee checks they are identical.
# --------------------------------------------------------------------------------------
_KERNEL_SRC = r"""
    uint gid = thread_position_in_grid.x;
    uint N = (uint)p_i[0];
    uint A = (uint)p_i[1];
    uint S = (uint)p_i[2];
    uint H = (uint)p_i[3];
    uint W = (uint)p_i[4];
    uint layout = (uint)p_i[5];
    if (gid >= N * A) return;
    uint i, j;
    if (layout == 0) { j = gid / N; i = gid - j * N; }   // az_major
    else             { i = gid / A; j = gid - i * A; }   // pixel_major

    float R = p_f[0], inv_scale = p_f[1], off_r = p_f[2], off_c = p_f[3];
    float sx = p_f[4], sy = p_f[5], swap = p_f[6], twoR = p_f[7], factor = p_f[8];

    float ux = obs[4*i], uy = obs[4*i+1], uz = obs[4*i+2], ho = obs[4*i+3];
    // local East = norm(-uy, ux, 0), North = u x East  (same as lumen/dem.py local_frame)
    float ex = -uy, ey = ux;
    float en = metal::rsqrt(ex*ex + ey*ey);
    ex *= en; ey *= en;
    float nx = -uz*ey, ny = uz*ex, nz = ux*ey - uy*ex;
    float ca = az[2*j], sa = az[2*j+1];
    float dx = ca*nx + sa*ex, dy = ca*ny + sa*ey, dz = ca*nz;

    float best = -1.0e30f;
    for (uint k = 0; k < S; k++) {
        float c = st[3*k], s = st[3*k+1], omc = st[3*k+2];
        float px = c*ux + s*dx, py = c*uy + s*dy, pz = c*uz + s*dz;   // point on the great circle
        float q = twoR / (1.0f - pz);                                // south polar stereographic
        float X = py*q, Y = px*q;
        float a = (swap > 0.5f) ? Y : X;
        float b = (swap > 0.5f) ? X : Y;
        float col = off_c + sx*a*inv_scale;
        float row = off_r - sy*b*inv_scale;
        if (col < 0.0f || row < 0.0f || col >= (float)(W - 1) || row >= (float)(H - 1)) break;
        uint c0 = (uint)col, r0 = (uint)row;
        float fc = col - (float)c0, fr = row - (float)r0;
        ulong base = (ulong)r0 * (ulong)W + (ulong)c0;
        float h00 = (float)dem[base],     h01 = (float)dem[base + 1];
        float h10 = (float)dem[base + W], h11 = (float)dem[base + W + 1];
        float h = ((h00*(1.0f - fc) + h01*fc)*(1.0f - fr) + (h10*(1.0f - fc) + h11*fc)*fr) * factor;
        float rt = R + h;
        // tan(elevation) = (rt cos t - ro) / (rt sin t), written without cancellation:
        float slope = ((h - ho) - rt*omc) / (rt*s);
        best = metal::max(best, slope);
    }
    out[gid] = metal::precise::atan(best) * 57.29577951308232f;
"""

_kernel = None
_dem_cache = {}


def _get_kernel():
    global _kernel
    if _kernel is None:
        import mlx.core as mx
        _kernel = mx.fast.metal_kernel(
            name="lumen_horizon",
            input_names=["dem", "obs", "az", "st", "p_f", "p_i"],
            output_names=["out"],
            source=_KERNEL_SRC,
        )
    return _kernel


def dem_on_gpu(dem):
    """Copy the int16 DEM into an MLX array once (1.85 GB for the 20 m map)."""
    import mlx.core as mx
    key = id(dem)
    if key not in _dem_cache:
        _dem_cache[key] = mx.array(np.ascontiguousarray(dem.data))
    return _dem_cache[key]


def horizon_mx(dem, obs_tab, n_az=360, max_km=200.0):
    """Like horizon_gpu but keeps the result on the GPU: MLX float32 array (n_az * N),
    azimuth-major (index = az * N + observer).  Used to feed the visibility kernel directly."""
    import mlx.core as mx
    kernel = _get_kernel()
    _, st = make_steps(dem.R, max_km=max_km)
    _, az = azimuths(n_az)
    N = obs_tab.shape[0]
    (res,) = kernel(inputs=[dem_on_gpu(dem), mx.array(obs_tab.astype(np.float32)),
                            mx.array(az.astype(np.float32)), mx.array(st.astype(np.float32)),
                            mx.array(dem.kernel_params()),
                            mx.array(np.array([N, n_az, st.shape[0], dem.H, dem.W, 0], dtype=np.int32))],
                    grid=(N * n_az, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(N * n_az,)], output_dtypes=[mx.float32])
    return res


def horizon_gpu(dem, obs_tab, n_az=360, max_km=200.0, layout="az_major", chunk=262144):
    """Horizon elevation (deg) for each observer row of obs_tab -> array (N, n_az) float32."""
    import mlx.core as mx
    kernel = _get_kernel()
    _, st = make_steps(dem.R, max_km=max_km)
    _, az = azimuths(n_az)
    g_dem = dem_on_gpu(dem)
    g_az = mx.array(az.astype(np.float32)); g_st = mx.array(st.astype(np.float32))
    g_pf = mx.array(dem.kernel_params())
    lay = 0 if layout == "az_major" else 1
    out = np.empty((obs_tab.shape[0], n_az), dtype=np.float32)
    for s0 in range(0, obs_tab.shape[0], chunk):
        sub = obs_tab[s0:s0 + chunk]
        N = sub.shape[0]
        g_obs = mx.array(sub.astype(np.float32))
        g_pi = mx.array(np.array([N, n_az, st.shape[0], dem.H, dem.W, lay], dtype=np.int32))
        (res,) = kernel(inputs=[g_dem, g_obs, g_az, g_st, g_pf, g_pi],
                        grid=(N * n_az, 1, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(N * n_az,)], output_dtypes=[mx.float32])
        mx.eval(res)
        r = np.array(res)
        out[s0:s0 + N] = r.reshape(n_az, N).T if lay == 0 else r.reshape(N, n_az)
    return out


def horizon_ref(dem, lat, lon, h_obs, n_az=360, max_km=200.0):
    """Float64 reference for one observer -> array (n_az,).
    Destination points from the spherical 'destination given bearing and distance'
    formula, heights by bilinear lookup through lat/lon, elevation angle from vectors."""
    R = dem.R
    d, _ = make_steps(R, max_km=max_km)
    az, _ = azimuths(n_az)
    phi1, lam1 = np.radians(lat), np.radians(lon)
    delta = d / R
    o = ll_to_unit(lat, lon) * (R + h_obs)
    up = ll_to_unit(lat, lon)
    out = np.empty(n_az)
    for j, a in enumerate(np.radians(az)):
        phi2 = np.arcsin(np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(a))
        lam2 = lam1 + np.arctan2(np.sin(a) * np.sin(delta) * np.cos(phi1),
                                 np.cos(delta) - np.sin(phi1) * np.sin(phi2))
        h = dem.height_ll(np.degrees(phi2), np.degrees(lam2))
        bad = np.isnan(h)
        if bad.any():                      # stop at the first sample outside the map, like the kernel
            first = np.argmax(bad)
            h = h[:first]; phi2 = phi2[:first]; lam2 = lam2[:first]
        if h.size == 0:
            out[j] = np.nan; continue
        t = ll_to_unit(np.degrees(phi2), np.degrees(lam2)) * (R + h)[:, None]
        v = t - o
        el = np.degrees(np.arcsin((v @ up) / np.linalg.norm(v, axis=1)))
        out[j] = el.max()
    return out
