// Lumen Atlas - NASA Space Apps Challenge 2026 (Lunar Landing Site Navigator)
// AI-assisted code: written with Claude (Anthropic), reviewed and run by Bignesh Das. Apache-2.0
//
// Everything here runs in the browser from precomputed NASA-derived files:
//   assets/ephem.bin       hourly Sun/Earth vectors in the Moon's ME frame (from SPICE DE421)
//   assets/sites.json      famous sites with 360-direction terrain horizons
//   assets/regions/...     every-pixel maps + horizon grids for the 9 Artemis III regions
'use strict';

const DEG = Math.PI / 180, RAD = 180 / Math.PI;
const R_KM = 1737.4;
const $ = (id) => document.getElementById(id);
const A = 360;                              // horizon azimuth bins

const S = {                                  // app state
  ephem: null, eStart: 0, eHours: 0,
  index: null, sites: [], overview: null, overviewImg: null,
  region: null, regionData: {}, layer: 'both',
  site: null, t: 0, mode: 'explore', compare: [], planSpots: [], playing: null,
};

// ---------------------------------------------------------------- geometry
function unitLL(lat, lon) {
  const a = lat * DEG, b = lon * DEG;
  return [Math.cos(a) * Math.cos(b), Math.cos(a) * Math.sin(b), Math.sin(a)];
}
function frame(u) {                          // same east/north as the Python + Metal code
  let ex = -u[1], ey = u[0]; const n = Math.hypot(ex, ey); ex /= n; ey /= n;
  return { u, e: [ex, ey, 0], n: [-u[2] * ey, u[2] * ex, u[0] * ey - u[1] * ex] };
}
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function altaz(v, f) {
  const ve = dot(v, f.e), vn = dot(v, f.n), vu = dot(v, f.u);
  let az = Math.atan2(ve, vn) * RAD; if (az < 0) az += 360;
  return { el: Math.atan2(vu, Math.hypot(ve, vn)) * RAD, az };
}
function hzAt(hz, az) {
  const x = az * A / 360, x0 = Math.floor(x), f = x - x0;
  return hz[((x0 % A) + A) % A] * (1 - f) + hz[(x0 + 1) % A] * f;
}
function diskFrac(d) {                       // fraction of a disk above a straight edge at offset d (in radii)
  if (d >= 1) return 1; if (d <= -1) return 0;
  return 1 - (Math.acos(d) - d * Math.sqrt(1 - d * d)) / Math.PI;
}
function distKm(a, b) {
  const ua = unitLL(a.lat, a.lon), ub = unitLL(b.lat, b.lon);
  return R_KM * Math.acos(Math.min(1, Math.max(-1, dot(ua, ub))));
}

// polar stereographic (south), same as lumen/dem.py
function ll2rc(lat, lon) {
  const p = S.index.projection, rho = 2 * p.R * Math.tan(Math.PI / 4 + lat * DEG / 2);
  const x = rho * Math.sin(lon * DEG), y = rho * Math.cos(lon * DEG);
  const a = p.swap ? y : x, b = p.swap ? x : y;
  return [p.line_off - p.sy * b / p.scale_m, p.sample_off + p.sx * a / p.scale_m];
}
function rc2ll(r, c) {
  const p = S.index.projection;
  const a = (c - p.sample_off) * p.scale_m / p.sx, b = (p.line_off - r) * p.scale_m / p.sy;
  const x = p.swap ? b : a, y = p.swap ? a : b;
  return [(2 * Math.atan(Math.hypot(x, y) / (2 * p.R)) - Math.PI / 2) * RAD, Math.atan2(x, y) * RAD];
}

// ---------------------------------------------------------------- sky
function sky(t) {                            // t in ms (UTC) -> {sun unit, sunR, earth km}
  const x = (t - S.eStart) / 3.6e6;
  const i = Math.max(0, Math.min(S.eHours - 2, Math.floor(x))), f = Math.max(0, Math.min(1, x - i));
  const E = S.ephem, a = i * 7, b = a + 7, g = (k) => E[a + k] * (1 - f) + E[b + k] * f;
  return { sun: [g(0), g(1), g(2)], sunR: g(3), earth: [g(4), g(5), g(6)] };
}
function at(site, t) {
  const s = sky(t), f = site.f;
  const sp = altaz(s.sun, f);
  const r = R_KM + site.h / 1000;
  const ep = altaz([s.earth[0] - r * f.u[0], s.earth[1] - r * f.u[1], s.earth[2] - r * f.u[2]], f);
  const sh = hzAt(site.hz, sp.az), eh = hzAt(site.hz, ep.az);
  const frac = diskFrac((sp.el - sh) / s.sunR);
  return { sunEl: sp.el, sunAz: sp.az, sunHz: sh, frac, lit: frac >= 0.5, sunR: s.sunR,
           earthEl: ep.el, earthAz: ep.az, earthHz: eh, see: ep.el > eh };
}
function nextChange(site, t0, key, maxH = 24 * 60) {
  const v0 = at(site, t0)[key];
  let prev = t0;
  for (let h = 1; h <= maxH; h++) {
    const t = t0 + h * 3.6e6;
    if (t > S.eStart + (S.eHours - 2) * 3.6e6) return null;
    if (at(site, t)[key] !== v0) {           // refine to ~1 minute
      let lo = prev, hi = t;
      for (let k = 0; k < 6; k++) { const m = (lo + hi) / 2; if (at(site, m)[key] === v0) lo = m; else hi = m; }
      return hi;
    }
    prev = t;
  }
  return Infinity;
}
function mission(site, t0, days) {           // hourly over the stay
  const n = Math.max(1, Math.round(days * 24));
  let lit = 0, see = 0, run = 0, maxDark = 0, firstDark = null, firstNoEarth = null;
  for (let h = 0; h < n; h++) {
    const t = t0 + h * 3.6e6, x = at(site, t);
    if (x.lit) lit++; else if (firstDark === null) firstDark = t;
    if (x.see) see++; else if (firstNoEarth === null) firstNoEarth = t;
    run = x.lit ? 0 : run + 1; maxDark = Math.max(maxDark, run);
  }
  return { sunPct: 100 * lit / n, earthPct: 100 * see / n, maxDark, firstDark, firstNoEarth, landEarth: at(site, t0).see };
}

// ---------------------------------------------------------------- formatting
const pad = (n) => String(n).padStart(2, '0');
function fmtT(t) { const d = new Date(t); return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`; }
function fmtDur(ms) {
  if (ms === Infinity) return 'more than 60 days';
  const h = ms / 3.6e6;
  if (h < 1) return `${Math.max(1, Math.round(h * 60))} min`;
  if (h < 48) return `${Math.floor(h)} h ${Math.round((h % 1) * 60)} min`;
  return `${Math.floor(h / 24)} d ${Math.round(h % 24)} h`;
}
const vs = (el) => `${Math.abs(el).toFixed(2)}° ${el >= 0 ? 'above' : 'below'} a smooth horizon`;
const fmtLL = (lat, lon) => `${Math.abs(lat).toFixed(4)}° ${lat < 0 ? 'S' : 'N'}, ${Math.abs(lon).toFixed(4)}° ${lon < 0 ? 'W' : 'E'}`;

// ---------------------------------------------------------------- loading
async function getJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); }
async function getBin(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.arrayBuffer(); }
function loadImg(u) { return new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => bad(new Error(u)); i.src = u; }); }

async function init() {
  const [em, eb, idx, sites, ov] = await Promise.all([
    getJSON('assets/ephem.json'), getBin('assets/ephem.bin'), getJSON('assets/regions/index.json'),
    getJSON('assets/sites.json'), getJSON('assets/overview.json')]);
  S.ephem = new Float32Array(eb); S.eStart = Date.parse(em.start_utc); S.eHours = em.hours;
  S.index = idx; S.sites = sites; S.overview = ov; S.overviewImg = await loadImg('assets/overview.png');
  const now = Date.now(), lo = S.eStart, hi = S.eStart + (S.eHours - 24 * 61) * 3.6e6;
  S.t = Math.floor(Math.min(hi, Math.max(lo, now)) / 3.6e6) * 3.6e6;
  buildSiteMenu();
  wire();
  const first = Object.values(idx.regions).find((r) => r.key === 'Malapert_Massif') || Object.values(idx.regions)[0];
  setMode(S.mode);
  await selectRegionBest(first.key, 'best_both');
}

function siteFrom(name, lat, lon, h, hz, extra = {}) {
  return Object.assign({ name, lat, lon, h, hz, f: frame(unitLL(lat, lon)) }, extra);
}

function buildSiteMenu() {
  const sel = $('sitesel'); sel.innerHTML = '';
  const g1 = document.createElement('optgroup'); g1.label = 'Artemis III regions: best spot for Sun and Earth';
  for (const r of Object.values(S.index.regions)) {
    const o = document.createElement('option'); o.value = 'R:' + r.key; o.textContent = r.name; g1.appendChild(o);
  }
  const g2 = document.createElement('optgroup'); g2.label = 'Landing sites, past and planned';
  S.sites.forEach((s, i) => { const o = document.createElement('option'); o.value = 'S:' + i; o.textContent = s.name; g2.appendChild(o); });
  sel.append(g1, g2);
}

async function regionData(key) {
  if (S.regionData[key]) return S.regionData[key];
  const meta = S.index.regions[key], base = `assets/regions/${key}/`;
  const [hz, hzh, st, ...imgs] = await Promise.all([getBin(base + 'hz.bin'), getBin(base + 'hzh.bin'), getBin(base + 'stats.bin'),
    ...['hillshade', 'sun', 'earth', 'both', 'night'].map((n) => loadImg(base + n + '.png'))]);
  const d = { meta, hz: new Uint8Array(hz), hzh: new Int16Array(hzh), stats: new Uint8Array(st),
              img: { hillshade: imgs[0], sun: imgs[1], earth: imgs[2], both: imgs[3], night: imgs[4] } };
  S.regionData[key] = d; return d;
}

// grid point (horizon sample) -> site
function gridSite(key, gi, gj) {
  const d = S.regionData[key], m = d.meta, g = m.grid, b = m.box;
  const k = gi * g.W + gj;
  const r = b.r0 + gi * g.g * b.stride, c = b.c0 + gj * g.g * b.stride;
  const [lat, lon] = rc2ll(r, c);
  const hz = new Float32Array(A);
  for (let a = 0; a < A; a++) hz[a] = d.hz[k * A + a] / 10 - 12.8;
  return siteFrom(`${m.name}`, lat, lon, d.hzh[k], hz, { region: key, gi, gj, spacing: g.g * b.stride * S.index.projection.scale_m });
}
function nearestGrid(key, lat, lon) {
  const m = S.index.regions[key], [r, c] = ll2rc(lat, lon);
  const gi = Math.round((r - m.box.r0) / (m.grid.g * m.box.stride)), gj = Math.round((c - m.box.c0) / (m.grid.g * m.box.stride));
  return [Math.max(0, Math.min(m.grid.H - 1, gi)), Math.max(0, Math.min(m.grid.W - 1, gj))];
}

async function selectRegionBest(key, which) {
  await regionData(key);
  const p = S.index.regions[key].summary[which];
  const [gi, gj] = nearestGrid(key, p.lat, p.lon);
  S.region = key;
  setSite(gridSite(key, gi, gj), `best spot for ${which === 'best_both' ? 'Sun and Earth together' : 'sunlight'}`);
  $('sitesel').value = 'R:' + key;
  showRegion(true);
}

function setSite(site, label) {
  S.site = site; site.label = label || '';
  S.planSpots = [];
  render();
}

// ---------------------------------------------------------------- UI wiring
function wire() {
  $('sitesel').addEventListener('change', async (e) => {
    const v = e.target.value;
    if (v.startsWith('R:')) return selectRegionBest(v.slice(2), 'best_both');
    const s = S.sites[+v.slice(2)];
    const site = siteFrom(s.name, s.lat, s.lon, s.height_m, Float32Array.from(s.horizon), { famous: s });
    const reg = Object.values(S.index.regions).find((r) => insideRegion(r, s.lat, s.lon));
    S.region = null; showRegion(false);
    setSite(site, '');
    if (reg) { S.region = reg.key; await regionData(reg.key); showRegion(true); render(); }
  });
  document.querySelectorAll('.step').forEach((b) => b.addEventListener('click', () => setTime(S.t + (+b.dataset.step) * 3.6e6)));
  $('when').addEventListener('change', (e) => { const t = Date.parse(e.target.value + ':00Z'); if (!isNaN(t)) setTime(t); });
  $('play').addEventListener('click', togglePlay);
  $('mode-explore').addEventListener('click', () => setMode('explore'));
  $('mode-plan').addEventListener('click', () => setMode('plan'));
  $('back').addEventListener('click', () => { S.region = null; showRegion(false); render(); });
  document.querySelectorAll('#layers button').forEach((b) => b.addEventListener('click', () => {
    S.layer = b.dataset.layer;
    document.querySelectorAll('#layers button').forEach((x) => { x.classList.toggle('on', x === b); x.setAttribute('aria-selected', x === b); });
    drawMap();
  }));
  ['p-days', 'p-sun', 'p-earth', 'p-landearth', 'p-slip'].forEach((id) => $(id).addEventListener('change', renderPlan));
  $('pin').addEventListener('click', pinScenario);
  const map = $('map');
  map.addEventListener('click', onMapClick);
  map.addEventListener('mousemove', onMapHover);
  map.addEventListener('mouseleave', () => { $('tip').hidden = true; });
  $('tl').addEventListener('click', (e) => {
    const r = e.target.getBoundingClientRect(), x = (e.clientX - r.left) / r.width;
    setTime(Math.round((S.t + x * 30 * 24 * 3.6e6) / 3.6e6) * 3.6e6);
  });
  window.addEventListener('resize', () => render());
}
function setMode(m) {
  S.mode = m;
  $('mode-explore').setAttribute('aria-pressed', m === 'explore'); $('mode-explore').classList.toggle('on', m === 'explore');
  $('mode-plan').setAttribute('aria-pressed', m === 'plan'); $('mode-plan').classList.toggle('on', m === 'plan');
  $('plan').hidden = m !== 'plan'; $('explain').hidden = m === 'plan';
  render();
}
function setTime(t) {
  const lo = S.eStart, hi = S.eStart + (S.eHours - 24 * 61) * 3.6e6;
  S.t = Math.min(hi, Math.max(lo, t)); render();
}
function togglePlay() {
  const b = $('play');
  if (S.playing) { clearInterval(S.playing); S.playing = null; b.textContent = 'Play'; b.setAttribute('aria-pressed', 'false'); return; }
  b.textContent = 'Pause'; b.setAttribute('aria-pressed', 'true');
  S.playing = setInterval(() => setTime(S.t + 3 * 3.6e6), 120);
}
function showRegion(on) {
  $('back').hidden = !on; $('layers').hidden = !on; $('legend').hidden = !on;
  $('maptitle').textContent = on ? S.index.regions[S.region].name : 'South pole, Artemis III candidate regions';
  $('mapnote').textContent = on
    ? 'Every 20 m pixel, 2027–2028. Click any spot to stand there. Faded areas are outside the official region.'
    : 'Click a region to see every 20 m pixel. Colours show 2027–2028.';
}

// ---------------------------------------------------------------- render
function render() {
  if (!S.site) return;
  const d = new Date(S.t);
  $('when').value = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:00`;
  renderNow(); drawPano(); drawMap(); drawTimeline();
  if (S.mode === 'plan') renderPlan(); else renderExplain();
}

function renderNow() {
  const s = S.site, x = at(s, S.t);
  $('sitename').textContent = s.name + (s.label ? `: ${s.label}` : '');
  let c = `${fmtLL(s.lat, s.lon)}, ground ${Math.round(s.h)} m`;
  if (s.spacing) c += `. Horizon from the nearest ${Math.round(s.spacing)} m sample point`;
  if (s.famous) c += `. ${s.famous.dem}`;
  $('coords').textContent = c;

  const margin = x.sunEl - x.sunHz;
  $('sun-state').textContent = x.frac >= 0.995 ? 'In full sunlight' : x.frac <= 0.005 ? 'In shadow' : `${Math.round(x.frac * 100)}% of the Sun visible`;
  $('sun-detail').textContent = `${vs(x.sunEl)}; terrain in that direction rises to ${x.sunHz.toFixed(2)}°, so the Sun's centre is ${margin >= 0 ? margin.toFixed(2) + '° above' : (-margin).toFixed(2) + '° below'} the real skyline.`;
  const tn = nextChange(s, S.t, 'lit');
  $('sun-next').textContent = tn === null ? '' : `${x.lit ? 'Sunset' : 'Sunrise'} in ${fmtDur(tn - S.t)}` + (tn !== Infinity ? ` (${fmtT(tn)})` : '');

  $('earth-state').textContent = x.see ? 'Earth in view' : 'Earth hidden by terrain';
  $('earth-detail').textContent = `${vs(x.earthEl)}; skyline at ${x.earthHz.toFixed(2)}°. ` +
    (x.see ? 'Direct-to-Earth radio is possible.' : 'A relay satellite would be needed.');
  const te = nextChange(s, S.t, 'see');
  $('earth-next').textContent = te === null ? '' : `${x.see ? 'Earthset' : 'Earthrise'} in ${fmtDur(te - S.t)}` + (te !== Infinity ? ` (${fmtT(te)})` : '');
}

function renderExplain() {
  const s = S.site, x = at(S.site, S.t);
  const ridge = Math.max(...s.hz);
  const az = s.hz.indexOf(ridge);
  $('explain').innerHTML = `
    <p>Near the Moon's south pole the Sun never climbs high. Right now it is ${Math.abs(x.sunEl).toFixed(1)}° ${x.sunEl >= 0 ? 'above' : 'below'} a smooth horizon,
    ${Math.abs(x.sunEl) < 2 * x.sunR ? 'less than the width of the Sun itself' : `about ${(Math.abs(x.sunEl) / (2 * x.sunR)).toFixed(0)} times the Sun's own width`}. On Earth at noon it can be 60° or more.
    ${x.sunEl < 0 && x.lit ? 'It can still shine here because this spot is high up and looks down past the curve of the Moon.' : ''}</p>
    <p>So the skyline decides everything. From this spot the highest terrain is ${ridge.toFixed(1)}° up, toward azimuth ${az}°.
    When the Sun slides behind a ridge like that, a solar-powered lander goes dark, sometimes for days.</p>
    <p>Earth hardly moves in the lunar sky; it nods up and down by a few degrees each month. Where Earth sits close to the skyline,
    radio contact switches on and off with that nod. Try the map: the bright pixels are where both stay up most often.</p>`;
}

function renderPlan() {
  const s = S.site, days = +$('p-days').value || 6.5, needSun = +$('p-sun').value, needEarth = +$('p-earth').value;
  const landEarth = $('p-landearth').checked, slip = +$('p-slip').value || 0;
  const t0 = S.t + slip * 24 * 3.6e6;
  const ok = (site, t) => { const m = mission(site, t, days); return [m.sunPct >= needSun - 1e-9 && m.earthPct >= needEarth - 1e-9 && (!landEarth || m.landEarth), m]; };
  const [pass, m] = ok(s, t0);
  const v = $('verdict');
  v.className = 'verdict ' + (pass ? 'ok' : 'no');
  let why = '';
  if (!pass) {
    if (m.sunPct < needSun) why = `sunlight only ${m.sunPct.toFixed(0)}% of the stay; first dark at ${fmtT(m.firstDark)}`;
    else if (landEarth && !m.landEarth) why = 'Earth is below the skyline at touchdown';
    else why = `Earth in view only ${m.earthPct.toFixed(0)}% of the stay`;
  }
  v.textContent = (slip ? `Landing ${fmtT(t0)}: ` : 'Landing now: ') + (pass
    ? `works. Sun ${m.sunPct.toFixed(0)}%, Earth ${m.earthPct.toFixed(0)}% of ${days} days.`
    : `does not work: ${why}.`);

  // same site, other dates: hourly start times over the next 60 days, grouped into windows
  const ol = $('pb-dates'); ol.innerHTML = '';
  let start = null, found = 0;
  const end = Math.min(t0 + 60 * 24 * 3.6e6, S.eStart + (S.eHours - days * 24 - 2) * 3.6e6);
  for (let t = t0; t <= end && found < 5; t += 3 * 3.6e6) {
    const [p] = ok(s, t);
    if (p && start === null) start = t;
    if ((!p || t + 3 * 3.6e6 > end) && start !== null) {
      const li = document.createElement('li');
      li.innerHTML = `${fmtT(start)} to ${fmtT(p ? t : t - 3 * 3.6e6)} (${fmtDur((p ? t : t - 3 * 3.6e6) - start + 3 * 3.6e6)} window) `;
      const b = document.createElement('button'); const tt = start; b.textContent = 'Go';
      b.addEventListener('click', () => { $('p-slip').value = 0; setTime(tt); });
      li.appendChild(b); ol.appendChild(li); start = null; found++;
    }
  }
  if (!found) ol.innerHTML = '<li>No start time in the next 60 days meets these limits at this spot.</li>';

  // same date, other spots in this region
  const sp = $('pb-spots'); sp.innerHTML = ''; S.planSpots = [];
  if (!S.region || !S.regionData[S.region]) { $('pb-hint').textContent = 'Pick a spot inside an Artemis region to search nearby spots.'; drawMap(); return; }
  const d = S.regionData[S.region], g = d.meta.grid, cand = [];
  for (let gi = 0; gi < g.H; gi++) for (let gj = 0; gj < g.W; gj++) {
    if (!g.mask[gi * g.W + gj]) continue;
    const site = gridSite(S.region, gi, gj), [p, mm] = ok(site, t0);
    const km = distKm(s, site);
    if (p && km > 0.05) cand.push({ site, mm, km });
  }
  cand.sort((a, b) => a.km - b.km);
  S.planSpots = cand;
  $('pb-hint').textContent = `${cand.length} of ${g.mask.reduce((a, b) => a + b, 0)} sample points in ${d.meta.name} meet the limits for this landing time (green on the map).`;
  cand.slice(0, 5).forEach((c) => {
    const li = document.createElement('li');
    li.textContent = `${c.km.toFixed(1)} km away: Sun ${c.mm.sunPct.toFixed(0)}%, Earth ${c.mm.earthPct.toFixed(0)}% `;
    const b = document.createElement('button'); b.textContent = 'Go';
    b.addEventListener('click', () => { $('p-slip').value = 0; S.t = t0; setSite(c.site, 'Plan B spot'); });
    li.appendChild(b); sp.appendChild(li);
  });
  drawMap();
}

// ---------------------------------------------------------------- panorama
function fitCanvas(cv, h) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return [ctx, w, h];
}
function drawPano() {
  const cv = $('pano'), [ctx, W, H] = fitCanvas(cv, Math.max(220, Math.min(320, cv.clientWidth * 0.22)));
  const s = S.site, hz = s.hz;
  let trackMax = -90;
  const track = [];
  for (let h = 0; h < 14 * 24; h += 2) { const x = at(s, S.t + h * 3.6e6); track.push(x); trackMax = Math.max(trackMax, x.sunEl, x.earthEl); }
  const top = Math.min(90, Math.max(10, Math.ceil(Math.max(...hz) + 2), Math.ceil(trackMax + 2)));
  const bot = Math.min(-4, Math.floor(Math.min(...hz)) - 1);
  const X = (az) => az / 360 * W, Y = (el) => (top - el) / (top - bot) * H;
  const exag = (H / (top - bot)) / (W / 360);
  $('exag').textContent = exag >= 1.5 ? `Heights stretched ${exag.toFixed(0)}× so the skyline is visible` : '';
  const g = ctx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, '#0b1019'); g.addColorStop(1, '#1d2738');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  // degree grid
  ctx.strokeStyle = 'rgba(154,160,166,.15)'; ctx.lineWidth = 1; ctx.fillStyle = 'rgba(154,160,166,.7)'; ctx.font = '12px Barlow, sans-serif';
  const gs = top > 40 ? 15 : top > 20 ? 10 : 5;
  for (let e = Math.ceil(bot / gs) * gs; e <= top; e += gs) { ctx.beginPath(); ctx.moveTo(0, Y(e)); ctx.lineTo(W, Y(e)); ctx.stroke(); ctx.fillText(`${e}°`, 4, Y(e) - 3); }
  // tracks for the next 14 days (2-hour steps)
  for (const x of track) {
    ctx.fillStyle = x.lit ? 'rgba(242,181,68,.85)' : 'rgba(242,181,68,.22)';
    ctx.beginPath(); ctx.arc(X(x.sunAz), Y(x.sunEl), 1.8, 0, 7); ctx.fill();
    ctx.fillStyle = x.see ? 'rgba(98,182,218,.9)' : 'rgba(98,182,218,.22)';
    ctx.beginPath(); ctx.arc(X(x.earthAz), Y(x.earthEl), 1.8, 0, 7); ctx.fill();
  }
  // current Sun and Earth (drawn before the terrain so the skyline can cut them)
  const now = at(s, S.t), pxDeg = H / (top - bot);
  ctx.fillStyle = '#ffd77a'; ctx.beginPath(); ctx.arc(X(now.sunAz), Y(now.sunEl), Math.max(4, now.sunR * pxDeg), 0, 7); ctx.fill();
  ctx.fillStyle = '#62b6da'; ctx.beginPath(); ctx.arc(X(now.earthAz), Y(now.earthEl), Math.max(4, 0.95 * pxDeg), 0, 7); ctx.fill();
  // terrain
  ctx.beginPath(); ctx.moveTo(0, H);
  for (let a = 0; a <= A; a++) ctx.lineTo(X(a * 360 / A), Y(hz[a % A]));
  ctx.lineTo(W, H); ctx.closePath();
  const tg = ctx.createLinearGradient(0, Y(top / 2), 0, H); tg.addColorStop(0, '#6d7279'); tg.addColorStop(1, '#2a3038');
  ctx.fillStyle = tg; ctx.fill(); ctx.strokeStyle = '#c9c6bd'; ctx.lineWidth = 1.2; ctx.stroke();
  // smooth-Moon horizon
  ctx.setLineDash([6, 5]); ctx.strokeStyle = 'rgba(230,225,214,.55)'; ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke(); ctx.setLineDash([]);
  // compass
  ctx.fillStyle = '#e6e1d6'; ctx.font = '600 14px "Barlow Condensed", sans-serif';
  [['N', 0], ['E', 90], ['S', 180], ['W', 270], ['N', 360]].forEach(([l, a]) => { ctx.fillText(l, Math.min(W - 12, Math.max(2, X(a) - 4)), H - 6); });
  // labels
  ctx.font = '600 15px "Barlow Condensed", sans-serif';
  const lab = (txt, x, y, col) => { ctx.fillStyle = col; const w = ctx.measureText(txt).width; ctx.fillText(txt, Math.min(W - w - 4, Math.max(4, x + 10)), Math.max(16, y - 10)); };
  lab(`Sun ${now.frac >= 0.5 ? 'up' : 'down'}`, X(now.sunAz), Y(now.sunEl), '#ffd77a');
  lab(`Earth ${now.see ? 'in view' : 'hidden'}`, X(now.earthAz), Y(now.earthEl), '#9fd6ee');
}

// ---------------------------------------------------------------- map
function insideRegion(r, lat, lon) {
  const pts = r.cells.flatMap((c) => densify(c)).map(([la, lo]) => ll2rc(la, lo));
  const [y, x] = ll2rc(lat, lon);
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [yi, xi] = pts[i], [yj, xj] = pts[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function densify(ring, n = 24) {
  const out = [];
  for (let i = 0; i < ring.length - 1; i++) {
    const [a0, b0] = ring[i], [a1, b1] = ring[i + 1]; let db = b1 - b0; db = ((db + 540) % 360) - 180;
    for (let k = 0; k < n; k++) out.push([a0 + (a1 - a0) * k / n, b0 + db * k / n]);
  }
  return out;
}
const RAMPS = {
  sun: ['#0b1026', '#2c2463', '#7a2f6b', '#d0563c', '#f6b443', '#fff7d6'],
  earth: ['#07121e', '#0e3550', '#13708a', '#2eb3b0', '#8fe8cf', '#effff9'],
  both: ['#0c0a18', '#3a1d55', '#7f2f7e', '#c95b8c', '#f6a38c', '#fff0e4'],
  night: ['#fff7d6', '#f6b443', '#d0563c', '#7a2f6b', '#2c2463', '#0b1026'],
};
const LEG = { both: ['0%', '100% of the time'], sun: ['0%', '100% average sunlight'], earth: ['0%', '100% of the time'], night: ['0 days', '15 days or more'] };

function mapView() {                         // returns transform for the current map view
  const cv = $('map'), size = cv.clientWidth;
  if (S.region) {
    const m = S.index.regions[S.region], im = m.image;
    const sc = size / Math.max(im.W, im.H), ox = (size - im.W * sc) / 2, oy = (size - im.H * sc) / 2;
    const unit = m.image.f * m.box.stride;     // 20 m pixels per image pixel
    return { size, kind: 'region', sc, ox, oy,
      toPx: (lat, lon) => { const [r, c] = ll2rc(lat, lon); return [ox + (c - m.box.c0) / unit * sc, oy + (r - m.box.r0) / unit * sc]; },
      fromPx: (x, y) => { const ix = (x - ox) / sc, iy = (y - oy) / sc; return { ix, iy, r: m.box.r0 + iy * unit, c: m.box.c0 + ix * unit }; } };
  }
  const o = S.overview, sc = size / Math.max(o.W, o.H);
  return { size, kind: 'overview', sc,
    toPx: (lat, lon) => { const [r, c] = ll2rc(lat, lon); return [(c - o.c0) / o.step * sc, (r - o.r0) / o.step * sc]; },
    fromPx: (x, y) => ({ r: o.r0 + y / sc * o.step, c: o.c0 + x / sc * o.step }) };
}
function drawMap() {
  const cv = $('map'), size = cv.clientWidth; if (!size) return;
  const [ctx] = fitCanvas(cv, size), v = mapView();
  ctx.fillStyle = '#0f1520'; ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
  if (v.kind === 'overview') {
    ctx.drawImage(S.overviewImg, 0, 0, S.overview.W * v.sc, S.overview.H * v.sc);
    for (const r of Object.values(S.index.regions)) {
      ctx.beginPath();
      r.cells.forEach((cell) => { densify(cell).map(([a, b]) => v.toPx(a, b)).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); });
      const best = r.summary.mean_both_pct / 100;
      ctx.fillStyle = `rgba(242,181,68,${0.12 + 0.45 * best})`; ctx.fill();
      ctx.strokeStyle = '#f2b544'; ctx.lineWidth = 1.5; ctx.stroke();
      const [cx, cy] = v.toPx(r.center[0], r.center[1]);
      ctx.font = '600 13px "Barlow Condensed", sans-serif'; ctx.fillStyle = '#e6e1d6';
      ctx.fillText(r.name, cx + 10, cy + 4);
    }
    const [px, py] = v.toPx(-90, 0); ctx.fillStyle = '#e6e1d6'; ctx.beginPath(); ctx.arc(px, py, 3, 0, 7); ctx.fill();
    ctx.fillText('South pole', px + 6, py - 6);
    S.sites.filter((s) => s.lat < -80).forEach((s) => { const [x, y] = v.toPx(s.lat, s.lon); ctx.strokeStyle = '#62b6da'; ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.stroke(); ctx.fillStyle = '#9fd6ee'; ctx.fillText(s.name, x + 8, y + 14); });
  } else {
    const d = S.regionData[S.region]; if (!d) return;
    const im = d.meta.image;
    ctx.drawImage(d.img[S.layer], v.ox, v.oy, im.W * v.sc, im.H * v.sc);
    ctx.beginPath();
    d.meta.cells.forEach((cell) => { densify(cell).map(([a, b]) => v.toPx(a, b)).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); });
    ctx.strokeStyle = 'rgba(230,225,214,.8)'; ctx.lineWidth = 1.5; ctx.stroke();
    if (S.mode === 'plan') S.planSpots.forEach((c) => { const [x, y] = v.toPx(c.site.lat, c.site.lon); ctx.fillStyle = '#9fd3a4'; ctx.fillRect(x - 2, y - 2, 4, 4); });
    const lg = LEG[S.layer]; $('lg-lo').textContent = lg[0]; $('lg-hi').textContent = lg[1];
    $('lg-bar').style.background = `linear-gradient(90deg, ${RAMPS[S.layer].join(',')})`;
  }
  // pinned scenarios and current site
  S.compare.forEach((c, i) => { const [x, y] = v.toPx(c.lat, c.lon); ctx.fillStyle = '#e6e1d6'; ctx.font = '600 13px "Barlow Condensed", sans-serif'; ctx.fillText(String(i + 1), x + 6, y - 6); ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill(); });
  const [x, y] = v.toPx(S.site.lat, S.site.lon);
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 8, 0, 7); ctx.stroke();
  ctx.strokeStyle = '#1a2230'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, 10, 0, 7); ctx.stroke();
}
function regionPixel(v, e) {
  const r = e.target.getBoundingClientRect(), p = v.fromPx(e.clientX - r.left, e.clientY - r.top);
  const m = S.index.regions[S.region], im = m.image;
  const ix = Math.floor(p.ix), iy = Math.floor(p.iy);
  if (ix < 0 || iy < 0 || ix >= im.W || iy >= im.H) return null;
  const k = (iy * im.W + ix) * 4, st = S.regionData[S.region].stats;
  const [lat, lon] = rc2ll(p.r, p.c);
  return { lat, lon, illum: st[k], earth: st[k + 1], both: st[k + 2], night: st[k + 3] * 4 };
}
async function onMapClick(e) {
  const v = mapView();
  if (v.kind === 'overview') {
    const r = e.target.getBoundingClientRect(), p = v.fromPx(e.clientX - r.left, e.clientY - r.top);
    const [lat, lon] = rc2ll(p.r, p.c);
    const reg = Object.values(S.index.regions).find((x) => insideRegion(x, lat, lon));
    if (reg) await selectRegionBest(reg.key, 'best_both');
    return;
  }
  const px = regionPixel(v, e); if (!px) return;
  const [gi, gj] = nearestGrid(S.region, px.lat, px.lon);
  setSite(gridSite(S.region, gi, gj), 'your pick');
}
function onMapHover(e) {
  const tip = $('tip'), v = mapView();
  if (v.kind !== 'region' || !S.regionData[S.region]) { tip.hidden = true; return; }
  const px = regionPixel(v, e); if (!px) { tip.hidden = true; return; }
  tip.innerHTML = `Sun and Earth together ${px.both}%<br>Average sunlight ${px.illum}%<br>Earth in view ${px.earth}%<br>Longest night ${(px.night / 24).toFixed(1)} days`;
  const r = e.target.getBoundingClientRect(); let x = e.clientX - r.left + 14, y = e.clientY - r.top + 14;
  tip.hidden = false; if (x + tip.offsetWidth > r.width) x -= tip.offsetWidth + 28; if (y + tip.offsetHeight > r.height) y -= tip.offsetHeight + 28;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}

// ---------------------------------------------------------------- timeline
function drawTimeline() {
  const cv = $('tl'), [ctx, W, H] = fitCanvas(cv, 120), s = S.site, hours = 30 * 24;
  ctx.fillStyle = '#0f1520'; ctx.fillRect(0, 0, W, H);
  const rows = [['Sunlight', 10, (x) => `rgba(242,181,68,${x.frac})`], ['Earth in view', 44, (x) => (x.see ? '#62b6da' : 'rgba(98,182,218,.08)')],
                ['Both', 78, (x) => (x.lit && x.see ? '#e6e1d6' : 'rgba(230,225,214,.06)')]];
  const L = 92, bw = (W - L - 8) / hours;
  const xs = []; for (let h = 0; h < hours; h++) xs.push(at(s, S.t + h * 3.6e6));
  ctx.font = '500 13px Barlow, sans-serif';
  rows.forEach(([name, y, col]) => {
    ctx.fillStyle = '#9aa0a6'; ctx.fillText(name, 6, y + 18);
    xs.forEach((x, h) => { ctx.fillStyle = col(x); ctx.fillRect(L + h * bw, y, Math.ceil(bw), 26); });
  });
  ctx.fillStyle = 'rgba(230,225,214,.55)';
  for (let dday = 0; dday <= 30; dday += 5) { const x = L + dday * 24 * bw; ctx.fillRect(x, 4, 1, H - 8); ctx.fillText(dday ? `+${dday} d` : 'now', x + 3, H - 2); }
  if (S.mode === 'plan') {
    const slip = +$('p-slip').value || 0, days = +$('p-days').value || 6.5;
    const x0 = L + slip * 24 * bw, x1 = x0 + days * 24 * bw;
    ctx.strokeStyle = '#9fd3a4'; ctx.lineWidth = 2; ctx.strokeRect(x0, 6, Math.min(W - 8, x1) - x0, 100);
  }
}

// ---------------------------------------------------------------- compare
function pinScenario() {
  if (S.compare.length >= 4) S.compare.shift();
  const s = S.site, x = at(s, S.t), days = +$('p-days').value || 6.5, m = mission(s, S.t, days);
  const ts = nextChange(s, S.t, 'lit'), te = nextChange(s, S.t, 'see');
  S.compare.push({ name: s.name + (s.label ? ` (${s.label})` : ''), lat: s.lat, lon: s.lon, t: S.t,
    sun: x.lit ? 'up' : 'down', sunNext: ts === null ? '' : `${x.lit ? 'sets' : 'rises'} in ${fmtDur(ts - S.t)}`,
    earth: x.see ? 'in view' : 'hidden', earthNext: te === null ? '' : `${x.see ? 'sets' : 'rises'} in ${fmtDur(te - S.t)}`,
    days, sunPct: m.sunPct, earthPct: m.earthPct, maxDark: m.maxDark });
  renderCompare(); drawMap();
}
function renderCompare() {
  $('compare').hidden = !S.compare.length;
  const rows = [['Site', (c) => c.name], ['Where', (c) => fmtLL(c.lat, c.lon)], ['Landing', (c) => fmtT(c.t)],
    ['Sun at landing', (c) => `${c.sun}, ${c.sunNext}`], ['Earth at landing', (c) => `${c.earth}, ${c.earthNext}`],
    ['Sunlight during stay', (c) => `${c.sunPct.toFixed(0)}% of ${c.days} days`], ['Earth during stay', (c) => `${c.earthPct.toFixed(0)}%`],
    ['Longest darkness in stay', (c) => `${c.maxDark} h`]];
  const t = $('cmp'); t.innerHTML = '';
  const head = document.createElement('tr'); head.innerHTML = '<th></th>' + S.compare.map((c, i) => `<th>${i + 1} <button data-rm="${i}" aria-label="Remove scenario ${i + 1}">Remove</button></th>`).join('');
  t.appendChild(head);
  rows.forEach(([name, f]) => { const tr = document.createElement('tr'); tr.innerHTML = `<th>${name}</th>` + S.compare.map((c) => `<td class="num">${f(c)}</td>`).join(''); t.appendChild(tr); });
  t.querySelectorAll('[data-rm]').forEach((b) => b.addEventListener('click', () => { S.compare.splice(+b.dataset.rm, 1); renderCompare(); drawMap(); }));
}

init().catch((e) => {
  document.body.insertAdjacentHTML('afterbegin', `<p style="padding:1rem;background:#5a2a22">Could not load the data files (${e.message}). Serve this folder over HTTP, for example: python -m http.server</p>`);
  console.error(e);
});
