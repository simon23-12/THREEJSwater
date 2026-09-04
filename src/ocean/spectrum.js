import * as THREE from 'three';

const G = 9.81;

/* ------------------------------------------------------------------ *
 * Deterministic RNG so the sea state is reproducible between reloads.
 * ------------------------------------------------------------------ */
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussPair(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const r = Math.sqrt(-2 * Math.log(u));
  const t = 2 * Math.PI * v;
  return [r * Math.cos(t), r * Math.sin(t)];
}

/* ------------------------------------------------------------------ *
 * log-gamma (Lanczos) — needed to normalise the directional spreading.
 * ------------------------------------------------------------------ */
const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7
];

function lgamma(z) {
  if (z < 0.5) {
    return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * z))) - lgamma(1 - z);
  }
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < 8; i++) x += LANCZOS[i] / (z + i + 1);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/* ------------------------------------------------------------------ *
 * JONSWAP energy spectrum (fetch-limited wind sea).
 * ------------------------------------------------------------------ */
function jonswap(omega, U, fetch, gamma) {
  if (omega <= 1e-5) return 0;
  const alpha = 0.076 * Math.pow((U * U) / (fetch * G), 0.22);
  const omegaP = 22 * Math.pow((G * G) / (U * fetch), 1 / 3);
  const sigma = omega <= omegaP ? 0.07 : 0.09;
  const r = Math.exp(-Math.pow(omega - omegaP, 2) / (2 * sigma * sigma * omegaP * omegaP));
  return (
    ((alpha * G * G) / Math.pow(omega, 5)) *
    Math.exp(-1.25 * Math.pow(omegaP / omega, 4)) *
    Math.pow(gamma, r)
  );
}

function peakOmega(U, fetch) {
  return 22 * Math.pow((G * G) / (U * fetch), 1 / 3);
}

/* TMA correction: shallow-water damping of the long waves. */
function tmaAttenuation(omega, depth) {
  const wh = omega * Math.sqrt(depth / G);
  if (wh <= 1) return 0.5 * wh * wh;
  if (wh < 2) return 1 - 0.5 * (2 - wh) * (2 - wh);
  return 1;
}

/* Hasselmann cos^2s spreading, properly normalised so the shape does not
 * secretly rescale the energy across frequencies. */
function directionalSpread(theta, omega, omegaP, swellBoost) {
  let s = omega > omegaP
    ? 9.77 * Math.pow(omega / omegaP, -2.5)
    : 6.97 * Math.pow(omega / omegaP, 4.06);
  s = Math.max(s * swellBoost, 0.4);

  const c = Math.cos(theta * 0.5);
  if (c <= 0) return 0;

  // Q(s) = 2^(2s-1)/pi * G(s+1)^2 / G(2s+1)
  const logQ =
    (2 * s - 1) * Math.LN2 - Math.log(Math.PI) + 2 * lgamma(s + 1) - lgamma(2 * s + 1);
  return Math.exp(logQ) * Math.pow(c, 2 * s);
}

/* ------------------------------------------------------------------ *
 * h0(k) — the frozen initial spectrum for one cascade.
 * RGBA = ( h0(k).re, h0(k).im, conj(h0(-k)).re, conj(h0(-k)).im )
 * ------------------------------------------------------------------ */
export function buildH0Texture(N, L, opts) {
  const {
    windSpeed = 11,
    windAngle = 0,
    fetch = 120000,
    depth = 500,
    gamma = 3.3,
    swellBoost = 1.0,
    boundLow = 0.0001,
    boundHigh = 1e9,
    shortCut = 0.5,     // metres — kills sub-texel ripples that only alias
    seed = 1337
  } = opts;

  const rng = mulberry32(seed);
  const cells = N * N;
  const gr = new Float32Array(cells);
  const gi = new Float32Array(cells);
  for (let i = 0; i < cells; i++) {
    const [a, b] = gaussPair(rng);
    gr[i] = a;
    gi[i] = b;
  }

  const dk = (2 * Math.PI) / L;
  const omegaP = peakOmega(windSpeed, fetch);
  const amp = new Float32Array(cells);

  for (let m = 0; m < N; m++) {
    for (let n = 0; n < N; n++) {
      const kx = (n - N / 2) * dk;
      const kz = (m - N / 2) * dk;
      const k = Math.hypot(kx, kz);
      let a = 0;

      if (k > 1e-6 && k >= boundLow && k < boundHigh) {
        const omega = Math.sqrt(G * k);
        const dOmegaDk = G / (2 * omega);

        // S(k) = S(omega) * (domega/dk) / k , then spread over direction
        let S = jonswap(omega, windSpeed, fetch, gamma) * tmaAttenuation(omega, depth);
        S *= dOmegaDk / k;
        S *= directionalSpread(Math.atan2(kz, kx) - windAngle, omega, omegaP, swellBoost);

        // suppress waves shorter than `shortCut` (pure aliasing otherwise)
        S *= Math.exp(-k * k * shortCut * shortCut);

        a = Math.sqrt(Math.max(2 * S * dk * dk, 0));
        if (!Number.isFinite(a)) a = 0;
      }
      amp[m * N + n] = a;
    }
  }

  const data = new Float32Array(cells * 4);
  const invSqrt2 = Math.SQRT1_2;

  for (let m = 0; m < N; m++) {
    for (let n = 0; n < N; n++) {
      const i = m * N + n;
      // index of -k on the centred lattice
      const j = ((N - m) % N) * N + ((N - n) % N);

      data[i * 4 + 0] = invSqrt2 * gr[i] * amp[i];
      data[i * 4 + 1] = invSqrt2 * gi[i] * amp[i];
      data[i * 4 + 2] = invSqrt2 * gr[j] * amp[j];
      data[i * 4 + 3] = -invSqrt2 * gi[j] * amp[j];   // conjugate
    }
  }

  const tex = new THREE.DataTexture(data, N, N, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/* ------------------------------------------------------------------ *
 * Butterfly / twiddle lookup for the Stockham-style GPU FFT.
 * width = log2(N) stages, height = N.
 * RGBA = ( twiddle.re, twiddle.im, topIndex, bottomIndex )
 * ------------------------------------------------------------------ */
export function buildButterflyTexture(N) {
  const stages = Math.log2(N);
  if (!Number.isInteger(stages)) throw new Error('FFT size must be a power of two');

  const bits = stages;
  const rev = new Uint16Array(N);
  for (let i = 0; i < N; i++) {
    let x = i, r = 0;
    for (let b = 0; b < bits; b++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    rev[i] = r;
  }

  const data = new Float32Array(stages * N * 4);

  for (let stage = 0; stage < stages; stage++) {
    const span = 1 << stage;
    const group = 1 << (stage + 1);
    for (let y = 0; y < N; y++) {
      const k = (y * (N / group)) % N;
      const twR = Math.cos((2 * Math.PI * k) / N);
      const twI = Math.sin((2 * Math.PI * k) / N);
      const topWing = y % group < span;

      let top, bot;
      if (stage === 0) {
        if (topWing) { top = rev[y]; bot = rev[y + 1]; }
        else { top = rev[y - 1]; bot = rev[y]; }
      } else {
        if (topWing) { top = y; bot = y + span; }
        else { top = y - span; bot = y; }
      }

      const idx = (y * stages + stage) * 4;
      data[idx + 0] = twR;
      data[idx + 1] = twI;
      data[idx + 2] = top;
      data[idx + 3] = bot;
    }
  }

  const tex = new THREE.DataTexture(data, stages, N, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}
