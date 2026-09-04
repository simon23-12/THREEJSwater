// Shared GLSL chunks (GLSL ES 1.00 so it runs everywhere three.js does).

export const NOISE = /* glsl */ `
float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

const mat2 M2 = mat2(1.62, 1.20, -1.20, 1.62);

float fbm(vec2 p, int oct) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 8; i++) {
    if (i >= oct) break;
    v += a * vnoise(p);
    p = M2 * p;
    a *= 0.5;
  }
  return v;
}
`;

export const COMPLEX = /* glsl */ `
vec2 cmul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
`;

// Analytic stormy sky. Used to bake an equirect environment texture once per frame,
// which the ocean then samples for reflection + aerial perspective.
export const SKY_GLSL = /* glsl */ `
uniform vec3 uZenithColor;
uniform vec3 uHorizonColor;
uniform vec3 uHazeColor;
uniform vec3 uGlowColor;
uniform vec3 uCloudDark;
uniform vec3 uCloudLit;
uniform vec3 uSunColor;
uniform vec3 uSunDir;
uniform float uHazeStrength;
uniform float uHazeSharp;
uniform float uHazeFalloff;
uniform float uGlowStrength;
uniform float uSunIntensity;
uniform float uCloudCover;
uniform float uCloudOpacity;
uniform float uCloudScale;
uniform float uCloudBase;
uniform float uCloudContrast;
uniform float uCloudLight;
uniform vec2  uCloudOffset;

vec3 skyRadiance(vec3 dir) {
  float y = clamp(dir.y, -1.0, 1.0);
  float up = clamp(y, 0.0, 1.0);

  // Slate overhead falling into a warm, hazy horizon.
  vec3 col = mix(uHorizonColor, uZenithColor, pow(up, uHazeFalloff));

  // Thin luminous band right on the horizon line.
  col += uHazeColor * exp(-up * uHazeSharp) * uHazeStrength;

  // Downward rays (water reflecting the underside) never hit a hard edge.
  col = mix(col, uHorizonColor * 0.55, clamp(-y * 4.0, 0.0, 1.0));

  // Warm glow around the sun azimuth, hugging the horizon.
  vec2 dh = normalize(dir.xz + vec2(1e-5));
  vec2 sh = normalize(uSunDir.xz + vec2(1e-5));
  float az = max(dot(dh, sh), 0.0);
  col += uGlowColor * pow(az, 2.5) * exp(-up * 9.0) * uGlowStrength;

  // Veiled sun — no hard disc, this is an overcast dusk.
  float sd = max(dot(dir, uSunDir), 0.0);
  col += uSunColor * pow(sd, 26.0) * uSunIntensity;

  // Cloud deck projected onto a plane so it converges toward the horizon.
  vec2 cp = dir.xz / (up + uCloudBase) * uCloudScale + uCloudOffset;
  float w = fbm(cp * 0.35, 4);
  float n = fbm(cp * 0.35 + vec2(w * 1.8, -w * 1.2), 6);
  n = clamp((n - 0.5) * uCloudContrast + 0.5, 0.0, 1.0);   // plain fbm is too flat
  n = mix(0.5, n, smoothstep(0.010, 0.10, up));           // detail converges -> soften

  float dens = smoothstep(uCloudCover, uCloudCover + 0.34, n);
  dens *= smoothstep(0.004, 0.135, y);          // dissolve into the haze
  dens *= 1.0 - 0.25 * smoothstep(0.60, 1.0, up);

  // Seen from below, a deck is bright where it is thin and dark where it piles
  // up. Tying brightness to density (not the other way round) is what makes
  // the mass read as weather instead of as a flat wash.
  float thick = smoothstep(uCloudCover - 0.02, uCloudCover + 0.30, n);
  vec3 cloud = mix(uCloudLit, uCloudDark, thick);
  cloud += uGlowColor * pow(az, 4.0) * (1.0 - thick) * uCloudLight;

  col = mix(col, cloud, clamp(dens * uCloudOpacity, 0.0, 1.0));

  return max(col, vec3(0.0));
}
`;
