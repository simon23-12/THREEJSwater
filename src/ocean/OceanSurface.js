import * as THREE from 'three';
import { NOISE } from '../shaders/common.js';

/* ------------------------------------------------------------------ *
 * Radial clipmap: dense under the camera, exponentially coarser out to
 * the horizon. Recentred on the camera every frame, so the ocean is
 * effectively infinite while staying a single draw call.
 * ------------------------------------------------------------------ */
function buildRadialGrid(rings, segments, rMin, rMax) {
  // solve for the growth factor that lands the outer ring exactly on rMax
  let lo = 1.0001, hi = 1.2;
  for (let it = 0; it < 60; it++) {
    const g = (lo + hi) * 0.5;
    const total = (rMin * (Math.pow(g, rings - 1) - 1)) / (g - 1);
    if (total < rMax) lo = g; else hi = g;
  }
  const g = (lo + hi) * 0.5;

  const radii = new Float32Array(rings);
  let r = 0, dr = rMin;
  for (let i = 1; i < rings; i++) {
    r += dr;
    dr *= g;
    radii[i] = r;
  }

  const count = rings * segments;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segments; j++) {
      const t = (j / segments) * Math.PI * 2;
      const idx = (i * segments + j) * 3;
      positions[idx + 0] = radii[i] * Math.cos(t);
      positions[idx + 1] = 0;
      positions[idx + 2] = radii[i] * Math.sin(t);
    }
  }

  const quads = (rings - 1) * segments;
  const indices = count > 65535 ? new Uint32Array(quads * 6) : new Uint16Array(quads * 6);
  let o = 0;
  for (let i = 0; i < rings - 1; i++) {
    for (let j = 0; j < segments; j++) {
      const j1 = (j + 1) % segments;
      const a = i * segments + j;
      const b = i * segments + j1;
      const c = (i + 1) * segments + j;
      const d = (i + 1) * segments + j1;
      indices[o++] = a; indices[o++] = b; indices[o++] = c;
      indices[o++] = b; indices[o++] = d; indices[o++] = c;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), rMax * 1.2);
  return geo;
}

const VERT = /* glsl */ `
uniform sampler2D uDisp0;
uniform sampler2D uDisp1;
uniform sampler2D uDisp2;
uniform vec3 uL;
uniform vec3 uCam;
uniform vec4 uFade;

varying vec3 vWorld;
varying float vDist;
varying vec3 vW;

void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  float d = length(wp.xz - uCam.xz);

  vec3 w;
  w.x = 1.0;
  w.y = 1.0 - smoothstep(uFade.x, uFade.y, d);
  w.z = 1.0 - smoothstep(uFade.z, uFade.w, d);

  vec3 disp =
      texture2D(uDisp0, wp.xz / uL.x).xyz * w.x
    + texture2D(uDisp1, wp.xz / uL.y).xyz * w.y
    + texture2D(uDisp2, wp.xz / uL.z).xyz * w.z;

  wp.xyz += disp;

  vWorld = wp.xyz;
  vDist = d;
  vW = w;

  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const FRAG = /* glsl */ `
precision highp float;
${NOISE}

uniform sampler2D uDisp0;
uniform sampler2D uDisp1;
uniform sampler2D uDisp2;
uniform sampler2D uFoam0;
uniform sampler2D uFoam1;
uniform sampler2D uFoam2;
uniform sampler2D uSky;

uniform vec3 uL;
uniform vec3 uCam;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uDeepColor;
uniform vec3 uScatterColor;
uniform vec3 uFoamColor;
uniform vec3 uSkyAmbient;

uniform float uFoamAmount;
uniform float uFoamSharpness;
uniform float uFoamCrest;
uniform float uFogDensity;
uniform float uFogPower;
uniform float uSpecular;
uniform float uRoughness;
uniform float uScatter;
uniform float uDetailFalloff;

varying vec3 vWorld;
varying float vDist;
varying vec3 vW;

vec3 sampleDisp(vec2 p) {
  return texture2D(uDisp0, p / uL.x).xyz * vW.x
       + texture2D(uDisp1, p / uL.y).xyz * vW.y
       + texture2D(uDisp2, p / uL.z).xyz * vW.z;
}

vec2 equirectUv(vec3 d) {
  return vec2(
    atan(d.z, d.x) * 0.15915494 + 0.5,
    asin(clamp(d.y, -1.0, 1.0)) * 0.31830989 + 0.5
  );
}

vec3 sampleSky(vec3 d) {
  return texture2D(uSky, equirectUv(normalize(d))).rgb;
}

void main() {
  vec2 p = vWorld.xz;

  // Finite differences widen with distance: a cheap low-pass that keeps
  // the far field from boiling into specular noise.
  float e = max(uL.z / 256.0, vDist * 0.0016);

  vec3 dxp = sampleDisp(p + vec2(e, 0.0));
  vec3 dxm = sampleDisp(p - vec2(e, 0.0));
  vec3 dzp = sampleDisp(p + vec2(0.0, e));
  vec3 dzm = sampleDisp(p - vec2(0.0, e));

  float inv = 1.0 / (2.0 * e);
  vec3 dPdx = vec3(1.0 + (dxp.x - dxm.x) * inv, (dxp.y - dxm.y) * inv, (dxp.z - dxm.z) * inv);
  vec3 dPdz = vec3((dzp.x - dzm.x) * inv, (dzp.y - dzm.y) * inv, 1.0 + (dzp.z - dzm.z) * inv);

  vec3 N = normalize(cross(dPdz, dPdx));

  // The Jacobian that decides "is this crest breaking?" is measured over a
  // wave-sized window, not a texel — otherwise every ripple foams.
  float fe = max(0.75, vDist * 0.004);
  vec3 fxp = sampleDisp(p + vec2(fe, 0.0));
  vec3 fxm = sampleDisp(p - vec2(fe, 0.0));
  vec3 fzp = sampleDisp(p + vec2(0.0, fe));
  vec3 fzm = sampleDisp(p - vec2(0.0, fe));
  float finv = 1.0 / (2.0 * fe);
  float Jxx = 1.0 + (fxp.x - fxm.x) * finv;
  float Jzz = 1.0 + (fzp.z - fzm.z) * finv;
  float Jxz = (fzp.x - fzm.x) * finv;
  float Jzx = (fxp.z - fxm.z) * finv;
  float jac = Jxx * Jzz - Jxz * Jzx;

  float far = smoothstep(120.0, uDetailFalloff, vDist);
  vec3 Nsh = normalize(mix(N, vec3(0.0, 1.0, 0.0), far * 0.82));

  vec3 V = normalize(uCam - vWorld);
  float NoV = clamp(dot(Nsh, V), 0.0, 1.0);
  float NoUp = clamp(dot(Nsh, vec3(0.0, 1.0, 0.0)), 0.0, 1.0);

  // ------------------------------------------------ foam
  float foam = texture2D(uFoam0, p / uL.x).r * vW.x
             + texture2D(uFoam1, p / uL.y).r * vW.y
             + texture2D(uFoam2, p / uL.z).r * vW.z;
  foam += clamp((uFoamCrest - jac) * 2.2, 0.0, 1.0);

  // Foam texture lives at wave scale, not pixel scale — sub-metre speckle is
  // what makes procedural foam read as noise instead of as broken water.
  vec2 gp = mod(p, 1024.0);
  float grain = fbm(gp * 0.14, 4) * 0.55 + fbm(gp * 0.55, 3) * 0.30 + fbm(gp * 1.9, 2) * 0.15;
  grain = clamp((grain - 0.5) * 2.1 + 0.5, 0.0, 1.0);   // break the crest lines up
  grain = mix(grain, 0.5, far);

  foam = clamp(foam * uFoamAmount, 0.0, 1.5);
  foam = smoothstep(0.5 - uFoamSharpness, 0.5 + uFoamSharpness, foam * (0.10 + 1.75 * grain));
  foam *= 1.0 - far * 0.3;

  // ------------------------------------------------ reflection
  vec3 R = reflect(-V, Nsh);
  R.y = max(R.y, 0.0035);
  vec3 refl = sampleSky(R);
  float rough = clamp(uRoughness + far * 0.65, 0.0, 1.0);
  refl = mix(refl, uSkyAmbient, rough * 0.5);

  float F = 0.02 + 0.98 * pow(1.0 - NoV, 5.0);

  // ------------------------------------------------ sun glint (GGX)
  vec3 Hv = normalize(V + uSunDir);
  float NoH = clamp(dot(Nsh, Hv), 0.0, 1.0);
  float a = max(0.03, uRoughness + far * 0.5);
  a *= a;
  float dd = NoH * NoH * (a * a - 1.0) + 1.0;
  float ggx = (a * a) / (3.14159265 * dd * dd);
  float NoLs = clamp(dot(Nsh, uSunDir), 0.0, 1.0);
  vec3 glint = uSunColor * ggx * NoLs * uSpecular;

  // ------------------------------------------------ subsurface scatter
  float height = clamp(vWorld.y * 0.30 + 0.36, 0.0, 1.0);
  float backlit = pow(clamp(dot(V, -uSunDir) * 0.5 + 0.5, 0.0, 1.0), 4.0);
  float rim = pow(1.0 - NoUp, 2.0);
  vec3 scatter = uScatterColor * height * (0.22 + 0.78 * backlit) * (0.15 + 0.85 * rim) * uScatter;
  scatter += uScatterColor * 0.22 * height * height;

  // Deep body darkens sharply in the troughs — that trough/crest contrast is
  // what makes a real sea read as heavy rather than as a bright pool.
  vec3 body = uDeepColor * (0.30 + 0.70 * NoUp) + scatter;

  vec3 col = mix(body, refl, F) + glint;

  // ------------------------------------------------ foam on top
  vec3 foamLit = uFoamColor * (uSkyAmbient * 1.6 + uSunColor * NoLs * 0.20);
  col = mix(col, foamLit, foam);

  // ------------------------------------------------ aerial perspective
  vec3 viewDir = normalize(vWorld - uCam);
  float fog = 1.0 - exp(-pow(max(vDist, 0.0) * uFogDensity, uFogPower));
  col = mix(col, sampleSky(viewDir), clamp(fog, 0.0, 1.0));

  gl_FragColor = vec4(col, 1.0);
}
`;

export function createOceanSurface(opts) {
  const geometry = buildRadialGrid(opts.rings, opts.segments, opts.rMin, opts.rMax);

  const material = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    side: THREE.DoubleSide,
    uniforms: {
      uDisp0: { value: null },
      uDisp1: { value: null },
      uDisp2: { value: null },
      uFoam0: { value: null },
      uFoam1: { value: null },
      uFoam2: { value: null },
      uSky: { value: null },
      uL: { value: new THREE.Vector3(1, 1, 1) },
      uCam: { value: new THREE.Vector3() },
      uFade: { value: new THREE.Vector4(700, 2600, 190, 760) },
      uSunDir: { value: new THREE.Vector3(0.55, 0.09, -0.83).normalize() },
      uSunColor: { value: new THREE.Color(1.0, 0.72, 0.55) },
      uDeepColor: { value: new THREE.Color(0.0042, 0.0272, 0.0350) },
      uScatterColor: { value: new THREE.Color(0.0155, 0.0980, 0.0910) },
      uFoamColor: { value: new THREE.Color(0.62, 0.665, 0.695) },
      uSkyAmbient: { value: new THREE.Color(0.06, 0.065, 0.085) },
      uFoamAmount: { value: 0.46 },
      uFoamSharpness: { value: 0.13 },
      uFoamCrest: { value: 0.02 },
      uFogDensity: { value: 1 / 2300 },
      uFogPower: { value: 1.15 },
      uSpecular: { value: 0.40 },
      uRoughness: { value: 0.070 },
      uScatter: { value: 1.0 },
      uDetailFalloff: { value: 2600 }
    }
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  return mesh;
}
