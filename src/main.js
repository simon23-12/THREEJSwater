import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import GUI from 'three/addons/libs/lil-gui.module.min.js';

import { Sky } from './sky.js';
import { OceanSim } from './ocean/OceanSim.js';
import { createOceanSurface } from './ocean/OceanSurface.js';

/* ================================================================== *
 * Configuration — a moderate, wind-driven dusk sea.
 * ================================================================== */
const CONFIG = {
  fftSize: 256,
  wind: {
    windSpeed: 10.5,
    windAngle: 1.95,
    fetch: 140000,
    depth: 180,
    gamma: 3.3,
    swellBoost: 2.2
  },
  // Three FFT cascades split the wave spectrum by wavenumber so nothing is
  // simulated twice. L is the world-space tile size; the band boundaries are
  // placed at 2*pi*6/L of the *next* cascade, the usual split point.
  //   0: swell + long chop   (wavelengths ~8.5 m .. 360 m)
  //   1: mid chop            (~1.8 m .. 8.5 m)
  //   2: ripples             (below ~1.8 m)
  // foamGain rises with wave size: sub-metre ripples fold constantly but do
  // not actually break, so cascade 2 barely contributes whitecaps.
  cascades: [
    { L: 360, lambda: 1.10, amplitude: 1.05, foamGain: 1.00, boundLow: 0.0001, boundHigh: 0.739, shortCut: 0.000 },
    { L: 51,  lambda: 1.02, amplitude: 1.00, foamGain: 0.85, boundLow: 0.739,  boundHigh: 3.427, shortCut: 0.120 },
    { L: 11,  lambda: 1.22, amplitude: 1.00, foamGain: 0.28, boundLow: 3.427,  boundHigh: 1e9,   shortCut: 0.055 }
  ],
  foam: { decay: 0.40, drain: 0.06, gain: 1.6, threshold: 0.16 },
  grid: { rings: 400, segments: 512, rMin: 0.09, rMax: 17000 }
};

/* ================================================================== *
 * Renderer
 * ================================================================== */
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  powerPreference: 'high-performance'
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.NoToneMapping;      // graded manually in the final pass
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;

if (!renderer.capabilities.isWebGL2) {
  document.getElementById('warn').style.display = 'block';
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.5, 40000);
camera.position.set(0, 13.5, 0);

/* ================================================================== *
 * Sky, simulation, surface
 * ================================================================== */
const sky = new Sky(renderer, 1024, 512);
scene.background = sky.texture;

const sim = new OceanSim(renderer, {
  size: CONFIG.fftSize,
  cascades: CONFIG.cascades,
  wind: CONFIG.wind,
  foam: CONFIG.foam
});

const ocean = createOceanSurface(CONFIG.grid);
scene.add(ocean);

const wu = ocean.material.uniforms;
wu.uL.value.set(CONFIG.cascades[0].L, CONFIG.cascades[1].L, CONFIG.cascades[2].L);

function bindSimTextures() {
  const d = sim.displacementTextures;
  const f = sim.foamTextures;
  wu.uDisp0.value = d[0]; wu.uDisp1.value = d[1]; wu.uDisp2.value = d[2];
  wu.uFoam0.value = f[0]; wu.uFoam1.value = f[1]; wu.uFoam2.value = f[2];
  wu.uSky.value = sky.texture;
}

/* ================================================================== *
 * Post-processing: bloom + ACES grade + vignette + grain
 * ================================================================== */
const GradeShader = {
  uniforms: {
    tDiffuse: { value: null },
    uExposure: { value: 1.55 },
    uContrast: { value: 1.06 },
    uSaturation: { value: 1.04 },
    uVignette: { value: 0.34 },
    uGrain: { value: 0.022 },
    uTime: { value: 0 }
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    precision highp float;
    uniform sampler2D tDiffuse;
    uniform float uExposure, uContrast, uSaturation, uVignette, uGrain, uTime;
    varying vec2 vUv;

    vec3 aces(vec3 x) {
      const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }

    vec3 toSRGB(vec3 c) {
      c = max(c, vec3(0.0));
      return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
    }

    float rand(vec2 p) {
      return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
      vec3 col = texture2D(tDiffuse, vUv).rgb * uExposure;

      col = aces(col);

      float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
      col = mix(vec3(l), col, uSaturation);
      col = clamp((col - 0.5) * uContrast + 0.5, 0.0, 1.0);

      vec2 d = vUv - 0.5;
      col *= 1.0 - uVignette * dot(d, d) * 2.2;

      col += (rand(vUv * 1024.0 + uTime) - 0.5) * uGrain;

      gl_FragColor = vec4(toSRGB(col), 1.0);
    }
  `
};

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight), 0.32, 0.85, 0.62
);
composer.addPass(bloom);

const gradePass = new ShaderPass(GradeShader);
gradePass.renderToScreen = true;
composer.addPass(gradePass);

/* ================================================================== *
 * Camera rig: slow drift + drag to look, WASD to move
 * ================================================================== */
const rig = {
  yaw: Math.PI,          // look down -Z
  pitch: -0.150,
  pos: new THREE.Vector3(0, 13.5, 0),
  vel: new THREE.Vector3(),
  drift: true,
  bob: 1.0
};

let dragging = false;
let lastX = 0, lastY = 0;
const keys = new Set();

canvas.addEventListener('pointerdown', (e) => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', (e) => {
  dragging = false;
  try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
});
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  rig.yaw -= (e.clientX - lastX) * 0.0022;
  rig.pitch = THREE.MathUtils.clamp(rig.pitch - (e.clientY - lastY) * 0.0018, -1.2, 0.9);
  lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener('wheel', (e) => {
  camera.fov = THREE.MathUtils.clamp(camera.fov + e.deltaY * 0.02, 18, 75);
  camera.updateProjectionMatrix();
  e.preventDefault();
}, { passive: false });

window.addEventListener('keydown', (e) => {
  keys.add(e.code);
  if (e.code === 'KeyH') gui._hidden ? gui.show() : gui.hide();
});
window.addEventListener('keyup', (e) => keys.delete(e.code));

/* ================================================================== *
 * GUI
 * ================================================================== */
const gui = new GUI({ title: 'Ocean' });
const params = {
  timeScale: 1.0,
  windSpeed: CONFIG.wind.windSpeed,
  windAngle: CONFIG.wind.windAngle,
  fetchKm: CONFIG.wind.fetch / 1000,
  choppiness: 1.0,
  foamAmount: wu.uFoamAmount.value,
  foamSharpness: wu.uFoamSharpness.value,
  foamCrest: wu.uFoamCrest.value,
  foamDecay: CONFIG.foam.decay,
  fogDistance: 2300,
  fogPower: wu.uFogPower.value,
  specular: wu.uSpecular.value,
  roughness: wu.uRoughness.value,
  scatter: wu.uScatter.value,
  exposure: GradeShader.uniforms.uExposure.value,
  bloom: bloom.strength,
  cloudCover: sky.uniforms.uCloudCover.value,
  cloudOpacity: sky.uniforms.uCloudOpacity.value,
  glow: sky.uniforms.uGlowStrength.value,
  cameraHeight: 13.5,
  drift: true
};

const baseLambda = CONFIG.cascades.map((c) => c.lambda);

const fSea = gui.addFolder('Seegang');
fSea.add(params, 'timeScale', 0, 3, 0.01).name('Zeit');
fSea.add(params, 'windSpeed', 3, 22, 0.1).name('Wind m/s').onFinishChange(rebuild);
fSea.add(params, 'windAngle', 0, 6.28, 0.01).name('Windrichtung').onFinishChange(rebuild);
fSea.add(params, 'fetchKm', 10, 500, 1).name('Fetch km').onFinishChange(rebuild);
fSea.add(params, 'choppiness', 0, 2, 0.01).name('Steilheit').onChange((v) => {
  sim.cascades.forEach((c, i) => { c.lambda = baseLambda[i] * v; });
});

const fFoam = gui.addFolder('Schaum');
fFoam.add(params, 'foamAmount', 0, 3, 0.01).name('Menge').onChange((v) => (wu.uFoamAmount.value = v));
fFoam.add(params, 'foamSharpness', 0.02, 0.5, 0.01).name('Härte').onChange((v) => (wu.uFoamSharpness.value = v));
fFoam.add(params, 'foamCrest', -0.6, 1.0, 0.01).name('Kammschwelle').onChange((v) => (wu.uFoamCrest.value = v));
fFoam.add(params, 'foamDecay', 0.02, 1.5, 0.01).name('Abklingen').onChange((v) => (sim.foamParams.decay = v));

const fWater = gui.addFolder('Wasser');
fWater.add(params, 'specular', 0, 3, 0.01).name('Glitzern').onChange((v) => (wu.uSpecular.value = v));
fWater.add(params, 'roughness', 0.01, 0.5, 0.005).name('Rauheit').onChange((v) => (wu.uRoughness.value = v));
fWater.add(params, 'scatter', 0, 3, 0.01).name('Streuung').onChange((v) => (wu.uScatter.value = v));
fWater.addColor({ c: '#' + wu.uDeepColor.value.getHexString() }, 'c').name('Tiefenfarbe')
  .onChange((v) => wu.uDeepColor.value.set(v));
fWater.addColor({ c: '#' + wu.uScatterColor.value.getHexString() }, 'c').name('Streufarbe')
  .onChange((v) => wu.uScatterColor.value.set(v));
fWater.add(params, 'fogDistance', 300, 12000, 10).name('Dunst-Distanz')
  .onChange((v) => (wu.uFogDensity.value = 1 / v));
fWater.add(params, 'fogPower', 0.6, 3, 0.01).name('Dunst-Kurve').onChange((v) => (wu.uFogPower.value = v));

const fSky = gui.addFolder('Himmel');
fSky.add({ v: 4.5 }, 'v', 0.5, 14, 0.1).name('Wolkengröße')
  .onChange((v) => (sky.uniforms.uCloudScale.value = v));
fSky.add(params, 'cloudCover', 0.05, 0.9, 0.01).name('Bewölkung').onChange((v) => (sky.uniforms.uCloudCover.value = v));
fSky.add(params, 'cloudOpacity', 0, 1, 0.01).name('Wolkendichte').onChange((v) => (sky.uniforms.uCloudOpacity.value = v));
fSky.add({ v: sky.uniforms.uCloudContrast.value }, 'v', 0.5, 6, 0.05).name('Wolkenkontrast')
  .onChange((v) => (sky.uniforms.uCloudContrast.value = v));
fSky.add(params, 'glow', 0, 2, 0.01).name('Horizontglühen').onChange((v) => (sky.uniforms.uGlowStrength.value = v));
fSky.addColor({ c: '#' + sky.uniforms.uHorizonColor.value.getHexString() }, 'c').name('Horizontfarbe')
  .onChange((v) => sky.uniforms.uHorizonColor.value.set(v));
fSky.addColor({ c: '#' + sky.uniforms.uZenithColor.value.getHexString() }, 'c').name('Zenitfarbe')
  .onChange((v) => sky.uniforms.uZenithColor.value.set(v));
fSky.addColor({ c: '#' + sky.uniforms.uCloudDark.value.getHexString() }, 'c').name('Wolke dunkel')
  .onChange((v) => sky.uniforms.uCloudDark.value.set(v));

const fView = gui.addFolder('Bild');
fView.add(params, 'exposure', 0.2, 4, 0.01).name('Belichtung')
  .onChange((v) => (GradeShader.uniforms.uExposure.value = v));
fView.add(params, 'bloom', 0, 1.5, 0.01).name('Bloom').onChange((v) => (bloom.strength = v));
fView.add(params, 'cameraHeight', 2, 120, 0.5).name('Kamerahöhe').onChange((v) => (rig.pos.y = v));
fView.add(params, 'drift').name('Kamera-Drift').onChange((v) => (rig.drift = v));
gui.close();

function rebuild() {
  CONFIG.wind.windSpeed = params.windSpeed;
  CONFIG.wind.windAngle = params.windAngle;
  CONFIG.wind.fetch = params.fetchKm * 1000;
  sim.rebuild(CONFIG.wind);
}

/* ================================================================== *
 * Loop
 * ================================================================== */
const clock = new THREE.Clock();
const forward = new THREE.Vector3();
const right = new THREE.Vector3();
const ambient = new THREE.Color();
let elapsed = 0;
let frames = 0, fpsAccum = 0;
const hud = document.getElementById('hud');

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloom.setSize(w, h);
}
window.addEventListener('resize', resize);

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  elapsed += dt;

  // --- camera ---
  forward.set(Math.sin(rig.yaw) * Math.cos(rig.pitch), Math.sin(rig.pitch), Math.cos(rig.yaw) * Math.cos(rig.pitch));
  right.set(Math.sin(rig.yaw - Math.PI / 2), 0, Math.cos(rig.yaw - Math.PI / 2));

  const speed = keys.has('ShiftLeft') ? 90 : 22;
  if (keys.has('KeyW')) rig.pos.addScaledVector(forward, speed * dt);
  if (keys.has('KeyS')) rig.pos.addScaledVector(forward, -speed * dt);
  if (keys.has('KeyA')) rig.pos.addScaledVector(right, -speed * dt);
  if (keys.has('KeyD')) rig.pos.addScaledVector(right, speed * dt);
  if (keys.has('KeyQ')) rig.pos.y -= speed * dt;
  if (keys.has('KeyE')) rig.pos.y += speed * dt;

  if (rig.drift) rig.pos.z -= dt * 1.6;

  // gentle swell-induced motion of the platform we are standing on
  const bobY = Math.sin(elapsed * 0.47) * 0.55 + Math.sin(elapsed * 0.83 + 1.7) * 0.28;
  const bobP = Math.sin(elapsed * 0.31 + 0.4) * 0.006 + Math.sin(elapsed * 0.71) * 0.003;
  const bobR = Math.sin(elapsed * 0.27 + 2.1) * 0.012;

  camera.position.set(rig.pos.x, rig.pos.y + bobY * rig.bob, rig.pos.z);
  camera.rotation.set(0, 0, 0);
  camera.rotateY(rig.yaw);
  camera.rotateX(rig.pitch + bobP * rig.bob);
  camera.rotateZ(bobR * rig.bob);

  // --- simulate ---
  sky.update(dt);
  sim.update(dt, params.timeScale);
  bindSimTextures();

  sky.ambient(ambient);
  wu.uSkyAmbient.value.copy(ambient);
  wu.uSunDir.value.copy(sky.uniforms.uSunDir.value);
  wu.uCam.value.copy(camera.position);

  ocean.position.set(camera.position.x, 0, camera.position.z);

  GradeShader.uniforms.uTime.value = elapsed;

  composer.render();

  // --- hud ---
  frames++; fpsAccum += dt;
  if (fpsAccum > 0.5) {
    hud.textContent = `${Math.round(frames / fpsAccum)} fps · ${CONFIG.fftSize}² FFT × 3 Kaskaden · H = Regler`;
    frames = 0; fpsAccum = 0;
  }
}

// handy for inspecting the running scene from the console
window.__ocean = { camera, rig, ocean, sim, sky, params, THREE };

resize();
animate();
