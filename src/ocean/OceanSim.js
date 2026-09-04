import * as THREE from 'three';
import { COMPLEX } from '../shaders/common.js';
import { buildH0Texture, buildButterflyTexture } from './spectrum.js';

const FS_VERT = /* glsl */ `
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

/* --- pass 1: evolve the frozen spectrum to time t ------------------ */
const SPECTRUM_FRAG = /* glsl */ `
precision highp float;
${COMPLEX}
uniform sampler2D uH0;
uniform float uN;
uniform float uL;
uniform float uTime;

void main() {
  vec2 xy = floor(gl_FragCoord.xy);
  vec2 uv = (xy + 0.5) / uN;

  vec2 k = (xy - uN * 0.5) * (6.28318530718 / uL);
  float kk = max(length(k), 1e-6);
  float w = sqrt(9.81 * kk);

  vec4 h0 = texture2D(uH0, uv);
  float c = cos(w * uTime);
  float s = sin(w * uTime);

  // h(k,t) = h0(k) e^{iwt} + conj(h0(-k)) e^{-iwt}
  vec2 h = cmul(h0.xy, vec2(c, s)) + cmul(h0.zw, vec2(c, -s));

  vec2 nk = k / kk;
  vec2 mih = vec2(h.y, -h.x);          // -i * h
  vec2 Dx = nk.x * mih;
  vec2 Dz = nk.y * mih;

  // Two real fields per complex transform: iFFT(A + iB) -> (a, b).
  vec2 A = vec2(h.x - Dx.y, h.y + Dx.x);   // height + i * Dx
  gl_FragColor = vec4(A, Dz);              // .zw carries Dz
}
`;

/* --- pass 2: butterfly stages (both complex channels at once) ------ */
const BUTTERFLY_FRAG = /* glsl */ `
precision highp float;
${COMPLEX}
uniform sampler2D uButterfly;
uniform sampler2D uSrc;
uniform float uStage;
uniform float uStages;
uniform float uN;
uniform float uVertical;

void main() {
  vec2 xy = floor(gl_FragCoord.xy);
  float idx = mix(xy.x, xy.y, uVertical);

  vec4 bf = texture2D(uButterfly, vec2((uStage + 0.5) / uStages, (idx + 0.5) / uN));
  vec2 w = bf.xy;

  vec2 topUV, botUV;
  if (uVertical < 0.5) {
    topUV = vec2((bf.z + 0.5) / uN, (xy.y + 0.5) / uN);
    botUV = vec2((bf.w + 0.5) / uN, (xy.y + 0.5) / uN);
  } else {
    topUV = vec2((xy.x + 0.5) / uN, (bf.z + 0.5) / uN);
    botUV = vec2((xy.x + 0.5) / uN, (bf.w + 0.5) / uN);
  }

  vec4 P = texture2D(uSrc, topUV);
  vec4 Q = texture2D(uSrc, botUV);
  gl_FragColor = vec4(P.xy + cmul(w, Q.xy), P.zw + cmul(w, Q.zw));
}
`;

/* --- pass 3: unshuffle + pack into a displacement map -------------- */
const ASSEMBLE_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uSrc;
uniform float uN;
uniform float uLambda;
uniform float uAmplitude;

void main() {
  vec2 xy = floor(gl_FragCoord.xy);
  float sgn = 1.0 - 2.0 * mod(xy.x + xy.y, 2.0);   // undo the centred k-lattice
  vec4 r = texture2D(uSrc, (xy + 0.5) / uN) * sgn * uAmplitude;
  // r.x = height, r.y = Dx, r.z = Dz
  gl_FragColor = vec4(r.y * uLambda, r.x, r.z * uLambda, 1.0);
}
`;

/* --- pass 4: whitecap foam, accumulated and decayed over time ------ */
const FOAM_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uDisp;
uniform sampler2D uPrev;
uniform float uN;
uniform float uL;
uniform float uDt;
uniform float uDecay;
uniform float uDrain;
uniform float uGain;
uniform float uThreshold;

void main() {
  vec2 uv = (floor(gl_FragCoord.xy) + 0.5) / uN;
  float t = 1.0 / uN;
  float e = uL / uN;

  vec3 xp = texture2D(uDisp, uv + vec2(t, 0.0)).xyz;
  vec3 xm = texture2D(uDisp, uv - vec2(t, 0.0)).xyz;
  vec3 zp = texture2D(uDisp, uv + vec2(0.0, t)).xyz;
  vec3 zm = texture2D(uDisp, uv - vec2(0.0, t)).xyz;

  float inv = 1.0 / (2.0 * e);
  float Jxx = 1.0 + (xp.x - xm.x) * inv;
  float Jzz = 1.0 + (zp.z - zm.z) * inv;
  float Jxz = (zp.x - zm.x) * inv;
  float Jzx = (xp.z - xm.z) * inv;

  float J = Jxx * Jzz - Jxz * Jzx;      // < 0 means the surface folds -> it breaks

  float inject = clamp((uThreshold - J) * uGain, 0.0, 1.0);
  float prev = texture2D(uPrev, uv).r;
  float decayed = prev * exp(-uDecay * uDt) - uDrain * uDt;

  gl_FragColor = vec4(clamp(max(decayed, inject), 0.0, 1.0));
}
`;

function makeRT(size, type, linear, repeat) {
  const rt = new THREE.WebGLRenderTarget(size, size, {
    type,
    format: THREE.RGBAFormat,
    minFilter: linear ? THREE.LinearFilter : THREE.NearestFilter,
    magFilter: linear ? THREE.LinearFilter : THREE.NearestFilter,
    wrapS: repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping,
    wrapT: repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false
  });
  rt.texture.colorSpace = THREE.NoColorSpace;
  return rt;
}

export class OceanSim {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {{size:number, cascades:Array, wind:object, foam:object}} config
   */
  constructor(renderer, config) {
    this.renderer = renderer;
    this.N = config.size;
    this.stages = Math.log2(this.N);
    this.time = 0;

    const gl = renderer.getContext();
    gl.getExtension('EXT_color_buffer_float');
    // Float RTs are required for the FFT; linear filtering of them is an
    // extension, so fall back to half-float for the sampled maps if missing.
    this.linearFloat = !!gl.getExtension('OES_texture_float_linear');
    const sampledType = this.linearFloat ? THREE.FloatType : THREE.HalfFloatType;

    this.quadScene = new THREE.Scene();
    this.quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quadGeo = new THREE.PlaneGeometry(2, 2);
    this.quadMesh = new THREE.Mesh(this.quadGeo, null);
    this.quadMesh.frustumCulled = false;
    this.quadScene.add(this.quadMesh);

    this.butterflyTex = buildButterflyTexture(this.N);

    this.matSpectrum = this._mat(SPECTRUM_FRAG, {
      uH0: { value: null }, uN: { value: this.N }, uL: { value: 1 }, uTime: { value: 0 }
    });
    this.matButterfly = this._mat(BUTTERFLY_FRAG, {
      uButterfly: { value: this.butterflyTex }, uSrc: { value: null },
      uStage: { value: 0 }, uStages: { value: this.stages },
      uN: { value: this.N }, uVertical: { value: 0 }
    });
    this.matAssemble = this._mat(ASSEMBLE_FRAG, {
      uSrc: { value: null }, uN: { value: this.N },
      uLambda: { value: 1 }, uAmplitude: { value: 1 }
    });
    this.matFoam = this._mat(FOAM_FRAG, {
      uDisp: { value: null }, uPrev: { value: null },
      uN: { value: this.N }, uL: { value: 1 }, uDt: { value: 0.016 },
      uDecay: { value: 0.28 }, uDrain: { value: 0.06 },
      uGain: { value: 1.4 }, uThreshold: { value: 0.62 }
    });

    this.cascades = config.cascades.map((c, i) => {
      const h0 = buildH0Texture(this.N, c.L, {
        ...config.wind,
        boundLow: c.boundLow,
        boundHigh: c.boundHigh,
        shortCut: c.shortCut,
        seed: 1337 + i * 7919
      });
      return {
        ...c,
        h0,
        ping: [makeRT(this.N, THREE.FloatType, false, false), makeRT(this.N, THREE.FloatType, false, false)],
        disp: makeRT(this.N, sampledType, true, true),
        foam: [makeRT(this.N, sampledType, true, true), makeRT(this.N, sampledType, true, true)],
        foamIdx: 0
      };
    });

    this.foamParams = config.foam;
  }

  _mat(fragmentShader, uniforms) {
    return new THREE.ShaderMaterial({
      uniforms,
      vertexShader: FS_VERT,
      fragmentShader,
      depthTest: false,
      depthWrite: false
    });
  }

  _blit(material, target) {
    this.quadMesh.material = material;
    const prev = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.quadScene, this.quadCamera);
    this.renderer.setRenderTarget(prev);
  }

  /** Rebuild h0 after the wind/sea-state parameters change. */
  rebuild(wind) {
    this.cascades.forEach((c, i) => {
      c.h0.dispose();
      c.h0 = buildH0Texture(this.N, c.L, {
        ...wind,
        boundLow: c.boundLow,
        boundHigh: c.boundHigh,
        shortCut: c.shortCut,
        seed: 1337 + i * 7919
      });
    });
  }

  update(dt, speed) {
    this.time += dt * speed;
    const dtc = Math.min(dt, 0.05);

    for (const c of this.cascades) {
      // 1. spectrum at time t
      this.matSpectrum.uniforms.uH0.value = c.h0;
      this.matSpectrum.uniforms.uL.value = c.L;
      this.matSpectrum.uniforms.uTime.value = this.time;
      this._blit(this.matSpectrum, c.ping[0]);

      // 2. 2-D inverse FFT: log2(N) horizontal + log2(N) vertical butterflies
      let src = 0;
      for (let axis = 0; axis < 2; axis++) {
        this.matButterfly.uniforms.uVertical.value = axis;
        for (let stage = 0; stage < this.stages; stage++) {
          this.matButterfly.uniforms.uStage.value = stage;
          this.matButterfly.uniforms.uSrc.value = c.ping[src].texture;
          this._blit(this.matButterfly, c.ping[1 - src]);
          src = 1 - src;
        }
      }

      // 3. unshuffle -> (Dx, height, Dz)
      this.matAssemble.uniforms.uSrc.value = c.ping[src].texture;
      this.matAssemble.uniforms.uLambda.value = c.lambda;
      this.matAssemble.uniforms.uAmplitude.value = c.amplitude;
      this._blit(this.matAssemble, c.disp);

      // 4. foam accumulation
      const f = this.foamParams;
      this.matFoam.uniforms.uDisp.value = c.disp.texture;
      this.matFoam.uniforms.uPrev.value = c.foam[c.foamIdx].texture;
      this.matFoam.uniforms.uL.value = c.L;
      this.matFoam.uniforms.uDt.value = dtc;
      this.matFoam.uniforms.uDecay.value = f.decay;
      this.matFoam.uniforms.uDrain.value = f.drain;
      this.matFoam.uniforms.uGain.value = f.gain * c.foamGain;
      this.matFoam.uniforms.uThreshold.value = f.threshold;
      this._blit(this.matFoam, c.foam[1 - c.foamIdx]);
      c.foamIdx = 1 - c.foamIdx;
    }
  }

  get displacementTextures() {
    return this.cascades.map((c) => c.disp.texture);
  }

  get foamTextures() {
    return this.cascades.map((c) => c.foam[c.foamIdx].texture);
  }
}
