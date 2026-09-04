import * as THREE from 'three';
import { NOISE, SKY_GLSL } from './shaders/common.js';

const VERT = /* glsl */ `
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const FRAG = /* glsl */ `
precision highp float;
${NOISE}
${SKY_GLSL}

uniform vec2 uRes;

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  // inverse of three.js equirectUv()
  float phi = (uv.x - 0.5) * 6.28318530718;
  float theta = (uv.y - 0.5) * 3.14159265359;
  float ct = cos(theta);
  vec3 dir = normalize(vec3(ct * cos(phi), sin(theta), ct * sin(phi)));

  gl_FragColor = vec4(skyRadiance(dir), 1.0);
}
`;

export class Sky {
  constructor(renderer, width = 1024, height = 512) {
    this.renderer = renderer;

    this.target = new THREE.WebGLRenderTarget(width, height, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.RepeatWrapping,      // seamless across the azimuth wrap
      wrapT: THREE.ClampToEdgeWrapping,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false
    });
    this.target.texture.colorSpace = THREE.NoColorSpace;
    this.target.texture.mapping = THREE.EquirectangularReflectionMapping;

    this.uniforms = {
      uRes: { value: new THREE.Vector2(width, height) },
      uZenithColor:  { value: new THREE.Color(0.0155, 0.0205, 0.0345) },
      uHorizonColor: { value: new THREE.Color(0.0930, 0.0900, 0.1040) },
      uHazeColor:    { value: new THREE.Color(0.1620, 0.1380, 0.1450) },
      uGlowColor:    { value: new THREE.Color(0.2200, 0.1080, 0.0950) },
      uCloudDark:    { value: new THREE.Color(0.0180, 0.0195, 0.0290) },
      uCloudLit:     { value: new THREE.Color(0.0880, 0.0810, 0.0930) },
      uSunColor:     { value: new THREE.Color(0.90, 0.56, 0.40) },
      uSunDir:       { value: new THREE.Vector3(0.62, 0.055, -0.78).normalize() },
      uHazeStrength: { value: 0.85 },
      uHazeSharp:    { value: 9.0 },
      uHazeFalloff:  { value: 0.30 },
      uGlowStrength: { value: 0.85 },
      uSunIntensity: { value: 0.30 },
      uCloudCover:   { value: 0.45 },
      uCloudContrast:{ value: 1.35 },
      uCloudLight:   { value: 0.55 },
      uCloudOpacity: { value: 0.78 },
      uCloudScale:   { value: 3.2 },
      uCloudBase:    { value: 0.055 },
      uCloudOffset:  { value: new THREE.Vector2() }
    };

    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: VERT,
      fragmentShader: FRAG,
      depthTest: false,
      depthWrite: false
    });

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);

    this._t = 0;
    this.windDrift = new THREE.Vector2(0.0075, 0.0022);
  }

  get texture() {
    return this.target.texture;
  }

  /** Average sky radiance — used as the ambient term for the water body. */
  ambient(out = new THREE.Color()) {
    const u = this.uniforms;
    const h = u.uHazeColor.value;
    out.copy(u.uHorizonColor.value).lerp(u.uZenithColor.value, 0.62);
    out.r += h.r * 0.18;
    out.g += h.g * 0.18;
    out.b += h.b * 0.18;
    return out;
  }

  update(dt) {
    this._t += dt;
    this.uniforms.uCloudOffset.value.set(
      this._t * this.windDrift.x,
      this._t * this.windDrift.y
    );

    const prev = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.target);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(prev);
  }
}
