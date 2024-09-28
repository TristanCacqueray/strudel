/*
shader.mjs - <short description TODO>
Copyright (C) 2022 Strudel contributors - see <https://github.com/tidalcycles/strudel/blob/main/packages/canvas/pianoroll.mjs>
Copyright (C) 2024 Tristan de Cacqueray
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details. You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

import { PicoGL } from "picogl";

import { Pattern, noteToMidi, freqToMidi } from '@strudel/core';
import { getTheme, getDrawContext } from './draw.mjs';

const vertexShader = `#version 300 es
layout(location=0) in vec2 position;
out vec2 fragPos;
void main() {
  fragPos = position;
  gl_Position = vec4(position, 1, 1);
}
`;

const fragmentShader = `#version 300 es
precision highp float;
in vec2 fragPos;
out vec4 fragColor;

// The scene values
uniform vec2  iResolution;
uniform float iTime;

// The modulation values
uniform float icolor;
#define moveFWD (iTime / 10.)
uniform float track1[5];
uniform float track2[5];

// TODO: inline these tweaks and remove the define indirection
#define t1f 1.0
#define t1p1 (t1f * track1[0])
#define t1p2 (t1f * track1[1])
#define t1p3 (-2. + track1[2])
#define t1p4 (t1f*track1[3])
#define t1p5 (t1f*track1[4])

#define t2f 1.0
#define t2p1 (t2f*track2[0])
#define t2p2 (t2f*track2[1])
#define t2p3 (-2. + t2f*track2[2])
#define t2p4 (t2f*track2[3])
#define t2p5 (t2f*track2[4])


// Forked from https://www.shadertoy.com/view/7lKSWW
// Created by mrange in 2021-12-28
// CC0: Truchet + Kaleidoscope FTW
//  Bit of experimenting with kaleidoscopes and truchet turned out nice
//  Quite similar to an earlier shader I did but I utilized a different truchet pattern this time

// The shader has been adapted^Wbutchered to support new parameters
// to be controlled from midi/audio inputs with animation-fractal.

// The new inputs are:
// - tXpY is the sum of the midi velocity of the track X's pitch Y%4
// - icolor/imoveFWD is the sound RMS volume of a track audio stem.

#define PI              3.141592654
#define TAU             (2.0*PI)
#define RESOLUTION      iResolution
#define TIME            (moveFWD * 6.)
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))
#define PCOS(x)         (0.5+0.5*cos(x))

// License: Unknown, author: Unknown, found: don't remember
vec4 alphaBlend(vec4 back, vec4 front) {
  float w = front.w + back.w*(1.0-front.w);
  vec3 xyz = (front.xyz*front.w + back.xyz*back.w*(1.0-front.w))/w;
  return w > 0.0 ? vec4(xyz, w) : vec4(0.0);
}

// License: Unknown, author: Unknown, found: don't remember
vec3 alphaBlend(vec3 back, vec4 front) {
  return mix(back, front.xyz, front.w);
}

// License: Unknown, author: Unknown, found: don't remember
float hash(float co) {
  return fract(sin(co*12.9898) * 13758.5453);
}

// License: Unknown, author: Unknown, found: don't remember
float hash(vec2 p) {
  float a = dot(p, vec2 (127.1, 311.7));
  return fract(sin (a)*43758.5453123);
}

// License: Unknown, author: Unknown, found: don't remember
float tanh_approx(float x) {
  //  Found this somewhere on the interwebs
  //  return tanh(x);
  float x2 = x*x;
  return clamp(x*(27.0 + x2)/(27.0+9.0*x2), -1.0, 1.0);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/smin
float pmin(float a, float b, float k) {
  float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/index.htm
vec3 postProcess(vec3 col, vec2 q) {
  col = clamp(col, 0.0, 1.0);
  col = pow(col, vec3(1.0/2.2));
  col = col*0.6+0.4*col*col*(3.0-2.0*col);
  col = mix(col, vec3(dot(col, vec3(0.33))), -0.4);
  col *=0.5+0.5*pow(19.0*q.x*q.y*(1.0-q.x)*(1.0-q.y),0.7);
  return col;
}

float pmax(float a, float b, float k) {
  return -pmin(-a, -b, k);
}

float pabs(float a, float k) {
  return pmax(a, -a, k);
}

vec2 toPolar(vec2 p) {
  return vec2(length(p), atan(p.y, p.x));
}

vec2 toRect(vec2 p) {
  return vec2(p.x*cos(p.y), p.x*sin(p.y));
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
float modMirror1(inout float p, float size) {
  float halfsize = size*0.5;
  float c = floor((p + halfsize)/size);
  p = mod(p + halfsize,size) - halfsize;
  p *= mod(c, 2.0)*2.0 - 1.0;
  return c;
}

float smoothKaleidoscope(inout vec2 p, float sm, float rep, float kmod) {
  vec2 hp = p;

  vec2 hpp = toPolar(hp);
  float rn = modMirror1(hpp.y, TAU/rep);

  float sa = PI/rep - pabs(PI/rep - abs(hpp.y), sm);
  hpp.y = sin(kmod) + sign(hpp.y)*(sa);

  hp = toRect(hpp);

  p = hp;

  return rn;
}

// The path function
vec3 offset(float z) {
  float a = z;
  vec2 p = -0.075*(vec2(cos(a), sin(a*sqrt(2.0))) + vec2(cos(a*sqrt(0.75)), sin(a*sqrt(0.5))));
  return vec3(p, z);
}

// The derivate of the path function
//  Used to generate where we are looking
vec3 doffset(float z) {
  float eps = 0.1;
  return 0.5*(offset(z + eps) - offset(z - eps))/eps;
}

// The second derivate of the path function
//  Used to generate tilt
vec3 ddoffset(float z) {
  float eps = 0.1;
  return 0.125*(doffset(z + eps) - doffset(z - eps))/eps;
}

vec2 cell_df(float r, vec2 np, vec2 mp, vec2 off, float cmod) {
  const vec2 n0 = normalize(vec2(1.0, 1.0));
  const vec2 n1 = normalize(vec2(1.0, -1.0));

  np += off;
  mp -= off;

  float hh = hash(np);
  float h0 = hh;

  vec2  p0 = mp;
  // p0 = abs(p0);
  p0 = abs(p0) + (0.05) * (0.4 + sin(cmod) * 0.4);
  p0 -= 0.4;
  float d0 = length(p0);
  float d1 = abs(d0-r);

  float dot0 = dot(n0, mp);
  float dot1 = dot(n1, mp);

  float d2 = abs(dot0);
  float t2 = dot1;
  d2 = abs(t2) > sqrt(0.5) ? d0 : d2;

  float d3 = abs(dot1);
  float t3 = dot0;
  d3 = abs(t3) > sqrt(0.5) ? d0 : d3;


  float d = d0;
  d = min(d, d1);
  if (h0 > .85)
  {
    d = min(d, d2);
    d = min(d, d3);
  }
  else if(h0 > 0.5)
  {
    d = min(d, d2);
  }
  else if(h0 > 0.15)
  {
    d = min(d, d3);
  }

  return vec2(d, d0-r);
}

vec2 truchet_df(float r, vec2 p, float cmod) {
  vec2 np = floor(p+0.5);
  vec2 mp = fract(p+0.5) - 0.5;
  return cell_df(r, np, mp, vec2(.0), cmod);
}


vec3 palette( float t ) {
    t *= 3.;
    vec3 a = vec3(0.5, 0.5, 0.9);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.1, 0.2, 0.6);

    return a + b*cos( 6.28318*(c*t+d) );
}

vec4 plane(vec3 ro, vec3 rd, vec3 pp, vec3 off, float aa, float n) {
  float h_ = hash(n);
  float h0 = fract(1777.0*h_);
  float h1 = fract(2087.0*h_);
  float h2 = fract(2687.0*h_);
  float h3 = fract(3167.0*h_);
  float h4 = fract(3499.0*h_);

  float l = length(pp - ro);

  vec3 hn;
  vec2 p = (pp-off*vec3(1.0, 1.0, 0.0)).xy;

  float len = length(p);

  float prot1, prot2;
  float pmod1, pmod2;
  float cmod;
  if (mod(n, 2.) == 0.) { prot1 = t1p1, prot2 = -t1p2, pmod1 = t2p3, pmod2 = t2p4, cmod = t1p5; }
  else { prot1 = -t2p1, prot2 = t2p2, pmod1 = t1p3, pmod2 = t1p4, cmod = t2p5; }

  float prot, pmod;
  if (len > .3) { prot = prot2; pmod = pmod2;
  } else { prot = prot1; pmod = pmod1; }
  p *= ROT(prot * 1.5);

  // p *= ROT(0.5*(h4 - 0.5)*TIME);
  float rep = 2.0*round(mix(5.0, 30.0, h2));
  float sm = 0.05*20.0/rep;
  float sn = smoothKaleidoscope(p, sm, rep, pmod);
  p *= ROT(TAU*h0+0.025*TIME);
  float z = mix(0.2, 0.4, h3);
  p /= z;
  p+=0.5+floor(h1*1000.0);
  float tl = tanh_approx(0.33*l);
  float r = mix(0.30, 0.45, PCOS(0.1*n + pmod * 2. + prot));
  vec2 poff = vec2(1. + pmod * .3, prot * .1);
  vec2 d2 = truchet_df(r, p, cmod);

  // d2.x = d1;

  // d2.x = pmin(d2.x, abs(d1) - 0.01, .05);
  d2 *= z;
  float d = d2.x;

  float lw = 0.025*z;
  d -= lw;

  vec3 bcol = palette(abs(d) * 2. + h_ * 2. + icolor * .2);

  vec3 col = mix(bcol, vec3(0.0), smoothstep(aa, -aa, d));
  col = mix(col, vec3(0.0), smoothstep(mix(1.0, -0.5, tl), 1.0, sin(PI*100.0*d)));
  col = mix(col, vec3(0.0), step(d2.y, 0.0));
  float t = smoothstep(aa, -aa, -d2.y-3.0*lw)*mix(0.5, 1.0, smoothstep(aa, -aa, -d2.y-lw));

  // border between the two rings
  float ct = 1.0 - smoothstep(0.0, .02, abs(.3-len));
  col = mix(col, vec3(0.), ct);
  t -= ct * .1;

  return vec4(col, t);
}

vec3 skyColor(vec3 ro, vec3 rd) {
  float d = pow(max(dot(rd, vec3(0.0, 0.0, 1.0)), 0.0), 20.0);
  return vec3(d);
}

vec3 color(vec3 ww, vec3 uu, vec3 vv, vec3 ro, vec2 p) {
  float lp = length(p);
  vec2 np = p + 1.0/RESOLUTION.xy;
  float rdd = (2.0+1.0*tanh_approx(lp));
//  float rdd = 2.0;
  vec3 rd = normalize(p.x*uu + p.y*vv + rdd*ww);
  vec3 nrd = normalize(np.x*uu + np.y*vv + rdd*ww);

  const float planeDist = 1.0-0.25;
  const int furthest = 6;
  const int fadeFrom = max(furthest-5, 0);

  const float fadeDist = planeDist*float(furthest - fadeFrom);
  float nz = floor(ro.z / planeDist);

  vec3 skyCol = skyColor(ro, rd);


  vec4 acol = vec4(0.0);
  const float cutOff = 0.95;
  bool cutOut = false;

  // Steps from nearest to furthest plane and accumulates the color
  for (int i = 1; i <= furthest; ++i) {
    float pz = planeDist*nz + planeDist*float(i);

    float pd = (pz - ro.z)/rd.z;

    if (pd > 0.0 && acol.w < cutOff) {
      vec3 pp = ro + rd*pd;
      vec3 npp = ro + nrd*pd;

      float aa = 3.0*length(pp - npp);

      vec3 off = offset(pp.z);

      vec4 pcol = plane(ro, rd, pp, off, aa, nz+float(i));

      float nz = pp.z-ro.z;
      float fadeIn = smoothstep(planeDist*float(furthest), planeDist*float(fadeFrom), nz);
      float fadeOut = smoothstep(0.0, planeDist*0.1, nz);
      pcol.xyz = mix(skyCol, pcol.xyz, fadeIn + .2);
      pcol.w *= fadeOut;
      pcol = clamp(pcol, 0.0, 1.0);

      acol = alphaBlend(pcol, acol);
    } else {
      cutOut = true;
      break;
    }

  }

  vec3 col = alphaBlend(skyCol, acol);
// To debug cutouts due to transparency
//  col += cutOut ? vec3(1.0, -1.0, 0.0) : vec3(0.0);
  return col;
}

vec3 effect(vec2 p, vec2 q) {
  float tm  = .127 + TIME*0.25;
  vec3 ro   = offset(tm);
  vec3 dro  = doffset(tm);
  vec3 ddro = ddoffset(tm);

  vec3 ww = normalize(dro);
  vec3 uu = normalize(cross(normalize(vec3(0.0,1.0,0.0)+ddro), ww));
  vec3 vv = normalize(cross(ww, uu));

  vec3 col = color(ww, uu, vv, ro, p);

  return col;
}

void main() {
  // Normalized pixel coordinates (from 0 to 1)
  vec2 q = fragPos + vec2(0.5);
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;

  vec3 col = effect(p, q);
  col *= smoothstep(0.0, 4.0, TIME + icolor);
  col = postProcess(col, q);

  fragColor = vec4(col, 1.0);
}
`;

// Setup the shader instance, ideally that should be done like a pattern,
// perhaps supporting multiple instances?
//
// For now, this is a global instance with an hardcoded config.
// Here is the desired API for live implem:
//  $video: shaderCode(glslSource to be loaded in a text editor)
//      .uniform({name: zoom})
//      .array({name: track1, count: 5})
//      .array({name: track2, count: 5})
//
// Then pattern can use:
//  $: note("...")._shader({uniform: "zoom"})
//
// - uniform: merge all the note to set the value
// - array: use the indiviual note value
setTimeout(() => {
  const ctx = getDrawContext("video", {contextType: "webgl2"});

  const app = PicoGL.createApp(ctx);
  const sceneUB = app.createUniformBuffer([
    // The canvas dimensions
    PicoGL.FLOAT_VEC2,
    // The current time
    PicoGL.FLOAT,
  ]);
  sceneUB.set(0, new Float32Array([ctx.canvas.width, ctx.canvas.height]));

  const resolution = new Float32Array([ctx.canvas.width, ctx.canvas.height])

  // track values
  const track1 = new Float32Array(5);
  const track2 = new Float32Array(5);

  // Two triangle to cover the whole canvas
  const afBuffer = app.createVertexBuffer(PicoGL.FLOAT, 2, new Float32Array([
    -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,
     0.5,  0.5,  0.5, -0.5, -0.5, -0.5,
  ]))

  // Setup the arrays
  const afArrays = app.createVertexArray().vertexAttributeBuffer(0, afBuffer);

  app
    .createPrograms([vertexShader, fragmentShader])
    .then(([afProgram]) => {
      const drawAF = app
            .createDrawCall(afProgram, afArrays)
            .uniformBlock("Scene", sceneUB)
      ;

      const draw = () => {
        const now = performance.now() / 1000;
        app.clear();
        drawAF
          // todo: only update resolution on resize
          .uniform("iResolution", resolution)
          .uniform("iTime", now)
          .uniform("track1[0]", track1)
          .uniform("track2[0]", track2)
          .draw();
        requestAnimationFrame(draw);
      }
      requestAnimationFrame(draw);

      globalThis.shaderSetArray = (options, idx, val) => {
        const {name, mod} = options;
        let track
        if (name == "track1") {
          track = track1
        } else if (name == "track2") {
          track = track2
        } else {
          console.error("Unknown array", name)
        }
        const pos = idx % track.length
        // console.log("Setting", name, idx, pos, val, track)
        const value = mod == "incr" ? track[pos] + val : val
        track[pos] = value
      }
    });
}, 100)

const scale = (normalized, min, max) => normalized * (max - min) + min;
const getValue = (e) => {
  let { value } = e;
  if (typeof e.value !== 'object') {
    value = { value };
  }
  let { note, n, freq, s } = value;
  if (freq) {
    return freqToMidi(freq);
  }
  note = note ?? n;
  if (typeof note === 'string') {
    try {
      // TODO: n(run(32)).scale("D:minor") fails when trying to query negative time..
      return noteToMidi(note);
    } catch (err) {
      // console.warn(`error converting note to midi: ${err}`); // this spams to crazy
      return 0;
    }
  }
  if (typeof note === 'number') {
    return note;
  }
  if (s) {
    return '_' + s;
  }
  return value;
};

Pattern.prototype.shader = function (options = {}) {
  options.ctx.canvas.style.display = "none"
  // The set of notes that are playing
  let playing = new Set();
  let pitchCount = 0;
  this.draw(
    (haps, time) => {
      // The set of notes currently playing
      const isOn = new Set()
      haps.forEach(event => {
        const isActive = event.whole.begin <= time && event.endClipped > time;
        if (isActive)
          isOn.add(getValue(event))
      })
      isOn.symmetricDifference(playing).forEach(note => {
        const value = isOn.has(note) ? 1.0 : 0.0;
        if (typeof note == "string") {
          note = pitchCount++;
        }
        shaderSetArray(options, note, value)
      })
      playing = isOn;
    },
    { id: options.id }
  )
  return this
  /* .onTrigger((time_deprecate, hap, currentTime, cps, targetTime) => {
    console.log("GOOO", cps, hap)
  }) */
};
