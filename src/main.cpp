// Terrain editor demo -- sculpt + paint + grass, Awaii-diorama style.
// Ground color comes from two tiles harvested out of the pack's own
// paintings (assets/grass.bmp, assets/dirt.bmp): the brush paints a splat
// mask and the shader blends the tiles with a noise-broken edge, so dirt
// paths get the ragged hand-painted border automatically. Instanced grass
// blades read the same mask (no blades on dirt) and the heightmap, so
// everything follows sculpting and painting live.
//
// Controls:
//   RMB drag    look around          WASD + Q/E   fly
//   LMB         apply current brush  wheel        brush radius
//   1           sculpt raise (hold LSHIFT to lower)
//   2           smooth
//   3           paint dirt
//   4           paint grass
//   F5 / F9     save / load map (terrain\map.bin)
//   G           toggle grass
//   Esc         quit
//   --shot <file.bmp>: stamp a demo path + hill, screenshot, exit

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include "gl_loader.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_opengl3.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

// world extent: the ground is 2*TER_HALF on a side
static const float TER_HALF = 24.0f;
static const int   GRID_N   = 256;   // terrain quads per side
static const int   HN       = 257;   // height samples per side
static const int   MASK_N   = 512;   // splat mask resolution
static const int   GRASS_N  = 320;   // grass instances per side

// ---------------------------------------------------------------- shaders

static const char* TER_VS = R"(#version 330 core
layout(location = 0) in vec2 aPos;         // xz
uniform mat4 uMvp;
uniform sampler2D uHeight;
uniform float uHalf;
out vec3 vWorld;
out vec3 vNormal;

float heightAt(vec2 xz) {
    vec2 uv = (xz + vec2(uHalf)) / (2.0 * uHalf);
    return texture(uHeight, uv).r;
}
void main() {
    float h = heightAt(aPos);
    float eps = 2.0 * uHalf / 256.0;
    float hx = heightAt(aPos + vec2(eps, 0.0)) - heightAt(aPos - vec2(eps, 0.0));
    float hz = heightAt(aPos + vec2(0.0, eps)) - heightAt(aPos - vec2(0.0, eps));
    vNormal = normalize(vec3(-hx, 2.0 * eps, -hz));
    vWorld = vec3(aPos.x, h, aPos.y);
    gl_Position = uMvp * vec4(vWorld, 1.0);
}
)";

static const char* TER_FS = R"(#version 330 core
in vec3 vWorld;
in vec3 vNormal;
out vec4 fragColor;
uniform sampler2D uMask;
uniform sampler2D uMask2;
uniform sampler2D uGrassTex;
uniform sampler2D uDirtTex;
uniform sampler2D uDirt2Tex;
uniform sampler2D uCliffTex;
uniform float uHalf;
uniform vec4  uBrush;    // xz, radius, active
uniform vec3  uBrushCol;
uniform sampler2D uBrushStamp;
uniform float uBrushFalloff;
uniform float uEdgeBreak;   // 0 = crisp edges, 1 = wide ragged breakup
uniform sampler2D uShadowMap;
uniform mat4 uLightMvp;
uniform int uShadowsOn;

float shadow_factor(vec3 world) {
    if (uShadowsOn == 0)
        return 1.0;
    vec4 lc = uLightMvp * vec4(world, 1.0);
    vec3 p = lc.xyz / lc.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0)
        return 1.0;
    float s = 0.0;
    vec2 ts = 1.0 / vec2(textureSize(uShadowMap, 0));
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            float d = texture(uShadowMap, p.xy + vec2(dx, dy) * ts).r;
            s += (p.z - 0.0022 > d) ? 0.0 : 1.0;
        }
    return s / 9.0;
}

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
    for (int i = 0; i < 4; i++) { v += a * vnoise(p); p = rot * p * 2.03; a *= 0.5; }
    return v;
}

void main() {
    vec2 maskUv = (vWorld.xz + vec2(uHalf)) / (2.0 * uHalf);
    float m = texture(uMask, maskUv).r;

    vec3 grass = texture(uGrassTex, vWorld.xz * 0.16).rgb;
    // large-scale colormap variation so the field isn't one flat green
    float tintN = fbm(vWorld.xz * 0.10);
    grass *= mix(vec3(0.92, 0.96, 0.80), vec3(1.06, 1.05, 0.95), tintN);

    // two scales + a swapped-axis second tap so the tile repeat never lines up
    vec3 dirtA = texture(uDirtTex, vWorld.xz * 0.20).rgb;
    vec3 dirtB = texture(uDirtTex, vWorld.zx * 0.083 + 0.37).rgb;
    vec3 dirt = mix(dirtA, dirtB, 0.45);
    // macro tone variation: sunned cream to packed brown patches
    dirt *= mix(vec3(0.86, 0.80, 0.68), vec3(1.08, 1.03, 0.92),
                fbm(vWorld.xz * 0.21));

    // soft dirt layer: calmer clearing material, gentler variation
    float m2 = texture(uMask2, maskUv).r;
    vec3 soft = texture(uDirt2Tex, vWorld.xz * 0.15).rgb;
    soft *= mix(vec3(0.96, 0.94, 0.88), vec3(1.05, 1.03, 0.98),
                fbm(vWorld.xz * 0.17 + 5.1));

    // noise-broken blend edges: painted borders go ragged by themselves;
    // uEdgeBreak widens the band and cranks the noise so even small
    // brushes keep the organic crumble
    float n = fbm(vWorld.xz * 1.1) - 0.5;
    float amp = uEdgeBreak * 0.8;
    float band = 0.03 + uEdgeBreak * 0.25;
    float edge = smoothstep(0.5 - band, 0.5 + band, m + n * amp);
    float edge2 = smoothstep(0.5 - band, 0.5 + band, m2 + n * amp);
    vec3 col = mix(grass, dirt, edge);
    col = mix(col, soft, edge2);
    // soft shadowed band where grass meets dirt grounds the path
    // (path dirt gets the full band, soft dirt only a whisper)
    col *= (1.0 - 0.13 * edge * (1.0 - edge) * 4.0) *
           (1.0 - 0.05 * edge2 * (1.0 - edge2) * 4.0);

    // slope-based cliff: triplanar so the rock drapes down sculpted walls
    // instead of smearing (two vertical projections blended by the normal)
    vec3 nrm = normalize(vNormal);
    float slope = 1.0 - nrm.y;
    vec3 cliffX = texture(uCliffTex, vWorld.zy * 0.14).rgb;
    vec3 cliffZ = texture(uCliffTex, vWorld.xy * 0.14).rgb;
    float wx = abs(nrm.x) / max(abs(nrm.x) + abs(nrm.z), 1e-4);
    vec3 cliff = mix(cliffZ, cliffX, wx);
    // ragged transition band, same trick as the dirt edge
    float cliffM = smoothstep(0.22, 0.42, slope + (fbm(vWorld.xz * 1.7) - 0.5) * 0.18);
    col = mix(col, cliff, cliffM);

    vec3 L = normalize(vec3(0.35, 0.8, -0.45));
    float diff = max(dot(normalize(vNormal), L), 0.0);
    diff *= shadow_factor(vWorld);
    col *= 0.72 + 0.38 * diff;

    // brush preview: project the stamp (with the falloff curve) inside
    // the cursor circle, plus the rim ring
    if (uBrush.w > 0.5) {
        vec2 buv = (vWorld.xz - uBrush.xy) / uBrush.z * 0.5 + 0.5;
        if (buv.x > 0.0 && buv.x < 1.0 && buv.y > 0.0 && buv.y < 1.0) {
            float a = texture(uBrushStamp, buv).r;
            a = a <= 0.0 ? 0.0 : pow(a, uBrushFalloff);
            col = mix(col, uBrushCol, a * 0.45);
        }
        float d = length(vWorld.xz - uBrush.xy);
        float ring = smoothstep(uBrush.z * 0.92, uBrush.z * 0.97, d) *
                     (1.0 - smoothstep(uBrush.z * 1.03, uBrush.z * 1.08, d));
        col = mix(col, uBrushCol, ring * 0.85);
    }
    fragColor = vec4(col, 1.0);
}
)";

static const char* GRASS_VS = R"(#version 330 core
layout(location = 0) in vec3 aBlade;   // x: -1..1 across, y: 0..1 up, z: plane 0/1
layout(location = 1) in vec4 aInst;    // xz, rot, seed
uniform mat4 uMvp;
uniform sampler2D uHeight;
uniform sampler2D uMask;
uniform sampler2D uMask2;
uniform sampler2D uKill;
uniform float uHalf;
uniform float uTime;
uniform float uDensity;
out float vV;
out vec2  vWorldXz;
out float vSeed;

void main() {
    vec2 xz = aInst.xy;
    vec2 uv = (xz + vec2(uHalf)) / (2.0 * uHalf);
    float ground = texture(uHeight, uv).r;
    float m = max(texture(uMask, uv).r, texture(uMask2, uv).r);

    // no grass on dirt; per-blade threshold jitter makes a soft ragged edge
    float show = step(m, 0.30 + fract(aInst.w * 7.31) * 0.12);
    // density mask (Remove Grass = 0, painted density in between) times
    // the global blade-density setting; each blade rolls its own die
    float dens = (1.0 - texture(uKill, uv).r) * uDensity;
    show *= step(fract(aInst.w * 9.77), dens);

    // no grass on cliffs: estimate slope from the heightmap
    float eps = 2.0 * uHalf / 256.0;
    float hx = texture(uHeight, uv + vec2(eps, 0.0) / (2.0 * uHalf)).r -
               texture(uHeight, uv - vec2(eps, 0.0) / (2.0 * uHalf)).r;
    float hz = texture(uHeight, uv + vec2(0.0, eps) / (2.0 * uHalf)).r -
               texture(uHeight, uv - vec2(0.0, eps) / (2.0 * uHalf)).r;
    float ny = 2.0 * eps / length(vec3(hx, 2.0 * eps, hz));
    show *= step(1.0 - ny, 0.20 + fract(aInst.w * 4.77) * 0.08);

    float c = cos(aInst.z), s = sin(aInst.z);
    float planeRot = aInst.z + aBlade.z * 1.5707963;
    vec2 dir = vec2(cos(planeRot), sin(planeRot));

    // scale EVERYTHING by show: a hidden blade must not leave a flat
    // ground-level splat behind (zero height but nonzero width)
    float hgt = (0.28 + fract(aInst.w * 3.17) * 0.30) * show;
    float wid = (0.035 + fract(aInst.w * 5.71) * 0.02) * show;

    vec3 p = vec3(xz.x, ground, xz.y);
    p.xz += dir * aBlade.x * wid * (1.0 - aBlade.y * 0.7);
    p.y  += aBlade.y * hgt;

    // wind: big rolling gust + per-blade flutter, bending the tip
    float phase = dot(xz, vec2(0.8, 0.6)) * 0.7 + uTime * 1.8;
    float gust  = sin(phase) * 0.5 + sin(phase * 0.37 + 1.7) * 0.5;
    float sway  = (gust * 0.10 + sin(uTime * 3.1 + aInst.w * 6.28) * 0.03)
                  * aBlade.y * aBlade.y;
    p.xz += vec2(0.85, 0.53) * sway;

    vV = aBlade.y;
    vWorldXz = xz;
    vSeed = fract(aInst.w * 11.13);
    gl_Position = uMvp * vec4(p, 1.0);
}
)";

static const char* GRASS_FS = R"(#version 330 core
in float vV;
in vec2  vWorldXz;
in float vSeed;
out vec4 fragColor;
uniform sampler2D uGrassTex;

void main() {
    // original blade gradient: dark root, bright tip -- but sampled from
    // the blurred ground (high mip) so single flower pixels still can't
    // turn blades into confetti
    vec3 groundCol = textureLod(uGrassTex, vWorldXz * 0.16, 4.5).rgb;
    vec3 root = groundCol * 0.55;
    vec3 tip  = groundCol * (1.15 + vSeed * 0.15);
    vec3 col = mix(root, tip, vV * vV);
    fragColor = vec4(col, 1.0);
}
)";

static const char* PROP_VS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNorm;
layout(location = 2) in vec2 aUv;
uniform mat4 uMvp;
uniform mat4 uModel;
out vec3 vNorm;
out vec2 vUv;
out float vLocalY;
out vec3 vWorld;
void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vNorm = mat3(uModel) * aNorm;
    vUv = aUv;
    vLocalY = aPos.y;
    vWorld = world.xyz;
    gl_Position = uMvp * world;
}
)";

static const char* PROP_FS = R"(#version 330 core
in vec3 vNorm;
in vec2 vUv;
in float vLocalY;
in vec3 vWorld;
out vec4 fragColor;
uniform sampler2D uTex;
uniform vec3 uKd;      // top/main color (the pack's material gradient)
uniform vec3 uKa;      // bottom color
uniform float uBoundH; // mesh height for the gradient
uniform int uHasTex;
uniform int uGrayMask; // texture is a grayscale mask, not albedo
uniform sampler2D uShadowMap;
uniform mat4 uLightMvp;
uniform int uShadowsOn;

float shadow_factor(vec3 world) {
    if (uShadowsOn == 0)
        return 1.0;
    vec4 lc = uLightMvp * vec4(world, 1.0);
    vec3 p = lc.xyz / lc.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0)
        return 1.0;
    float s = 0.0;
    vec2 ts = 1.0 / vec2(textureSize(uShadowMap, 0));
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            float d = texture(uShadowMap, p.xy + vec2(dx, dy) * ts).r;
            s += (p.z - 0.003 > d) ? 0.0 : 1.0;
        }
    return s / 9.0;
}

void main() {
    vec4 t = uHasTex == 1 ? texture(uTex, vUv) : vec4(1.0);
    // mask textures (all-opaque alpha) cut out by luminance instead
    if (uHasTex == 1) {
        if (uGrayMask == 1) {
            if (t.r < 0.35 && t.a > 0.99)
                discard;
        }
        if (t.a < 0.5)
            discard;
    }
    // the pack colors meshes with a bottom->top material gradient; mask
    // textures only modulate detail
    vec3 grad = mix(uKa, uKd, clamp(vLocalY / max(uBoundH, 0.001), 0.0, 1.0));
    vec3 col = uGrayMask == 1 ? grad * (0.55 + 0.9 * t.r)
                              : t.rgb * uKd;
    vec3 L = normalize(vec3(0.35, 0.8, -0.45));
    vec3 n = normalize(vNorm);
    float sf = shadow_factor(vWorld);
    if (uGrayMask == 1) {
        // foliage: soft wrap lighting and mostly shadow-immune, so
        // canopies read as toon masses instead of speckled black --
        // they still CAST onto the ground
        float wrap = clamp(dot(n, L) * 0.4 + 0.6, 0.0, 1.0);
        col *= (0.55 + 0.5 * wrap) * mix(0.82, 1.0, sf);
    } else {
        float diff = max(dot(n, L), 0.0);
        // double-sided: faces flip freely
        diff = max(diff, max(dot(-n, L), 0.0) * 0.8);
        diff *= mix(1.0, sf, 0.85);
        col *= 0.68 + 0.42 * diff;
    }
    fragColor = vec4(col, 1.0);
}
)";

// depth-only passes for the shadow map
static const char* DEPTH_FS = R"(#version 330 core
void main() {}
)";

static const char* DEPTH_PROP_FS = R"(#version 330 core
in vec2 vUv;
uniform sampler2D uTex;
uniform int uHasTex;
uniform int uGrayMask;
void main() {
    if (uHasTex == 1) {
        vec4 t = texture(uTex, vUv);
        if (uGrayMask == 1) {
            if (t.r < 0.35 && t.a > 0.99)
                discard;
        }
        if (t.a < 0.5)
            discard;
    }
}
)";

// procedural sky (flat-white cloud variant, same as sky/water demos)
static const char* SKY_VS = R"(#version 330 core
const vec2 verts[3] = vec2[3](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
out vec2 vNdc;
void main() { vNdc = verts[gl_VertexID]; gl_Position = vec4(verts[gl_VertexID], 1.0, 1.0); }
)";

static const char* SKY_FS = R"(#version 330 core
in vec2 vNdc;
out vec4 fragColor;
uniform mat4  uCamRot;
uniform float uAspect;
uniform float uTanHalfFov;
uniform float uTime;

const vec3  kZenith   = vec3(0.22, 0.42, 0.86);
const vec3  kHorizon  = vec3(0.66, 0.80, 0.95);
const vec3  kSunDir   = vec3(0.45, 0.35, -0.60);
const float kCoverage = 0.52;

float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1,0)), u.x), mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = rot * p * 2.03; a *= 0.5; }
    return v;
}
void main() {
    vec3 rayCam = vec3(vNdc.x * uTanHalfFov * uAspect, vNdc.y * uTanHalfFov, -1.0);
    vec3 dir = normalize(mat3(uCamRot) * rayCam);
    vec3 sun = normalize(kSunDir);
    float h = clamp(dir.y, 0.0, 1.0);
    vec3 col = mix(kHorizon, kZenith, pow(h, 0.55));
    float sd = dot(dir, sun);
    col += vec3(1.0, 0.9, 0.7) * 0.25 * pow(clamp(sd, 0.0, 1.0), 64.0);
    if (dir.y > 0.01) {
        vec2 uv = dir.xz / dir.y * 1.6;
        uv += normalize(vec2(1.0, 0.35)) * 0.02 * uTime;
        vec2 warp = vec2(fbm(uv * 0.5 + 13.7), fbm(uv * 0.5 - 7.2)) - 0.5;
        float f = fbm(uv + warp * 1.4);
        float mask = smoothstep(kCoverage, kCoverage + 0.10, f);
        mask *= smoothstep(0.02, 0.14, dir.y);
        col = mix(col, vec3(1.0), mask);
    }
    fragColor = vec4(col, 1.0);
}
)";

// ------------------------------------------------------------------ gl utils

static GLuint compile(GLenum type, const char* src)
{
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        SDL_Log("shader compile failed:\n%s", log);
    }
    return sh;
}

static GLuint make_program(const char* vs, const char* fs)
{
    GLuint p = glCreateProgram();
    GLuint v = compile(GL_VERTEX_SHADER, vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        SDL_Log("program link failed:\n%s", log);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

static GLuint load_bmp_texture(const char* path)
{
    SDL_Surface* raw = SDL_LoadBMP(path);
    if (!raw) {
        SDL_Log("missing texture %s: %s", path, SDL_GetError());
        return 0;
    }
    SDL_Surface* s = SDL_ConvertSurface(raw, SDL_PIXELFORMAT_RGB24);
    SDL_DestroySurface(raw);
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, s->w, s->h, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, s->pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    SDL_DestroySurface(s);
    return tex;
}

// column-major mat4, minimal (same helpers as the water demo)
struct Mat4 { float m[16]; };

static Mat4 mat4_mul(const Mat4& a, const Mat4& b)
{
    Mat4 r{};
    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++)
            for (int k = 0; k < 4; k++)
                r.m[c * 4 + row] += a.m[k * 4 + row] * b.m[c * 4 + k];
    return r;
}

static Mat4 mat4_perspective(float fovY, float aspect, float zn, float zf)
{
    float t = 1.0f / tanf(fovY * 0.5f);
    Mat4 r{};
    r.m[0] = t / aspect;
    r.m[5] = t;
    r.m[10] = (zf + zn) / (zn - zf);
    r.m[11] = -1.0f;
    r.m[14] = 2.0f * zf * zn / (zn - zf);
    return r;
}

static Mat4 cam_rotation(float yaw, float pitch)
{
    float cy = cosf(yaw), sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    Mat4 r{};
    r.m[0] = cy;      r.m[1] = 0;   r.m[2] = -sy;
    r.m[4] = sy * sp; r.m[5] = cp;  r.m[6] = cy * sp;
    r.m[8] = sy * cp; r.m[9] = -sp; r.m[10] = cy * cp;
    r.m[15] = 1.0f;
    return r;
}

static Mat4 view_matrix(const Mat4& camRot, float px, float py, float pz)
{
    Mat4 v{};
    for (int c = 0; c < 3; c++)
        for (int row = 0; row < 3; row++)
            v.m[c * 4 + row] = camRot.m[row * 4 + c];
    v.m[12] = -(v.m[0] * px + v.m[4] * py + v.m[8] * pz);
    v.m[13] = -(v.m[1] * px + v.m[5] * py + v.m[9] * pz);
    v.m[14] = -(v.m[2] * px + v.m[6] * py + v.m[10] * pz);
    v.m[15] = 1.0f;
    return v;
}

static Mat4 mat4_ortho(float l, float r, float b, float t, float zn, float zf)
{
    Mat4 m{};
    m.m[0] = 2.0f / (r - l);
    m.m[5] = 2.0f / (t - b);
    m.m[10] = -2.0f / (zf - zn);
    m.m[12] = -(r + l) / (r - l);
    m.m[13] = -(t + b) / (t - b);
    m.m[14] = -(zf + zn) / (zf - zn);
    m.m[15] = 1.0f;
    return m;
}

static Mat4 mat4_lookat(const float eye[3], const float at[3])
{
    float f[3] = { at[0] - eye[0], at[1] - eye[1], at[2] - eye[2] };
    float fl = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    f[0] /= fl; f[1] /= fl; f[2] /= fl;
    float up[3] = { 0, 1, 0 };
    float s[3] = { f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2],
                   f[0] * up[1] - f[1] * up[0] };
    float sl = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    s[0] /= sl; s[1] /= sl; s[2] /= sl;
    float u[3] = { s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2],
                   s[0] * f[1] - s[1] * f[0] };
    Mat4 m{};
    m.m[0] = s[0]; m.m[4] = s[1]; m.m[8] = s[2];
    m.m[1] = u[0]; m.m[5] = u[1]; m.m[9] = u[2];
    m.m[2] = -f[0]; m.m[6] = -f[1]; m.m[10] = -f[2];
    m.m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    m.m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    m.m[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
    m.m[15] = 1.0f;
    return m;
}

static Mat4 model_trs(float x, float y, float z, float yaw, float s)
{
    float c = cosf(yaw), sn = sinf(yaw);
    Mat4 r{};
    r.m[0] = c * s;   r.m[2] = -sn * s;
    r.m[5] = s;
    r.m[8] = sn * s;  r.m[10] = c * s;
    r.m[12] = x; r.m[13] = y; r.m[14] = z;
    r.m[15] = 1.0f;
    return r;
}

static void save_screenshot(SDL_Window* win, const char* path)
{
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(win, &w, &h);
    SDL_Surface* s = SDL_CreateSurface(w, h, SDL_PIXELFORMAT_RGB24);
    if (!s)
        return;
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, s->pixels);
    SDL_FlipSurface(s, SDL_FLIP_VERTICAL);
    SDL_SaveBMP(s, path);
    SDL_DestroySurface(s);
    SDL_Log("saved %s", path);
}

// ------------------------------------------------------------------ editor state

static std::vector<float>   gHeights(HN* HN, 0.0f);
static std::vector<Uint8>   gMask(MASK_N* MASK_N, 0);   // path dirt layer
static std::vector<Uint8>   gMask2(MASK_N* MASK_N, 0);  // soft dirt layer
static std::vector<Uint8>   gKill(MASK_N* MASK_N, 255); // 255 = no blades;
                                                        // maps start bare,
                                                        // blades are painted
static GLuint gHeightTex = 0, gMaskTex = 0, gMask2Tex = 0, gKillTex = 0;
static bool gHeightsDirty = true, gMaskDirty = true, gMask2Dirty = true,
            gKillDirty = true;

// placed prop instances (mesh index into gPropMeshes)
struct PropInst {
    int mesh;
    float x, y, z, yaw, scale;
};
static std::vector<PropInst> gProps;

static float height_at(float x, float z)
{
    float u = (x + TER_HALF) / (2.0f * TER_HALF) * (HN - 1);
    float v = (z + TER_HALF) / (2.0f * TER_HALF) * (HN - 1);
    int i = SDL_clamp((int)u, 0, HN - 2);
    int j = SDL_clamp((int)v, 0, HN - 2);
    float fu = SDL_clamp(u - i, 0.0f, 1.0f);
    float fv = SDL_clamp(v - j, 0.0f, 1.0f);
    float a = gHeights[j * HN + i], b = gHeights[j * HN + i + 1];
    float c = gHeights[(j + 1) * HN + i], d = gHeights[(j + 1) * HN + i + 1];
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv;
}

// march a ray against the heightfield; true on hit inside the terrain bounds
static bool ray_terrain(const float ro[3], const float rd[3], float out[3])
{
    float t = 0.0f;
    float prevT = 0.0f;
    bool wasAbove = true;
    for (int i = 0; i < 2000; i++) {
        t += 0.12f;
        float x = ro[0] + rd[0] * t;
        float y = ro[1] + rd[1] * t;
        float z = ro[2] + rd[2] * t;
        if (t > 250.0f)
            return false;
        bool inside = fabsf(x) <= TER_HALF && fabsf(z) <= TER_HALF;
        bool above = !inside || y > height_at(x, z);
        if (!above && wasAbove && i > 0) {
            // bisect between prevT and t
            float lo = prevT, hi = t;
            for (int k = 0; k < 10; k++) {
                float mid = 0.5f * (lo + hi);
                float mx = ro[0] + rd[0] * mid;
                float my = ro[1] + rd[1] * mid;
                float mz = ro[2] + rd[2] * mid;
                if (my > height_at(mx, mz)) lo = mid; else hi = mid;
            }
            out[0] = ro[0] + rd[0] * hi;
            out[1] = ro[1] + rd[1] * hi;
            out[2] = ro[2] + rd[2] * hi;
            return fabsf(out[0]) <= TER_HALF && fabsf(out[2]) <= TER_HALF;
        }
        wasAbove = above;
        prevT = t;
    }
    return false;
}

enum BrushMode { BRUSH_RAISE, BRUSH_SMOOTH, BRUSH_FLATTEN,
                 BRUSH_SHARPEN, BRUSH_TERRACE,
                 BRUSH_DIRT, BRUSH_DIRT2, BRUSH_ERASEDIRT,
                 BRUSH_GRASS, BRUSH_KILLGRASS };

static const float kBrushColors[10][3] = {
    { 1.0f, 0.85f, 0.3f },   // raise: yellow
    { 0.4f, 0.8f, 1.0f },    // smooth: blue
    { 0.9f, 0.5f, 0.9f },    // flatten: purple
    { 1.0f, 0.55f, 0.2f },   // sharpen edges: orange
    { 0.3f, 0.9f, 0.8f },    // terrace: teal
    { 0.72f, 0.5f, 0.28f },  // path dirt: brown
    { 0.85f, 0.75f, 0.5f },  // soft dirt: sand
    { 0.35f, 0.8f, 0.35f },  // grass ground (erases dirt): deep green
    { 0.5f, 1.0f, 0.4f },    // grass blades: bright green
    { 0.9f, 0.35f, 0.3f },   // remove grass: red
};
static const char* kBrushNames[10] = { "Sculpt", "Smooth", "Flatten",
                                       "Sharpen", "Terrace",
                                       "Path Dirt", "Soft Dirt", "Grass Ground",
                                       "Grass Blades", "Remove Grass" };

// terrace step height (world units), set from the panel
static float gTerraceStep = 2.0f;

// stroke-level undo/redo: whole-state snapshots (~1 MB each, capped)
struct Snapshot {
    std::vector<float> h;
    std::vector<Uint8> m, m2, k;
    std::vector<PropInst> p;
};
static std::vector<Snapshot> gUndoStack, gRedoStack;

static void mark_all_dirty()
{
    gHeightsDirty = gMaskDirty = gMask2Dirty = gKillDirty = true;
}

static void push_undo()
{
    gUndoStack.push_back({ gHeights, gMask, gMask2, gKill, gProps });
    if (gUndoStack.size() > 32)
        gUndoStack.erase(gUndoStack.begin());
    gRedoStack.clear();
}

static void do_undo()
{
    if (gUndoStack.empty())
        return;
    gRedoStack.push_back({ gHeights, gMask, gMask2, gKill, gProps });
    const Snapshot& s = gUndoStack.back();
    gHeights = s.h; gMask = s.m; gMask2 = s.m2; gKill = s.k; gProps = s.p;
    gUndoStack.pop_back();
    mark_all_dirty();
}

static void do_redo()
{
    if (gRedoStack.empty())
        return;
    gUndoStack.push_back({ gHeights, gMask, gMask2, gKill, gProps });
    const Snapshot& s = gRedoStack.back();
    gHeights = s.h; gMask = s.m; gMask2 = s.m2; gKill = s.k; gProps = s.p;
    gRedoStack.pop_back();
    mark_all_dirty();
}

// flatten pulls terrain toward the height captured when the stroke began
static float gFlattenTarget = 0.0f;

// Unity-style alpha-stamp brushes: each brush is a little grayscale image;
// painting stamps its opacity. Generated procedurally at startup.
struct BrushStamp {
    std::string name;
    std::vector<float> alpha;   // STAMP_N x STAMP_N, 0..1
    unsigned tex = 0;           // GL texture for the gallery thumbnail
};
static const int STAMP_N = 64;
static std::vector<BrushStamp> gStamps;
static int gStamp = 1;   // default: soft round
static float gBrushFalloff = 1.0f;   // power curve: <1 softer, >1 harder

// grass brush settings: painted density and placement pattern, baked
// into the density mask so different areas keep different patterns
static float gGrassDensity = 1.0f;
static int gGrassPattern = 0;
static const char* kGrassPatterns[] = { "Uniform", "Clumps", "Speckle" };

static float cpu_vnoise(float x, float y)
{
    auto hash = [](int ix, int iy) {
        unsigned h = (unsigned)(ix * 374761393 + iy * 668265263);
        h = (h ^ (h >> 13)) * 1274126177u;
        return ((h ^ (h >> 16)) & 0xffffff) / 16777215.0f;
    };
    int ix = (int)floorf(x), iy = (int)floorf(y);
    float fx = x - ix, fy = y - iy;
    fx = fx * fx * (3.0f - 2.0f * fx);
    fy = fy * fy * (3.0f - 2.0f * fy);
    float a = hash(ix, iy), b = hash(ix + 1, iy);
    float c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
    return a + (b - a) * fx + (c - a) * fy * (1 - fx) + (d - b) * fx * fy;
}

static float grass_pattern(float x, float z)
{
    switch (gGrassPattern) {
    case 1: {   // clumps: large soft patches with bare gaps
        float n = cpu_vnoise(x * 0.45f, z * 0.45f);
        return SDL_clamp((n - 0.32f) / 0.30f, 0.0f, 1.0f);
    }
    case 2: {   // speckle: small tufts scattered on mostly-bare ground
        float n = cpu_vnoise(x * 2.7f, z * 2.7f);
        return n * n * n;
    }
    default:
        return 1.0f;
    }
}

// sample the active stamp's alpha at a brush-space offset (bilinear)
static float brush_weight(float dx, float dz, float radius, float, float)
{
    float u = (dx / radius) * 0.5f + 0.5f;
    float v = (dz / radius) * 0.5f + 0.5f;
    if (u < 0.0f || u >= 1.0f || v < 0.0f || v >= 1.0f)
        return 0.0f;
    const std::vector<float>& a = gStamps[gStamp].alpha;
    float fx = u * (STAMP_N - 1), fy = v * (STAMP_N - 1);
    int ix = (int)fx, iy = (int)fy;
    float tx = fx - ix, ty = fy - iy;
    int ix1 = SDL_min(ix + 1, STAMP_N - 1);
    int iy1 = SDL_min(iy + 1, STAMP_N - 1);
    float top = a[iy * STAMP_N + ix] * (1 - tx) + a[iy * STAMP_N + ix1] * tx;
    float bot = a[iy1 * STAMP_N + ix] * (1 - tx) + a[iy1 * STAMP_N + ix1] * tx;
    float w = top * (1 - ty) + bot * ty;
    return w <= 0.0f ? 0.0f : powf(w, gBrushFalloff);
}

// build the default gallery: soft/hard rounds, splotches, hexagon, star
static void make_stamps()
{
    auto add = [](const char* name, float (*f)(float, float, int), int seed) {
        BrushStamp st;
        st.name = name;
        st.alpha.resize(STAMP_N * STAMP_N);
        std::vector<unsigned char> rgba(STAMP_N * STAMP_N * 4);
        for (int y = 0; y < STAMP_N; y++)
            for (int x = 0; x < STAMP_N; x++) {
                float u = (x + 0.5f) / STAMP_N * 2.0f - 1.0f;
                float v = (y + 0.5f) / STAMP_N * 2.0f - 1.0f;
                float a = SDL_clamp(f(u, v, seed), 0.0f, 1.0f);
                st.alpha[y * STAMP_N + x] = a;
                unsigned char c = (unsigned char)(a * 255.0f);
                int i = (y * STAMP_N + x) * 4;
                rgba[i] = rgba[i + 1] = rgba[i + 2] = c;
                rgba[i + 3] = 255;
            }
        glGenTextures(1, &st.tex);
        glBindTexture(GL_TEXTURE_2D, st.tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, STAMP_N, STAMP_N, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, rgba.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        gStamps.push_back(st);
    };
    auto soft = [](float u, float v, int) {
        float r = sqrtf(u * u + v * v);
        return expf(-4.5f * r * r) * SDL_clamp(1.0f - (r - 0.85f) / 0.15f, 0.0f, 1.0f);
    };
    auto hard = [](float u, float v, int) {
        float r = sqrtf(u * u + v * v);
        return SDL_clamp((0.95f - r) / 0.08f, 0.0f, 1.0f);
    };
    auto mid = [](float u, float v, int) {
        float r = sqrtf(u * u + v * v);
        float t = SDL_clamp(1.0f - r, 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };
    auto tight = [](float u, float v, int) {
        float r = sqrtf(u * u + v * v);
        return expf(-10.0f * r * r);
    };
    auto splotch = [](float u, float v, int seed) {
        float r = sqrtf(u * u + v * v);
        float fall = SDL_clamp(1.0f - r, 0.0f, 1.0f);
        float n = cpu_vnoise(u * 3.1f + seed * 17.7f, v * 3.1f - seed * 9.3f)
                * 0.65f
                + cpu_vnoise(u * 7.9f - seed * 5.1f, v * 7.9f + seed * 12.9f)
                * 0.35f;
        return SDL_clamp((n - 0.45f) * 4.0f, 0.0f, 1.0f) * fall * fall * 1.6f;
    };
    auto square = [](float u, float v, int) {
        float d = SDL_max(fabsf(u), fabsf(v));
        return SDL_clamp((0.90f - d) / 0.06f, 0.0f, 1.0f);
    };
    auto hexagon = [](float u, float v, int) {
        float ax = fabsf(u), ay = fabsf(v);
        float d = SDL_max(ax * 0.866025f + ay * 0.5f, ay);
        return SDL_clamp((0.85f - d) / 0.06f, 0.0f, 1.0f);
    };
    auto star = [](float u, float v, int) {
        float r = sqrtf(u * u + v * v);
        float th = atan2f(v, u);
        float edge = 0.35f + 0.5f * fabsf(cosf(2.5f * th));
        return SDL_clamp((edge - r) / 0.05f, 0.0f, 1.0f);
    };
    // Unity's remaining defaults: soft misty splotches, streaks, stars
    auto softsplotch = [](float u, float v, int seed) {
        float r = sqrtf(u * u + v * v);
        float fall = SDL_clamp(1.0f - r, 0.0f, 1.0f);
        float n = cpu_vnoise(u * 2.3f + seed * 23.1f, v * 2.3f - seed * 7.7f)
                * 0.6f
                + cpu_vnoise(u * 5.1f + seed * 3.9f, v * 5.1f - seed * 15.3f)
                * 0.4f;
        return SDL_clamp((n - 0.40f) * 2.2f, 0.0f, 1.0f) * fall * fall;
    };
    auto streaks = [](float u, float v, int) {
        float r = sqrtf(u * u + v * v);
        float fall = SDL_clamp(1.0f - r, 0.0f, 1.0f);
        float n = cpu_vnoise(u * 9.0f, v * 1.8f) * 0.7f
                + cpu_vnoise(u * 17.0f + 31.7f, v * 3.1f) * 0.3f;
        return SDL_clamp((n - 0.48f) * 5.0f, 0.0f, 1.0f) * fall * fall * 1.4f;
    };
    auto star6 = [](float u, float v, int) {
        float r = sqrtf(u * u + v * v);
        float th = atan2f(v, u);
        float edge = 0.35f + 0.5f * fabsf(cosf(3.0f * th));
        return SDL_clamp((edge - r) / 0.05f, 0.0f, 1.0f);
    };
    auto starring = [](float u, float v, int) {
        float r = sqrtf(u * u + v * v);
        float th = atan2f(v, u);
        float edge = 0.35f + 0.5f * fabsf(cosf(2.5f * th));
        float d = fabsf(r - edge);
        return SDL_clamp((0.05f - d) / 0.03f, 0.0f, 1.0f);
    };
    add("Soft", soft, 0);
    add("Round", mid, 0);
    add("Hard", hard, 0);
    add("Tight", tight, 0);
    add("Splotch 1", splotch, 1);
    add("Splotch 2", splotch, 2);
    add("Splotch 3", splotch, 3);
    add("Splotch 4", splotch, 4);
    add("Mist 1", softsplotch, 1);
    add("Mist 2", softsplotch, 2);
    add("Mist 3", softsplotch, 3);
    add("Streaks", streaks, 0);
    add("Square", square, 0);
    add("Hexagon", hexagon, 0);
    add("Star", star, 0);
    add("Star 6", star6, 0);
    add("Star Ring", starring, 0);
}

// drop any stamp that fills its corners (reads as a square brush)
static void prune_square_stamps()
{
    for (size_t i = 0; i < gStamps.size();) {
        const std::vector<float>& a = gStamps[i].alpha;
        float corners = a[0] + a[STAMP_N - 1] + a[(STAMP_N - 1) * STAMP_N] +
                        a[STAMP_N * STAMP_N - 1];
        if (corners > 1.2f) {
            glDeleteTextures(1, &gStamps[i].tex);
            gStamps.erase(gStamps.begin() + i);
        } else {
            i++;
        }
    }
    if (gStamp >= (int)gStamps.size())
        gStamp = 0;
}

// re-render every gallery thumbnail with the current falloff curve so
// the previews track the Falloff slider
static void update_stamp_thumbnails()
{
    std::vector<unsigned char> rgba(STAMP_N * STAMP_N * 4);
    for (BrushStamp& st : gStamps) {
        for (int i = 0; i < STAMP_N * STAMP_N; i++) {
            float a = st.alpha[i];
            a = a <= 0.0f ? 0.0f : powf(a, gBrushFalloff);
            unsigned char c = (unsigned char)(SDL_clamp(a, 0.0f, 1.0f) * 255.0f);
            rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = c;
            rgba[i * 4 + 3] = 255;
        }
        glBindTexture(GL_TEXTURE_2D, st.tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, STAMP_N, STAMP_N, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, rgba.data());
    }
}

// ------------------------------------------------------------------ props
// OBJ/MTL prop library (exported from the So Stylized Unity pack),
// lazy-loaded per mesh; instances live in the map.

struct PropMaterial {
    unsigned tex = 0;
    bool grayMask = false;
    float kd[3] = { 1, 1, 1 };   // top/main color
    float ka[3] = { 1, 1, 1 };   // bottom color (gradient)
};
struct PropSubmesh {
    int first = 0, count = 0, mat = 0;
};
struct PropMesh {
    std::string label;
    std::string category;
    std::string objPath;
    bool loaded = false, failed = false;
    unsigned vao = 0, vbo = 0;
    std::vector<PropSubmesh> subs;
    std::vector<PropMaterial> mats;
    float boundR = 1.0f, boundH = 1.0f;
};
struct PropCategory {
    std::string name;
    std::vector<int> meshes;
};
static std::vector<PropMesh> gPropMeshes;
static std::vector<PropCategory> gPropCats;
struct PropTex {
    unsigned tex = 0;
    bool gray = false;
};
static std::unordered_map<std::string, PropTex> gPropTexCache;
static std::string gPropsDir;

// RGBA BMP -> texture (32-bit BMPs keep their alpha for leaf cutouts);
// grayOut reports whether the image is a grayscale mask (r==g==b)
static unsigned load_bmp_texture_rgba(const char* path, bool* grayOut = nullptr)
{
    SDL_Surface* raw = SDL_LoadBMP(path);
    if (!raw)
        return 0;
    SDL_Surface* s = SDL_ConvertSurface(raw, SDL_PIXELFORMAT_RGBA32);
    SDL_DestroySurface(raw);
    if (!s)
        return 0;
    if (grayOut) {
        bool gray = true;
        const unsigned char* px = (const unsigned char*)s->pixels;
        for (int y = 0; y < s->h && gray; y += SDL_max(1, s->h / 32))
            for (int x = 0; x < s->w && gray; x += SDL_max(1, s->w / 32)) {
                const unsigned char* p = px + y * s->pitch + x * 4;
                if (abs(p[0] - p[1]) > 10 || abs(p[1] - p[2]) > 10)
                    gray = false;
            }
        *grayOut = gray;
    }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, s->w, s->h, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, s->pixels);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    SDL_DestroySurface(s);
    return tex;
}

static PropTex prop_texture(const std::string& file)
{
    auto it = gPropTexCache.find(file);
    if (it != gPropTexCache.end())
        return it->second;
    std::string full = gPropsDir + "/" + file;
    PropTex pt;
    pt.tex = load_bmp_texture_rgba(full.c_str(), &pt.gray);
    gPropTexCache[file] = pt;
    return pt;
}

static void load_prop_mtl(PropMesh& m, const std::string& path,
                          std::unordered_map<std::string, int>& matIndex)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char name[256];
        float r, g, b;
        if (sscanf(line, "newmtl %255s", name) == 1) {
            matIndex[name] = (int)m.mats.size();
            m.mats.push_back({});
        } else if (!m.mats.empty() &&
                   sscanf(line, "Kd %f %f %f", &r, &g, &b) == 3) {
            m.mats.back().kd[0] = r;
            m.mats.back().kd[1] = g;
            m.mats.back().kd[2] = b;
        } else if (!m.mats.empty() &&
                   sscanf(line, "Ka %f %f %f", &r, &g, &b) == 3) {
            m.mats.back().ka[0] = r;
            m.mats.back().ka[1] = g;
            m.mats.back().ka[2] = b;
        } else if (!m.mats.empty() &&
                   sscanf(line, "map_Kd %255s", name) == 1) {
            PropTex pt = prop_texture(name);
            m.mats.back().tex = pt.tex;
            m.mats.back().grayMask = pt.gray;
        }
    }
    fclose(f);
}

static bool load_prop(int idx)
{
    PropMesh& m = gPropMeshes[idx];
    if (m.loaded || m.failed)
        return m.loaded;
    FILE* f = fopen(m.objPath.c_str(), "rb");
    if (!f) {
        m.failed = true;
        return false;
    }
    std::vector<float> vs, ns, ts;
    std::vector<float> data;   // interleaved pos3 norm3 uv2
    std::unordered_map<std::string, int> matIndex;
    int curMat = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        float a, b, c;
        char name[256];
        if (sscanf(line, "v %f %f %f", &a, &b, &c) == 3) {
            vs.insert(vs.end(), { a, b, c });
        } else if (sscanf(line, "vn %f %f %f", &a, &b, &c) == 3) {
            ns.insert(ns.end(), { a, b, c });
        } else if (sscanf(line, "vt %f %f", &a, &b) == 2) {
            ts.insert(ts.end(), { a, b });
        } else if (sscanf(line, "mtllib %255s", name) == 1) {
            std::string dir = m.objPath.substr(0, m.objPath.find_last_of("/\\") + 1);
            load_prop_mtl(m, dir + name, matIndex);
        } else if (sscanf(line, "usemtl %255s", name) == 1) {
            auto it = matIndex.find(name);
            curMat = it != matIndex.end() ? it->second : 0;
            if (m.subs.empty() || m.subs.back().count > 0)
                m.subs.push_back({ (int)(data.size() / 8), 0, curMat });
            else
                m.subs.back().mat = curMat;
        } else if (line[0] == 'f' && line[1] == ' ') {
            int vi[3] = { 0, 0, 0 };
            if (sscanf(line, "f %d/%*d/%*d %d/%*d/%*d %d/%*d/%*d",
                       &vi[0], &vi[1], &vi[2]) == 3) {
                if (m.subs.empty())
                    m.subs.push_back({ 0, 0, 0 });
                for (int k = 0; k < 3; k++) {
                    int i = vi[k] - 1;
                    float px = vs[i * 3], py = vs[i * 3 + 1], pz = vs[i * 3 + 2];
                    data.insert(data.end(), { px, py, pz });
                    if ((size_t)(i * 3 + 2) < ns.size())
                        data.insert(data.end(),
                                    { ns[i * 3], ns[i * 3 + 1], ns[i * 3 + 2] });
                    else
                        data.insert(data.end(), { 0, 1, 0 });
                    if ((size_t)(i * 2 + 1) < ts.size())
                        data.insert(data.end(), { ts[i * 2], ts[i * 2 + 1] });
                    else
                        data.insert(data.end(), { 0, 0 });
                    float r = sqrtf(px * px + pz * pz);
                    m.boundR = SDL_max(m.boundR, r);
                    m.boundH = SDL_max(m.boundH, py);
                }
                m.subs.back().count += 3;
            }
        }
    }
    fclose(f);
    if (data.empty()) {
        m.failed = true;
        return false;
    }
    if (m.mats.empty())
        m.mats.push_back({});
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void*)(6 * sizeof(float)));
    glBindVertexArray(0);
    m.loaded = true;
    return true;
}

static void scan_props(const std::string& dir)
{
    namespace fs = std::filesystem;
    gPropsDir = dir;
    std::error_code ec;
    std::vector<std::string> cats;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.is_directory() && e.path().filename() != "textures")
            cats.push_back(e.path().filename().string());
    std::sort(cats.begin(), cats.end());
    for (const std::string& c : cats) {
        PropCategory cat;
        cat.name = c;
        std::vector<fs::path> objs;
        for (const auto& e : fs::directory_iterator(dir + "/" + c, ec))
            if (e.path().extension() == ".obj")
                objs.push_back(e.path());
        std::sort(objs.begin(), objs.end());
        for (const auto& p : objs) {
            PropMesh m;
            m.label = p.stem().string();
            m.category = c;
            m.objPath = p.string();
            cat.meshes.push_back((int)gPropMeshes.size());
            gPropMeshes.push_back(m);
        }
        if (!cat.meshes.empty())
            gPropCats.push_back(cat);
    }
    SDL_Log("props: %d meshes in %d categories",
            (int)gPropMeshes.size(), (int)gPropCats.size());
}

// grayscale BMPs in assets/brushes/ become stamps too (e.g. the real
// Unity built-in brushes exported from the editor)
static void load_stamp_files(const char* dir)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<fs::path> paths;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.path().extension() == ".bmp")
            paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    for (const auto& p : paths) {
        SDL_Surface* raw = SDL_LoadBMP(p.string().c_str());
        if (!raw)
            continue;
        SDL_Surface* s = SDL_ConvertSurface(raw, SDL_PIXELFORMAT_RGB24);
        SDL_DestroySurface(raw);
        if (!s)
            continue;
        BrushStamp st;
        st.name = p.stem().string();
        st.alpha.resize(STAMP_N * STAMP_N);
        std::vector<unsigned char> rgba(STAMP_N * STAMP_N * 4);
        const unsigned char* px = (const unsigned char*)s->pixels;
        for (int y = 0; y < STAMP_N; y++)
            for (int x = 0; x < STAMP_N; x++) {
                int sx = x * s->w / STAMP_N;
                int sy = y * s->h / STAMP_N;
                unsigned char c = px[sy * s->pitch + sx * 3];
                st.alpha[y * STAMP_N + x] = c / 255.0f;
                int i = (y * STAMP_N + x) * 4;
                rgba[i] = rgba[i + 1] = rgba[i + 2] = c;
                rgba[i + 3] = 255;
            }
        SDL_DestroySurface(s);
        glGenTextures(1, &st.tex);
        glBindTexture(GL_TEXTURE_2D, st.tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, STAMP_N, STAMP_N, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, rgba.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        gStamps.push_back(st);
    }
    SDL_Log("loaded %d brush stamps total", (int)gStamps.size());
}

static void apply_brush(BrushMode mode, float cx, float cz, float radius,
                        float dt, bool invert, float strength = 1.0f,
                        float paintTarget = 1.0f)
{
    if (mode <= BRUSH_TERRACE) {
        dt *= strength;
        float cell = 2.0f * TER_HALF / (HN - 1);
        int i0 = SDL_clamp((int)((cx - radius + TER_HALF) / cell), 0, HN - 1);
        int i1 = SDL_clamp((int)((cx + radius + TER_HALF) / cell) + 1, 0, HN - 1);
        int j0 = SDL_clamp((int)((cz - radius + TER_HALF) / cell), 0, HN - 1);
        int j1 = SDL_clamp((int)((cz + radius + TER_HALF) / cell) + 1, 0, HN - 1);
        std::vector<float> snap;
        if (mode == BRUSH_SMOOTH || mode == BRUSH_SHARPEN)
            snap = gHeights;
        for (int j = j0; j <= j1; j++)
            for (int i = i0; i <= i1; i++) {
                float x = -TER_HALF + i * cell;
                float z = -TER_HALF + j * cell;
                float w = brush_weight(x - cx, z - cz, radius, x, z);
                if (w <= 0.0f)
                    continue;
                float& h = gHeights[j * HN + i];
                if (mode == BRUSH_RAISE) {
                    h += (invert ? -1.0f : 1.0f) * 3.5f * w * dt;
                } else if (mode == BRUSH_FLATTEN) {
                    h += (gFlattenTarget - h) * SDL_min(1.0f, 12.0f * w * dt);
                } else if (mode == BRUSH_TERRACE) {
                    // snap toward stepped plateaus: flat tops, cliff sides
                    float target = roundf(h / gTerraceStep) * gTerraceStep;
                    h += (target - h) * SDL_min(1.0f, 10.0f * w * dt);
                } else if (mode == BRUSH_SHARPEN) {
                    // unsharp mask: push height away from the local average
                    // so slopes steepen into defined sides
                    float sum = 0.0f;
                    int n = 0;
                    for (int dj = -2; dj <= 2; dj++)
                        for (int di = -2; di <= 2; di++) {
                            int ii = SDL_clamp(i + di, 0, HN - 1);
                            int jj = SDL_clamp(j + dj, 0, HN - 1);
                            sum += snap[jj * HN + ii];
                            n++;
                        }
                    h += (h - sum / n) * SDL_min(0.5f, 6.0f * w * dt);
                } else {
                    float sum = 0.0f;
                    int n = 0;
                    for (int dj = -2; dj <= 2; dj++)
                        for (int di = -2; di <= 2; di++) {
                            int ii = SDL_clamp(i + di, 0, HN - 1);
                            int jj = SDL_clamp(j + dj, 0, HN - 1);
                            sum += snap[jj * HN + ii];
                            n++;
                        }
                    h += (sum / n - h) * SDL_min(1.0f, 30.0f * w * dt);
                }
            }
        gHeightsDirty = true;
    } else {
        float cell = 2.0f * TER_HALF / MASK_N;
        int i0 = SDL_clamp((int)((cx - radius + TER_HALF) / cell), 0, MASK_N - 1);
        int i1 = SDL_clamp((int)((cx + radius + TER_HALF) / cell) + 1, 0, MASK_N - 1);
        int j0 = SDL_clamp((int)((cz - radius + TER_HALF) / cell), 0, MASK_N - 1);
        int j1 = SDL_clamp((int)((cz + radius + TER_HALF) / cell) + 1, 0, MASK_N - 1);
        for (int j = j0; j <= j1; j++)
            for (int i = i0; i <= i1; i++) {
                float x = -TER_HALF + (i + 0.5f) * cell;
                float z = -TER_HALF + (j + 0.5f) * cell;
                float w = brush_weight(x - cx, z - cz, radius, x, z);
                if (w <= 0.0f)
                    continue;
                // Unity-style target strength: strokes climb fast toward a
                // ceiling and stop, so hover time doesn't overshoot
                float step = 255.0f * SDL_min(1.0f, 8.0f * w * dt);
                float cap = paintTarget * 255.0f;
                auto raiseTo = [step](Uint8& v, float ceilv) {
                    if (v >= ceilv)
                        return;
                    v = (Uint8)SDL_min(ceilv, v + step);
                };
                auto lowerTo = [step](Uint8& v, float floorv) {
                    if (v <= floorv)
                        return;
                    v = (Uint8)SDL_max(floorv, v - step);
                };
                if (mode == BRUSH_DIRT) {
                    raiseTo(gMask[j * MASK_N + i], cap);
                    lowerTo(gMask2[j * MASK_N + i], 0.0f);
                } else if (mode == BRUSH_DIRT2) {
                    raiseTo(gMask2[j * MASK_N + i], cap);
                    lowerTo(gMask[j * MASK_N + i], 0.0f);
                } else if (mode == BRUSH_ERASEDIRT) {
                    lowerTo(gMask[j * MASK_N + i], (1.0f - paintTarget) * 255.0f);
                    lowerTo(gMask2[j * MASK_N + i], (1.0f - paintTarget) * 255.0f);
                } else if (mode == BRUSH_KILLGRASS) {
                    raiseTo(gKill[j * MASK_N + i], cap);
                } else {   // grass blades only: density shaped by pattern,
                           // never touches the painted ground layers
                    float dens = gGrassDensity * grass_pattern(x, z);
                    float target = (1.0f - dens) * 255.0f;
                    Uint8& v = gKill[j * MASK_N + i];
                    if (v < target)
                        raiseTo(v, target);
                    else
                        lowerTo(v, target);
                }
            }
        gMaskDirty = gMask2Dirty = gKillDirty = true;
    }
}

static void upload_dirty()
{
    if (gHeightsDirty) {
        glBindTexture(GL_TEXTURE_2D, gHeightTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, HN, HN, 0, GL_RED, GL_FLOAT,
                     gHeights.data());
        gHeightsDirty = false;
    }
    if (gMaskDirty) {
        glBindTexture(GL_TEXTURE_2D, gMaskTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, MASK_N, MASK_N, 0, GL_RED,
                     GL_UNSIGNED_BYTE, gMask.data());
        gMaskDirty = false;
    }
    if (gMask2Dirty) {
        glBindTexture(GL_TEXTURE_2D, gMask2Tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, MASK_N, MASK_N, 0, GL_RED,
                     GL_UNSIGNED_BYTE, gMask2.data());
        gMask2Dirty = false;
    }
    if (gKillDirty) {
        glBindTexture(GL_TEXTURE_2D, gKillTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, MASK_N, MASK_N, 0, GL_RED,
                     GL_UNSIGNED_BYTE, gKill.data());
        gKillDirty = false;
    }
}

// bundled with the map so an exported file restores the whole session:
// view settings + camera (synced from main() around save/load calls)
static float gSetBlade = 0.8f, gSetEdge = 0.5f;
static float gSetCam[5] = { 0.0f, 12.0f, 30.0f, 0.0f, -0.42f };
static bool gLoadedSettings = false;

// every live tweak in one blob so an exported map restores the exact
// session: brush setup, tool selections, prop tuning, UI
struct TuneBlob {
    float brushRadius = 2.5f, brushStrength = 1.0f, paintTarget = 1.0f;
    float falloff = 1.0f, grassDensity = 1.0f, terraceStep = 2.0f;
    float propScale = 1.0f, propScaleRand = 0.35f, propDensity = 2.0f;
    float propSpacing = 2.0f, propYawFixed = 0.0f, uiScale = 1.5f;
    int stamp = 1, grassPattern = 0, showGrass = 1, randomYaw = 1;
    int sculptTool = 0, paintLayer = 0, detailTool = 0, propTool = 0;
    int propCat = 0, shadows = 1;
};
static TuneBlob gTune;
static bool gLoadedTune = false;

static void save_map(const char* path)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        SDL_Log("save failed: %s", path);
        return;
    }
    const char magic[8] = { 'T','E','R','M','A','P','0','6' };
    fwrite(magic, 1, 8, f);
    fwrite(gHeights.data(), sizeof(float), gHeights.size(), f);
    fwrite(gMask.data(), 1, gMask.size(), f);
    fwrite(gKill.data(), 1, gKill.size(), f);
    fwrite(gMask2.data(), 1, gMask2.size(), f);
    // props: identified by "category/label" so mesh order can change
    Uint32 n = (Uint32)gProps.size();
    fwrite(&n, 4, 1, f);
    for (const PropInst& pi : gProps) {
        std::string id = gPropMeshes[pi.mesh].category + "/" +
                         gPropMeshes[pi.mesh].label;
        Uint16 len = (Uint16)id.size();
        fwrite(&len, 2, 1, f);
        fwrite(id.data(), 1, len, f);
        float tr[5] = { pi.x, pi.y, pi.z, pi.yaw, pi.scale };
        fwrite(tr, sizeof(float), 5, f);
    }
    fwrite(&gSetBlade, sizeof(float), 1, f);
    fwrite(&gSetEdge, sizeof(float), 1, f);
    fwrite(gSetCam, sizeof(float), 5, f);
    fwrite(&gTune, sizeof gTune, 1, f);
    fclose(f);
    SDL_Log("saved %s", path);
}

static bool load_map(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "TERMAP0", 7) != 0) {
        fclose(f);
        return false;
    }
    fread(gHeights.data(), sizeof(float), gHeights.size(), f);
    fread(gMask.data(), 1, gMask.size(), f);
    if (magic[7] >= '2')
        fread(gKill.data(), 1, gKill.size(), f);
    else
        std::fill(gKill.begin(), gKill.end(), (Uint8)0);
    if (magic[7] >= '3')
        fread(gMask2.data(), 1, gMask2.size(), f);
    else
        std::fill(gMask2.begin(), gMask2.end(), (Uint8)0);
    gProps.clear();
    if (magic[7] >= '4') {
        Uint32 n = 0;
        if (fread(&n, 4, 1, f) == 1) {
            for (Uint32 i = 0; i < n; i++) {
                Uint16 len = 0;
                if (fread(&len, 2, 1, f) != 1)
                    break;
                std::string id(len, '\0');
                fread(id.data(), 1, len, f);
                float tr[5];
                if (fread(tr, sizeof(float), 5, f) != 5)
                    break;
                int meshIdx = -1;
                for (int mi = 0; mi < (int)gPropMeshes.size(); mi++)
                    if (gPropMeshes[mi].category + "/" +
                        gPropMeshes[mi].label == id) {
                        meshIdx = mi;
                        break;
                    }
                if (meshIdx >= 0)
                    gProps.push_back({ meshIdx, tr[0], tr[1], tr[2],
                                       tr[3], tr[4] });
            }
        }
    }
    gLoadedSettings = false;
    gLoadedTune = false;
    if (magic[7] >= '5') {
        if (fread(&gSetBlade, sizeof(float), 1, f) == 1 &&
            fread(&gSetEdge, sizeof(float), 1, f) == 1 &&
            fread(gSetCam, sizeof(float), 5, f) == 5)
            gLoadedSettings = true;
    }
    if (magic[7] >= '6' && fread(&gTune, sizeof gTune, 1, f) == 1)
        gLoadedTune = true;
    fclose(f);
    gHeightsDirty = gMaskDirty = gMask2Dirty = gKillDirty = true;
    SDL_Log("loaded %s", path);
    return true;
}

// async file-dialog plumbing (SDL may invoke the callback off-thread:
// stash the result, act on it from the main loop)
static volatile int gDialogAction = 0;   // 1 = save, 2 = load
static char gDialogFile[1024];
static const SDL_DialogFileFilter kMapFilters[] = {
    { "Windward map", "wmap" },
};

static void SDLCALL map_dialog_cb(void* userdata,
                                  const char* const* filelist, int)
{
    if (filelist && filelist[0]) {
        SDL_strlcpy(gDialogFile, filelist[0], sizeof gDialogFile);
        gDialogAction = (int)(intptr_t)userdata;
    }
}

// stamp the demo diorama for --shot: a wobbly two-rut path plus a low hill
static void stamp_demo_scene()
{
    for (int step = 0; step <= 400; step++) {
        float t = step / 400.0f;
        float z = -TER_HALF + 2.0f * TER_HALF * t;
        float wob = sinf(t * 5.2f) * 2.2f + sinf(t * 11.7f + 1.3f) * 0.8f;
        float x = 4.0f + wob;
        // two ruts with a grassy strip between, like the reference painting
        apply_brush(BRUSH_DIRT, x - 0.55f, z, 0.55f, 0.05f, false);
        apply_brush(BRUSH_DIRT, x + 0.55f, z, 0.55f, 0.05f, false);
        apply_brush(BRUSH_DIRT, x, z, 0.35f, 0.012f, false);
    }
    // dirt patch clearing, like the diorama's bare spot
    for (int i = 0; i < 30; i++) {
        float a = i * 0.7f;
        apply_brush(BRUSH_DIRT, -6.0f + cosf(a) * 1.8f, 5.0f + sinf(a) * 1.2f,
                    1.6f, 0.03f, false);
    }
    // gentle hill in the back corner
    for (int i = 0; i < 60; i++)
        apply_brush(BRUSH_RAISE, -12.0f, -14.0f, 9.0f, 0.016f, false);
    // steep bluff on the right so the triplanar cliff shows
    for (int i = 0; i < 260; i++)
        apply_brush(BRUSH_RAISE, 15.0f, -8.0f, 5.5f, 0.016f, false);
    for (int i = 0; i < 40; i++)
        apply_brush(BRUSH_SMOOTH, 15.0f, -8.0f, 7.0f, 0.010f, false);
}

// ------------------------------------------------------------------ main

int main(int argc, char** argv)
{
    const char* shotPath = nullptr;
    if (argc >= 3 && SDL_strcmp(argv[1], "--shot") == 0)
        shotPath = argv[2];

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 1;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window* win = SDL_CreateWindow("terrain editor", 1280, 720,
                                       SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!win) {
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
        return 1;
    }
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx || !gl_load_functions())
        return 1;
    SDL_GL_SetSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    float uiScale = 1.5f;
    auto applyUiScale = [&uiScale]() {
        ImGuiStyle s;
        ImGui::StyleColorsDark(&s);
        s.WindowRounding = 6.0f;
        s.ScaleAllSizes(uiScale);
        ImGui::GetStyle() = s;
        ImGui::GetIO().FontGlobalScale = uiScale;
    };
    applyUiScale();
    ImGui_ImplSDL3_InitForOpenGL(win, ctx);
    ImGui_ImplOpenGL3_Init("#version 330");

    GLuint terProg = make_program(TER_VS, TER_FS);
    GLuint grassProg = make_program(GRASS_VS, GRASS_FS);
    GLuint skyProg = make_program(SKY_VS, SKY_FS);
    GLuint propProg = make_program(PROP_VS, PROP_FS);
    GLuint depthTerProg = make_program(TER_VS, DEPTH_FS);
    GLuint depthPropProg = make_program(PROP_VS, DEPTH_PROP_FS);
    make_stamps();

    // shadow map
    const int SHADOW_N = 2048;
    GLuint shadowTex = 0, shadowFbo = 0;
    glGenTextures(1, &shadowTex);
    glBindTexture(GL_TEXTURE_2D, shadowTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_N, SHADOW_N,
                 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &shadowFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, shadowTex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        SDL_Log("shadow FBO incomplete");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // sun matrices (static light direction shared with the shaders)
    Mat4 lightMvp;
    {
        float L[3] = { 0.35f, 0.8f, -0.45f };
        float ll = sqrtf(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
        float eye[3] = { L[0] / ll * 60.0f, L[1] / ll * 60.0f,
                         L[2] / ll * 60.0f };
        float at[3] = { 0, 0, 0 };
        Mat4 lv = mat4_lookat(eye, at);
        Mat4 lp = mat4_ortho(-40, 40, -40, 40, 15.0f, 130.0f);
        lightMvp = mat4_mul(lp, lv);
    }

    // asset textures live next to the exe's source tree
    char base[512];
    SDL_snprintf(base, sizeof base, "%s", SDL_GetBasePath());
    // walk up from build dirs to terrain\ if needed: try a few candidates
    const char* candidates[] = { "assets/", "../assets/", "../../assets/",
                                 "../../../assets/" };
    GLuint grassTex = 0, dirtTex = 0, dirt2Tex = 0, cliffTex = 0;
    char assetsDir[700] = { 0 };
    for (const char* c : candidates) {
        char p[600];
        SDL_snprintf(p, sizeof p, "%s%sgrass.bmp", base, c);
        grassTex = load_bmp_texture(p);
        if (grassTex) {
            SDL_snprintf(assetsDir, sizeof assetsDir, "%s%s", base, c);
            SDL_snprintf(p, sizeof p, "%s%sdirt.bmp", base, c);
            dirtTex = load_bmp_texture(p);
            SDL_snprintf(p, sizeof p, "%s%sdirt2.bmp", base, c);
            dirt2Tex = load_bmp_texture(p);
            SDL_snprintf(p, sizeof p, "%s%scliff.bmp", base, c);
            cliffTex = load_bmp_texture(p);
            break;
        }
    }
    if (!grassTex || !dirtTex || !dirt2Tex || !cliffTex) {
        SDL_Log("could not find assets/grass|dirt|dirt2|cliff.bmp near exe");
        return 1;
    }
    {
        char brushDir[760];
        SDL_snprintf(brushDir, sizeof brushDir, "%sbrushes", assetsDir);
        load_stamp_files(brushDir);
        prune_square_stamps();
        scan_props(std::string(assetsDir) + "props");
    }

    // terrain grid (static xz, heights come from the texture)
    std::vector<float> verts;
    verts.reserve((GRID_N + 1) * (GRID_N + 1) * 2);
    for (int j = 0; j <= GRID_N; j++)
        for (int i = 0; i <= GRID_N; i++) {
            verts.push_back(-TER_HALF + 2.0f * TER_HALF * i / GRID_N);
            verts.push_back(-TER_HALF + 2.0f * TER_HALF * j / GRID_N);
        }
    std::vector<unsigned> idx;
    idx.reserve(GRID_N * GRID_N * 6);
    for (int j = 0; j < GRID_N; j++)
        for (int i = 0; i < GRID_N; i++) {
            unsigned a = j * (GRID_N + 1) + i;
            unsigned b = a + 1;
            unsigned c = a + (GRID_N + 1);
            unsigned d = c + 1;
            idx.insert(idx.end(), { a, c, b, b, c, d });
        }
    GLuint terVao = 0, terVbo = 0, terIbo = 0;
    glGenVertexArrays(1, &terVao);
    glGenBuffers(1, &terVbo);
    glGenBuffers(1, &terIbo);
    glBindVertexArray(terVao);
    glBindBuffer(GL_ARRAY_BUFFER, terVbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terIbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned),
                 idx.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);

    // height + mask textures
    glGenTextures(1, &gHeightTex);
    glBindTexture(GL_TEXTURE_2D, gHeightTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenTextures(1, &gMaskTex);
    glBindTexture(GL_TEXTURE_2D, gMaskTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenTextures(1, &gMask2Tex);
    glBindTexture(GL_TEXTURE_2D, gMask2Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenTextures(1, &gKillTex);
    glBindTexture(GL_TEXTURE_2D, gKillTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // grass blade geometry: two crossed tapered quads, 12 verts
    // per-vertex: x (-1..1), y (0..1), plane (0/1)
    const float blade[12][3] = {
        { -1, 0, 0 }, { 1, 0, 0 }, { -1, 1, 0 },
        { 1, 0, 0 }, { 1, 1, 0 }, { -1, 1, 0 },
        { -1, 0, 1 }, { 1, 0, 1 }, { -1, 1, 1 },
        { 1, 0, 1 }, { 1, 1, 1 }, { -1, 1, 1 },
    };
    // instances: jittered grid, deterministic
    std::vector<float> inst;
    inst.reserve(GRASS_N * GRASS_N * 4);
    unsigned rng = 12345u;
    auto frand = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return (rng >> 8) * (1.0f / 16777216.0f);
    };
    for (int j = 0; j < GRASS_N; j++)
        for (int i = 0; i < GRASS_N; i++) {
            float x = -TER_HALF + 2.0f * TER_HALF * (i + frand()) / GRASS_N;
            float z = -TER_HALF + 2.0f * TER_HALF * (j + frand()) / GRASS_N;
            inst.push_back(x);
            inst.push_back(z);
            inst.push_back(frand() * 6.2831853f);
            inst.push_back(frand());
        }
    int instCount = GRASS_N * GRASS_N;

    GLuint grassVao = 0, bladeVbo = 0, instVbo = 0;
    glGenVertexArrays(1, &grassVao);
    glGenBuffers(1, &bladeVbo);
    glGenBuffers(1, &instVbo);
    glBindVertexArray(grassVao);
    glBindBuffer(GL_ARRAY_BUFFER, bladeVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof blade, blade, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, instVbo);
    glBufferData(GL_ARRAY_BUFFER, inst.size() * sizeof(float), inst.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glVertexAttribDivisor(1, 1);
    glBindVertexArray(0);

    GLuint emptyVao = 0;
    glGenVertexArrays(1, &emptyVao);

    // camera + editor state
    float yaw = 0.0f, pitch = -0.42f, fov = 55.0f;
    float camPos[3] = { 0.0f, 12.0f, 30.0f };
    float brushRadius = 2.5f;
    float brushStrength = 1.0f;
    float paintTarget = 1.0f;
    float edgeBreak = 0.5f;
    float bladeDensity = 0.8f;
    bool showGrass = true;
    bool shadowsOn = true;
    // prop tools
    int activeTab = 0;          // 0 sculpt, 1 paint, 2 details, 3 props
    int sculptTool = 0, paintLayer = 0, detailTool = 0;
    int propTool = 0;           // 0 place, 1 scatter, 2 erase, 3 select
    int propCat = 0, propSel = -1, selInst = -1;
    float propScale = 1.0f, propScaleRand = 0.35f, propYawFixed = 0.0f;
    float propDensity = 2.0f, propSpacing = 2.0f;
    bool propRandomYaw = true;
    unsigned propRng = 777u;
    auto pRand = [&propRng]() {
        propRng = propRng * 1664525u + 1013904223u;
        return (propRng >> 8) * (1.0f / 16777216.0f);
    };
    auto syncSettingsOut = [&]() {
        gSetBlade = bladeDensity;
        gSetEdge = edgeBreak;
        gSetCam[0] = camPos[0]; gSetCam[1] = camPos[1]; gSetCam[2] = camPos[2];
        gSetCam[3] = yaw; gSetCam[4] = pitch;
        gTune.brushRadius = brushRadius;
        gTune.brushStrength = brushStrength;
        gTune.paintTarget = paintTarget;
        gTune.falloff = gBrushFalloff;
        gTune.grassDensity = gGrassDensity;
        gTune.terraceStep = gTerraceStep;
        gTune.propScale = propScale;
        gTune.propScaleRand = propScaleRand;
        gTune.propDensity = propDensity;
        gTune.propSpacing = propSpacing;
        gTune.propYawFixed = propYawFixed;
        gTune.uiScale = uiScale;
        gTune.stamp = gStamp;
        gTune.grassPattern = gGrassPattern;
        gTune.showGrass = showGrass ? 1 : 0;
        gTune.randomYaw = propRandomYaw ? 1 : 0;
        gTune.sculptTool = sculptTool;
        gTune.paintLayer = paintLayer;
        gTune.detailTool = detailTool;
        gTune.propTool = propTool;
        gTune.propCat = propCat;
        gTune.shadows = shadowsOn ? 1 : 0;
    };
    auto applySettingsIn = [&]() {
        if (gLoadedSettings) {
            bladeDensity = gSetBlade;
            edgeBreak = gSetEdge;
            camPos[0] = gSetCam[0]; camPos[1] = gSetCam[1];
            camPos[2] = gSetCam[2];
            yaw = gSetCam[3]; pitch = gSetCam[4];
        }
        if (gLoadedTune) {
            brushRadius = gTune.brushRadius;
            brushStrength = gTune.brushStrength;
            paintTarget = gTune.paintTarget;
            gBrushFalloff = gTune.falloff;
            gGrassDensity = gTune.grassDensity;
            gTerraceStep = gTune.terraceStep;
            propScale = gTune.propScale;
            propScaleRand = gTune.propScaleRand;
            propDensity = gTune.propDensity;
            propSpacing = gTune.propSpacing;
            propYawFixed = gTune.propYawFixed;
            gStamp = SDL_clamp(gTune.stamp, 0, (int)gStamps.size() - 1);
            gGrassPattern = SDL_clamp(gTune.grassPattern, 0, 2);
            showGrass = gTune.showGrass != 0;
            propRandomYaw = gTune.randomYaw != 0;
            sculptTool = SDL_clamp(gTune.sculptTool, 0, 4);
            paintLayer = SDL_clamp(gTune.paintLayer, 0, 2);
            detailTool = SDL_clamp(gTune.detailTool, 0, 1);
            propTool = SDL_clamp(gTune.propTool, 0, 3);
            propCat = gTune.propCat;
            shadowsOn = gTune.shadows != 0;
            update_stamp_thumbnails();
            if (fabsf(uiScale - gTune.uiScale) > 0.01f) {
                uiScale = gTune.uiScale;
                applyUiScale();
            }
        }
    };
    BrushMode mode = BRUSH_RAISE;
    bool wasPainting = false;
    bool running = true;
    double simTime = 0.0;
    int frame = 0;
    Uint64 prev = SDL_GetTicksNS();

    char mapPath[600];
    SDL_snprintf(mapPath, sizeof mapPath, "%s../map.bin", SDL_GetBasePath());

    if (shotPath) {
        stamp_demo_scene();
        simTime = 7.0;
        yaw = 0.0f;
        pitch = -0.30f;
        camPos[0] = 4.5f; camPos[1] = 7.0f; camPos[2] = 33.0f;
    } else {
        load_map(mapPath);
    }

    SDL_Log("RMB=look WASD/QE=fly LMB=brush wheel=radius "
            "1=raise(+shift lower) 2=smooth 3=dirt 4=grass G=grass F5/F9=save/load");

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL3_ProcessEvent(&e);
            switch (e.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (e.key.key == SDLK_ESCAPE) running = false;
                if (e.key.key == SDLK_1) mode = BRUSH_RAISE;
                if (e.key.key == SDLK_2) mode = BRUSH_SMOOTH;
                if (e.key.key == SDLK_5) mode = BRUSH_FLATTEN;
                if (e.key.key == SDLK_3) mode = BRUSH_DIRT;
                if (e.key.key == SDLK_4) mode = BRUSH_GRASS;
                if (e.key.key == SDLK_6) mode = BRUSH_ERASEDIRT;
                if (e.key.key == SDLK_G) showGrass = !showGrass;
                if (e.key.key == SDLK_Z && (e.key.mod & SDL_KMOD_CTRL)) {
                    if (e.key.mod & SDL_KMOD_SHIFT) do_redo(); else do_undo();
                }
                if (e.key.key == SDLK_Y && (e.key.mod & SDL_KMOD_CTRL))
                    do_redo();
                if (e.key.key == SDLK_F5) { syncSettingsOut(); save_map(mapPath); }
                if (e.key.key == SDLK_F9) { load_map(mapPath); applySettingsIn(); }
                if (e.key.key == SDLK_LEFTBRACKET)
                    brushRadius = SDL_max(0.4f, brushRadius - 0.4f);
                if (e.key.key == SDLK_RIGHTBRACKET)
                    brushRadius = SDL_min(10.0f, brushRadius + 0.4f);
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (e.motion.state & SDL_BUTTON_RMASK) {
                    yaw -= e.motion.xrel * 0.003f;
                    pitch -= e.motion.yrel * 0.003f;
                    pitch = SDL_clamp(pitch, -1.45f, 1.45f);
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                if (!ImGui::GetIO().WantCaptureMouse)
                    brushRadius = SDL_clamp(brushRadius + e.wheel.y * 0.3f,
                                            0.4f, 10.0f);
                break;
            }
        }

        // act on finished file dialogs (callback may run off-thread)
        if (gDialogAction == 1) {
            gDialogAction = 0;
            std::string p = gDialogFile;
            if (p.size() < 5 || p.substr(p.size() - 5) != ".wmap")
                p += ".wmap";
            syncSettingsOut();
            save_map(p.c_str());
        } else if (gDialogAction == 2) {
            gDialogAction = 0;
            if (load_map(gDialogFile))
                applySettingsIn();
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        Uint64 now = SDL_GetTicksNS();
        float dt = (float)((double)(now - prev) * 1e-9);
        prev = now;
        dt = SDL_min(dt, 0.05f);
        simTime += dt;

        // fly camera
        const bool* keys = SDL_GetKeyboardState(nullptr);
        {
            float speed = keys[SDL_SCANCODE_LCTRL] ? 24.0f : 9.0f;
            float cy = cosf(yaw), sy = sinf(yaw);
            float cp = cosf(pitch), sp = sinf(pitch);
            float fwd[3] = { -sy * cp, sp, -cy * cp };
            float right[3] = { cy, 0.0f, -sy };
            float mx = 0.0f, my = 0.0f, mz = 0.0f;
            if (keys[SDL_SCANCODE_W]) { mx += fwd[0]; my += fwd[1]; mz += fwd[2]; }
            if (keys[SDL_SCANCODE_S]) { mx -= fwd[0]; my -= fwd[1]; mz -= fwd[2]; }
            if (keys[SDL_SCANCODE_D]) { mx += right[0]; mz += right[2]; }
            if (keys[SDL_SCANCODE_A]) { mx -= right[0]; mz -= right[2]; }
            if (keys[SDL_SCANCODE_E]) my += 1.0f;
            if (keys[SDL_SCANCODE_Q]) my -= 1.0f;
            camPos[0] += mx * speed * dt;
            camPos[1] += my * speed * dt;
            camPos[2] += mz * speed * dt;
        }

        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(win, &w, &h);
        float aspect = h ? (float)w / (float)h : 1.0f;
        float fovRad = fov * 0.01745329252f;

        // brush ray under the cursor
        float hit[3] = { 0, 0, 0 };
        bool hasHit = false;
        {
            float mxp = 0.0f, myp = 0.0f;
            SDL_MouseButtonFlags mb = SDL_GetMouseState(&mxp, &myp);
            float ndcX = 2.0f * mxp / (float)SDL_max(w, 1) - 1.0f;
            float ndcY = 1.0f - 2.0f * myp / (float)SDL_max(h, 1);
            float tanF = tanf(fovRad * 0.5f);
            float cy = cosf(yaw), sy = sinf(yaw);
            float cp = cosf(pitch), sp = sinf(pitch);
            // camera basis (matches cam_rotation)
            float rt[3] = { cy, 0.0f, -sy };
            float up[3] = { sy * sp, cp, cy * sp };
            float fw[3] = { -sy * cp, sp, -cy * cp };
            float rd[3] = {
                fw[0] + rt[0] * ndcX * tanF * aspect + up[0] * ndcY * tanF,
                fw[1] + rt[1] * ndcX * tanF * aspect + up[1] * ndcY * tanF,
                fw[2] + rt[2] * ndcX * tanF * aspect + up[2] * ndcY * tanF,
            };
            float len = sqrtf(rd[0] * rd[0] + rd[1] * rd[1] + rd[2] * rd[2]);
            rd[0] /= len; rd[1] /= len; rd[2] /= len;
            hasHit = ray_terrain(camPos, rd, hit);
            if (ImGui::GetIO().WantCaptureMouse)
                hasHit = false;   // cursor over the panel: never paint through
            bool painting = hasHit && (mb & SDL_BUTTON_LMASK) && !shotPath;
            bool clickEdge = painting && !wasPainting;
            if (painting && activeTab != 3) {
                if (!wasPainting)
                    push_undo();   // one undo step per stroke
                // sculpt modifiers: Ctrl smooths, Alt flattens
                BrushMode active = mode;
                if (mode == BRUSH_RAISE) {
                    if (keys[SDL_SCANCODE_LCTRL])
                        active = BRUSH_SMOOTH;
                    else if (keys[SDL_SCANCODE_LALT] || keys[SDL_SCANCODE_RALT])
                        active = BRUSH_FLATTEN;
                }
                if (!wasPainting && active == BRUSH_FLATTEN)
                    gFlattenTarget = height_at(hit[0], hit[2]);
                apply_brush(active, hit[0], hit[2], brushRadius, dt,
                            keys[SDL_SCANCODE_LSHIFT] != 0, brushStrength,
                            paintTarget);
            } else if (activeTab == 3 && hasHit) {
                // ---- prop tools
                if (propTool == 0 && clickEdge && propSel >= 0 &&
                    load_prop(propSel)) {
                    push_undo();
                    float yaw = propRandomYaw ? pRand() * 6.2831853f
                                              : propYawFixed;
                    float sc = propScale *
                        (1.0f + (pRand() - 0.5f) * 2.0f * propScaleRand);
                    gProps.push_back({ propSel, hit[0],
                                       height_at(hit[0], hit[2]), hit[2],
                                       yaw, sc });
                    selInst = (int)gProps.size() - 1;
                } else if (propTool == 1 && painting && propSel >= 0 &&
                           load_prop(propSel)) {
                    if (clickEdge)
                        push_undo();
                    int attempts = (int)SDL_ceilf(propDensity * dt * 12.0f);
                    for (int a = 0; a < attempts; a++) {
                        float ang = pRand() * 6.2831853f;
                        float rad = sqrtf(pRand()) * brushRadius;
                        float px = hit[0] + cosf(ang) * rad;
                        float pz = hit[2] + sinf(ang) * rad;
                        if (fabsf(px) > TER_HALF || fabsf(pz) > TER_HALF)
                            continue;
                        bool tooClose = false;
                        for (const PropInst& pi : gProps) {
                            float dx = pi.x - px, dz = pi.z - pz;
                            if (dx * dx + dz * dz <
                                propSpacing * propSpacing) {
                                tooClose = true;
                                break;
                            }
                        }
                        if (tooClose)
                            continue;
                        float sc = propScale *
                            (1.0f + (pRand() - 0.5f) * 2.0f * propScaleRand);
                        gProps.push_back({ propSel, px, height_at(px, pz),
                                           pz, pRand() * 6.2831853f, sc });
                    }
                } else if (propTool == 2 && painting) {
                    if (clickEdge)
                        push_undo();
                    for (size_t pi = 0; pi < gProps.size();) {
                        float dx = gProps[pi].x - hit[0];
                        float dz = gProps[pi].z - hit[2];
                        if (dx * dx + dz * dz < brushRadius * brushRadius) {
                            gProps.erase(gProps.begin() + pi);
                            selInst = -1;
                        } else {
                            pi++;
                        }
                    }
                } else if (propTool == 3 && clickEdge) {
                    selInst = -1;
                    float best = 4.0f;   // pick radius (squared below)
                    for (int pi = 0; pi < (int)gProps.size(); pi++) {
                        float dx = gProps[pi].x - hit[0];
                        float dz = gProps[pi].z - hit[2];
                        float d2 = dx * dx + dz * dz;
                        float r = SDL_max(1.5f,
                            gPropMeshes[gProps[pi].mesh].boundR *
                            gProps[pi].scale);
                        if (d2 < r * r && d2 < best * best) {
                            best = sqrtf(d2);
                            selInst = pi;
                        }
                    }
                }
            }
            wasPainting = painting;
            if (selInst >= (int)gProps.size())
                selInst = -1;
        }

        upload_dirty();

        // shadow pass: terrain + props into the depth map from the sun
        if (shadowsOn) {
            glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo);
            glViewport(0, 0, SHADOW_N, SHADOW_N);
            glClear(GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glUseProgram(depthTerProg);
            glUniformMatrix4fv(glGetUniformLocation(depthTerProg, "uMvp"),
                               1, GL_FALSE, lightMvp.m);
            glUniform1f(glGetUniformLocation(depthTerProg, "uHalf"), TER_HALF);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, gHeightTex);
            glUniform1i(glGetUniformLocation(depthTerProg, "uHeight"), 0);
            glBindVertexArray(terVao);
            glDrawElements(GL_TRIANGLES, (GLsizei)idx.size(), GL_UNSIGNED_INT,
                           nullptr);
            if (!gProps.empty()) {
                glUseProgram(depthPropProg);
                glUniformMatrix4fv(
                    glGetUniformLocation(depthPropProg, "uMvp"), 1, GL_FALSE,
                    lightMvp.m);
                glUniform1i(glGetUniformLocation(depthPropProg, "uTex"), 0);
                GLint dModel = glGetUniformLocation(depthPropProg, "uModel");
                GLint dHasTex = glGetUniformLocation(depthPropProg, "uHasTex");
                GLint dGray = glGetUniformLocation(depthPropProg, "uGrayMask");
                for (const PropInst& inst : gProps) {
                    PropMesh& pm = gPropMeshes[inst.mesh];
                    if (!pm.loaded)
                        continue;
                    Mat4 mdl = model_trs(inst.x, inst.y, inst.z, inst.yaw,
                                         inst.scale);
                    glUniformMatrix4fv(dModel, 1, GL_FALSE, mdl.m);
                    glBindVertexArray(pm.vao);
                    for (const PropSubmesh& sub : pm.subs) {
                        const PropMaterial& mat = pm.mats[sub.mat];
                        glUniform1i(dHasTex, mat.tex ? 1 : 0);
                        glUniform1i(dGray, mat.grayMask ? 1 : 0);
                        if (mat.tex)
                            glBindTexture(GL_TEXTURE_2D, mat.tex);
                        glDrawArrays(GL_TRIANGLES, sub.first, sub.count);
                    }
                }
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        glViewport(0, 0, w, h);
        Mat4 camRot = cam_rotation(yaw, pitch);
        Mat4 view = view_matrix(camRot, camPos[0], camPos[1], camPos[2]);
        Mat4 proj = mat4_perspective(fovRad, aspect, 0.1f, 500.0f);
        Mat4 mvp = mat4_mul(proj, view);

        glClearColor(0.66f, 0.80f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);

        // terrain
        glUseProgram(terProg);
        glUniformMatrix4fv(glGetUniformLocation(terProg, "uMvp"), 1, GL_FALSE, mvp.m);
        glUniform1f(glGetUniformLocation(terProg, "uHalf"), TER_HALF);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, gHeightTex);
        glUniform1i(glGetUniformLocation(terProg, "uHeight"), 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, gMaskTex);
        glUniform1i(glGetUniformLocation(terProg, "uMask"), 1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, grassTex);
        glUniform1i(glGetUniformLocation(terProg, "uGrassTex"), 2);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, dirtTex);
        glUniform1i(glGetUniformLocation(terProg, "uDirtTex"), 3);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, cliffTex);
        glUniform1i(glGetUniformLocation(terProg, "uCliffTex"), 4);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, gMask2Tex);
        glUniform1i(glGetUniformLocation(terProg, "uMask2"), 5);
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, dirt2Tex);
        glUniform1i(glGetUniformLocation(terProg, "uDirt2Tex"), 6);
        float brushU[4] = { hit[0], hit[2], brushRadius, hasHit && !shotPath ? 1.0f : 0.0f };
        glUniform4fv(glGetUniformLocation(terProg, "uBrush"), 1, brushU);
        glUniform3fv(glGetUniformLocation(terProg, "uBrushCol"), 1,
                     kBrushColors[mode]);
        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, gStamps[gStamp].tex);
        glUniform1i(glGetUniformLocation(terProg, "uBrushStamp"), 7);
        // falloff is baked into the stamp textures by
        // update_stamp_thumbnails, so the preview shader applies none
        glUniform1f(glGetUniformLocation(terProg, "uBrushFalloff"), 1.0f);
        glUniform1f(glGetUniformLocation(terProg, "uEdgeBreak"), edgeBreak);
        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_2D, shadowTex);
        glUniform1i(glGetUniformLocation(terProg, "uShadowMap"), 8);
        glUniformMatrix4fv(glGetUniformLocation(terProg, "uLightMvp"), 1,
                           GL_FALSE, lightMvp.m);
        glUniform1i(glGetUniformLocation(terProg, "uShadowsOn"),
                    shadowsOn ? 1 : 0);
        glBindVertexArray(terVao);
        glDrawElements(GL_TRIANGLES, (GLsizei)idx.size(), GL_UNSIGNED_INT, nullptr);

        // props
        if (!gProps.empty()) {
            glUseProgram(propProg);
            glUniformMatrix4fv(glGetUniformLocation(propProg, "uMvp"), 1,
                               GL_FALSE, mvp.m);
            glDisable(GL_CULL_FACE);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, shadowTex);
            glUniform1i(glGetUniformLocation(propProg, "uShadowMap"), 5);
            glUniformMatrix4fv(glGetUniformLocation(propProg, "uLightMvp"),
                               1, GL_FALSE, lightMvp.m);
            glUniform1i(glGetUniformLocation(propProg, "uShadowsOn"),
                        shadowsOn ? 1 : 0);
            glActiveTexture(GL_TEXTURE0);
            glUniform1i(glGetUniformLocation(propProg, "uTex"), 0);
            GLint locModel = glGetUniformLocation(propProg, "uModel");
            GLint locKd = glGetUniformLocation(propProg, "uKd");
            GLint locKa = glGetUniformLocation(propProg, "uKa");
            GLint locBoundH = glGetUniformLocation(propProg, "uBoundH");
            GLint locHasTex = glGetUniformLocation(propProg, "uHasTex");
            GLint locGray = glGetUniformLocation(propProg, "uGrayMask");
            for (int pi = 0; pi < (int)gProps.size(); pi++) {
                const PropInst& inst = gProps[pi];
                PropMesh& pm = gPropMeshes[inst.mesh];
                if (!pm.loaded)
                    continue;
                Mat4 mdl = model_trs(inst.x, inst.y, inst.z, inst.yaw,
                                     inst.scale);
                glUniformMatrix4fv(locModel, 1, GL_FALSE, mdl.m);
                glUniform1f(locBoundH, pm.boundH);
                glBindVertexArray(pm.vao);
                float sel = (pi == selInst && activeTab == 3) ? 1.45f : 1.0f;
                for (const PropSubmesh& sub : pm.subs) {
                    const PropMaterial& mat = pm.mats[sub.mat];
                    glUniform3f(locKd, mat.kd[0] * sel, mat.kd[1] * sel,
                                mat.kd[2] * sel);
                    glUniform3f(locKa, mat.ka[0] * sel, mat.ka[1] * sel,
                                mat.ka[2] * sel);
                    glUniform1i(locHasTex, mat.tex ? 1 : 0);
                    glUniform1i(locGray, mat.grayMask ? 1 : 0);
                    if (mat.tex)
                        glBindTexture(GL_TEXTURE_2D, mat.tex);
                    glDrawArrays(GL_TRIANGLES, sub.first, sub.count);
                }
            }
        }

        // grass
        if (showGrass) {
            glUseProgram(grassProg);
            glUniformMatrix4fv(glGetUniformLocation(grassProg, "uMvp"), 1, GL_FALSE, mvp.m);
            glUniform1f(glGetUniformLocation(grassProg, "uHalf"), TER_HALF);
            glUniform1f(glGetUniformLocation(grassProg, "uTime"), (float)simTime);
            glUniform1f(glGetUniformLocation(grassProg, "uDensity"), bladeDensity);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, gHeightTex);
            glUniform1i(glGetUniformLocation(grassProg, "uHeight"), 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, gMaskTex);
            glUniform1i(glGetUniformLocation(grassProg, "uMask"), 1);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, grassTex);
            glUniform1i(glGetUniformLocation(grassProg, "uGrassTex"), 2);
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, gKillTex);
            glUniform1i(glGetUniformLocation(grassProg, "uKill"), 3);
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, gMask2Tex);
            glUniform1i(glGetUniformLocation(grassProg, "uMask2"), 4);
            glDisable(GL_CULL_FACE);
            glBindVertexArray(grassVao);
            glDrawArraysInstanced(GL_TRIANGLES, 0, 12, instCount);
        }

        // sky fills the rest
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        glUseProgram(skyProg);
        glUniformMatrix4fv(glGetUniformLocation(skyProg, "uCamRot"), 1, GL_FALSE, camRot.m);
        glUniform1f(glGetUniformLocation(skyProg, "uAspect"), aspect);
        glUniform1f(glGetUniformLocation(skyProg, "uTanHalfFov"), tanf(fovRad * 0.5f));
        glUniform1f(glGetUniformLocation(skyProg, "uTime"), (float)simTime);
        glBindVertexArray(emptyVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDepthMask(GL_TRUE);

        // tool panel
        if (!shotPath) {
            // docked inspector: pinned to the right edge, full height
            const float panelW = 320.0f * uiScale;
            ImGui::SetNextWindowPos(ImVec2((float)w - panelW, 0.0f));
            ImGui::SetNextWindowSize(ImVec2(panelW, (float)h));
            ImGui::Begin("Terrain", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoCollapse |
                         ImGuiWindowFlags_NoBringToFrontOnFocus);

            // Unity-style brush gallery + shared size/opacity controls
            auto brushGallery = [&]() {
                ImGui::SeparatorText("Brushes");
                for (int i = 0; i < (int)gStamps.size(); i++) {
                    if (i % 5)
                        ImGui::SameLine();
                    bool sel = (i == gStamp);
                    if (sel)
                        ImGui::PushStyleColor(ImGuiCol_Button,
                                              ImVec4(0.26f, 0.55f, 0.96f, 1.0f));
                    ImGui::PushID(i);
                    if (ImGui::ImageButton("stamp",
                            (ImTextureID)(intptr_t)gStamps[i].tex,
                            ImVec2(40 * uiScale, 40 * uiScale)))
                        gStamp = i;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("%s", gStamps[i].name.c_str());
                    ImGui::PopID();
                    if (sel)
                        ImGui::PopStyleColor();
                }
                ImGui::SliderFloat("Brush Size", &brushRadius, 0.4f, 10.0f, "%.1f");
                if (ImGui::SliderFloat("Falloff", &gBrushFalloff,
                                       0.05f, 8.0f, "%.2f",
                                       ImGuiSliderFlags_Logarithmic))
                    update_stamp_thumbnails();
            };

            if (ImGui::BeginTabBar("tools")) {
                if (ImGui::BeginTabItem("Sculpt")) {
                    activeTab = 0;
                    const char* tools[] = { "Raise or Lower Terrain",
                                            "Smooth Height", "Flatten",
                                            "Sharpen Edges", "Terrace" };
                    ImGui::Combo("##sculpttool", &sculptTool, tools, 5);
                    static const char* helps[] = {
                        "Left click to raise.\nHold Shift and left click to "
                        "lower.\nHold Ctrl to smooth.\nHold Alt to flatten.",
                        "Left click to smooth the height.",
                        "Left click to flatten toward the height where the "
                        "stroke began.",
                        "Left click to steepen slopes into defined sides "
                        "and edges.",
                        "Left click to step the terrain into flat plateaus "
                        "with cliff sides.",
                    };
                    ImGui::TextWrapped("%s", helps[sculptTool]);
                    static const BrushMode toolModes[] = {
                        BRUSH_RAISE, BRUSH_SMOOTH, BRUSH_FLATTEN,
                        BRUSH_SHARPEN, BRUSH_TERRACE,
                    };
                    mode = toolModes[sculptTool];
                    brushGallery();
                    ImGui::SliderFloat("Strength", &brushStrength,
                                       0.1f, 3.0f, "%.1f");
                    if (mode == BRUSH_TERRACE)
                        ImGui::SliderFloat("Step Height", &gTerraceStep,
                                           0.5f, 6.0f, "%.1f");
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Paint")) {
                    activeTab = 1;
                    ImGui::TextWrapped("Paints the selected layer onto the "
                                       "terrain.");
                    ImGui::SeparatorText("Terrain Layers");
                    struct Layer { const char* name; GLuint tex; BrushMode m; };
                    const Layer layers[] = {
                        { "Grass",     grassTex, BRUSH_ERASEDIRT },
                        { "Path Dirt", dirtTex,  BRUSH_DIRT },
                        { "Soft Dirt", dirt2Tex, BRUSH_DIRT2 },
                    };
                    for (int i = 0; i < 3; i++) {
                        if (i)
                            ImGui::SameLine();
                        bool sel = (paintLayer == i);
                        if (sel)
                            ImGui::PushStyleColor(ImGuiCol_Button,
                                ImVec4(0.26f, 0.55f, 0.96f, 1.0f));
                        ImGui::PushID(100 + i);
                        if (ImGui::ImageButton("layer",
                                (ImTextureID)(intptr_t)layers[i].tex,
                                ImVec2(48 * uiScale, 48 * uiScale)))
                            paintLayer = i;
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("%s", layers[i].name);
                        ImGui::PopID();
                        if (sel)
                            ImGui::PopStyleColor();
                    }
                    mode = layers[paintLayer].m;
                    brushGallery();
                    // ceiling a stroke paints up to -- hovering never
                    // overshoots it (Unity's Target Strength)
                    ImGui::SliderFloat("Target Strength", &paintTarget,
                                       0.05f, 1.0f, "%.2f");
                    ImGui::SliderFloat("Edge Breakup", &edgeBreak,
                                       0.0f, 1.0f, "%.2f");
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Details")) {
                    activeTab = 2;
                    ImGui::TextWrapped("Paint grass blades. Density and "
                                       "pattern are baked into the stroke.");
                    if (ImGui::RadioButton("Paint Blades", detailTool == 0))
                        detailTool = 0;
                    ImGui::SameLine();
                    if (ImGui::RadioButton("Erase Blades", detailTool == 1))
                        detailTool = 1;
                    mode = detailTool == 0 ? BRUSH_GRASS : BRUSH_KILLGRASS;
                    if (detailTool == 0) {
                        ImGui::SliderFloat("Density", &gGrassDensity,
                                           0.0f, 1.0f, "%.2f");
                        ImGui::Combo("Pattern", &gGrassPattern,
                                     kGrassPatterns, 3);
                    }
                    ImGui::Checkbox("Show Grass", &showGrass);
                    ImGui::SliderFloat("Global Density", &bladeDensity,
                                       0.1f, 1.0f, "%.2f");
                    brushGallery();
                    ImGui::SliderFloat("Strength", &brushStrength,
                                       0.1f, 3.0f, "%.1f");
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Props")) {
                    activeTab = 3;
                    ImGui::RadioButton("Place", &propTool, 0);
                    ImGui::SameLine();
                    ImGui::RadioButton("Scatter", &propTool, 1);
                    ImGui::SameLine();
                    ImGui::RadioButton("Erase", &propTool, 2);
                    ImGui::SameLine();
                    ImGui::RadioButton("Select", &propTool, 3);
                    ImGui::Separator();
                    if (gPropCats.empty()) {
                        ImGui::TextWrapped("No props found in assets/props/");
                    } else {
                        propCat = SDL_clamp(propCat, 0,
                                            (int)gPropCats.size() - 1);
                        if (ImGui::BeginCombo("Category",
                                gPropCats[propCat].name.c_str())) {
                            for (int c = 0; c < (int)gPropCats.size(); c++)
                                if (ImGui::Selectable(
                                        gPropCats[c].name.c_str(),
                                        c == propCat))
                                    propCat = c;
                            ImGui::EndCombo();
                        }
                        ImGui::BeginChild("proplist",
                            ImVec2(0, 170 * uiScale), ImGuiChildFlags_Borders);
                        for (int mi : gPropCats[propCat].meshes) {
                            if (ImGui::Selectable(
                                    gPropMeshes[mi].label.c_str(),
                                    mi == propSel))
                                propSel = mi;
                        }
                        ImGui::EndChild();
                    }
                    if (propTool == 0) {
                        ImGui::TextWrapped("Click the ground to place the "
                                           "selected prop.");
                        ImGui::SliderFloat("Scale", &propScale, 0.2f, 3.0f, "%.2f");
                        ImGui::SliderFloat("Scale Random", &propScaleRand,
                                           0.0f, 1.0f, "%.2f");
                        ImGui::Checkbox("Random Rotation", &propRandomYaw);
                        if (!propRandomYaw)
                            ImGui::SliderAngle("Rotation", &propYawFixed,
                                               0.0f, 360.0f);
                    } else if (propTool == 1) {
                        ImGui::TextWrapped("Drag to scatter props inside "
                                           "the brush.");
                        ImGui::SliderFloat("Density", &propDensity,
                                           0.2f, 10.0f, "%.1f");
                        ImGui::SliderFloat("Spacing", &propSpacing,
                                           0.3f, 8.0f, "%.1f");
                        ImGui::SliderFloat("Scale", &propScale, 0.2f, 3.0f, "%.2f");
                        ImGui::SliderFloat("Scale Random", &propScaleRand,
                                           0.0f, 1.0f, "%.2f");
                        ImGui::SliderFloat("Brush Size", &brushRadius,
                                           0.4f, 10.0f, "%.1f");
                    } else if (propTool == 2) {
                        ImGui::TextWrapped("Drag to erase props inside the "
                                           "brush.");
                        ImGui::SliderFloat("Brush Size", &brushRadius,
                                           0.4f, 10.0f, "%.1f");
                    } else if (selInst >= 0) {
                        PropInst& si = gProps[selInst];
                        ImGui::SeparatorText(
                            gPropMeshes[si.mesh].label.c_str());
                        ImGui::SliderAngle("Rotation##sel", &si.yaw,
                                           0.0f, 360.0f);
                        ImGui::SliderFloat("Scale##sel", &si.scale,
                                           0.2f, 3.0f, "%.2f");
                        float pos[2] = { si.x, si.z };
                        if (ImGui::DragFloat2("Position", pos, 0.05f)) {
                            si.x = SDL_clamp(pos[0], -TER_HALF, TER_HALF);
                            si.z = SDL_clamp(pos[1], -TER_HALF, TER_HALF);
                            si.y = height_at(si.x, si.z);
                        }
                        if (ImGui::Button("Delete")) {
                            push_undo();
                            gProps.erase(gProps.begin() + selInst);
                            selInst = -1;
                        }
                        ImGui::SameLine();
                        if (selInst >= 0 && ImGui::Button("Duplicate")) {
                            push_undo();
                            PropInst copy = gProps[selInst];
                            copy.x += 1.0f;
                            gProps.push_back(copy);
                            selInst = (int)gProps.size() - 1;
                        }
                    } else {
                        ImGui::TextWrapped("Click a prop to select it.");
                    }
                    ImGui::TextDisabled("%d props placed", (int)gProps.size());
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Map")) {
                    activeTab = 4;
                    if (ImGui::Button("Save (F5)")) {
                        syncSettingsOut();
                        save_map(mapPath);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Load (F9)")) {
                        load_map(mapPath);
                        applySettingsIn();
                    }
                    ImGui::Checkbox("Shadows", &shadowsOn);
                    ImGui::SeparatorText("File");
                    if (ImGui::Button("Export Map..."))
                        SDL_ShowSaveFileDialog(map_dialog_cb, (void*)1, win,
                                               kMapFilters, 1, nullptr);
                    ImGui::SameLine();
                    if (ImGui::Button("Import Map..."))
                        SDL_ShowOpenFileDialog(map_dialog_cb, (void*)2, win,
                                               kMapFilters, 1, nullptr,
                                               false);
                    ImGui::TextWrapped("Exports everything: terrain, paint "
                                       "layers, grass, props, view settings "
                                       "and camera.");
                    if (ImGui::SliderFloat("UI Scale", &uiScale,
                                           1.0f, 2.5f, "%.1f"))
                        applyUiScale();
                    ImGui::TextDisabled("RMB look  WASD/QE fly  [ ] size");
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            ImGui::End();
        }
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(win);

        if (shotPath && ++frame == 5) {
            save_screenshot(win, shotPath);
            running = false;
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DestroyContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
