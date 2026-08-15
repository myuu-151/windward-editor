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

    // noise-broken blend edges: painted borders go ragged by themselves
    float n = fbm(vWorld.xz * 1.1) - 0.5;
    float edge = smoothstep(0.42, 0.58, m + n * 0.38);
    float edge2 = smoothstep(0.42, 0.58, m2 + n * 0.38);
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
    col *= 0.72 + 0.38 * diff;

    // brush ring
    if (uBrush.w > 0.5) {
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
    // blades take the LOCAL AVERAGE ground color (high mip = blurred), so
    // they never pick up single flower/shadow pixels, and the root matches
    // the ground almost exactly -- blades grow out of the floor instead of
    // sitting on it
    vec3 groundCol = textureLod(uGrassTex, vWorldXz * 0.16, 4.5).rgb;
    vec3 tip  = groundCol * (1.16 + vSeed * 0.08);
    vec3 col = mix(groundCol * 0.94, tip, vV * vV);
    fragColor = vec4(col, 1.0);
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
                 BRUSH_DIRT, BRUSH_DIRT2, BRUSH_ERASEDIRT,
                 BRUSH_GRASS, BRUSH_KILLGRASS };

static const float kBrushColors[8][3] = {
    { 1.0f, 0.85f, 0.3f },   // raise: yellow
    { 0.4f, 0.8f, 1.0f },    // smooth: blue
    { 0.9f, 0.5f, 0.9f },    // flatten: purple
    { 0.72f, 0.5f, 0.28f },  // path dirt: brown
    { 0.85f, 0.75f, 0.5f },  // soft dirt: sand
    { 0.6f, 0.65f, 0.6f },   // erase dirt: grey
    { 0.5f, 1.0f, 0.4f },    // grass blades: green
    { 0.9f, 0.35f, 0.3f },   // remove grass: red
};
static const char* kBrushNames[8] = { "Sculpt", "Smooth", "Flatten",
                                      "Path Dirt", "Soft Dirt", "Erase Dirt",
                                      "Grass Blades", "Remove Grass" };

// flatten pulls terrain toward the height captured when the stroke began
static float gFlattenTarget = 0.0f;

// UE-style brush falloff presets
enum Falloff { FALLOFF_SMOOTH, FALLOFF_LINEAR, FALLOFF_SPHERE, FALLOFF_TIP };
static int gFalloff = FALLOFF_SMOOTH;
static const char* kFalloffNames[] = { "Smooth", "Linear", "Sphere", "Tip" };

static float falloff_weight(float dOverR)
{
    float t = SDL_clamp(1.0f - dOverR, 0.0f, 1.0f);
    switch (gFalloff) {
    case FALLOFF_LINEAR: return t;
    case FALLOFF_SPHERE: return sqrtf(SDL_max(0.0f, 1.0f - dOverR * dOverR));
    case FALLOFF_TIP:    return t * t;
    default:             return t * t * (3.0f - 2.0f * t);
    }
}

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

// brush shapes: circle, square, and a speckled "noise" alpha stamp
enum BrushShape { SHAPE_CIRCLE, SHAPE_SQUARE, SHAPE_NOISE };
static int gShape = SHAPE_CIRCLE;
static const char* kShapeNames[] = { "Circle", "Square", "Noise" };

static float brush_weight(float dx, float dz, float radius, float wx, float wz)
{
    float d = (gShape == SHAPE_SQUARE)
                  ? SDL_max(fabsf(dx), fabsf(dz))
                  : sqrtf(dx * dx + dz * dz);
    if (d > radius)
        return 0.0f;
    float w = falloff_weight(d / radius);
    if (gShape == SHAPE_NOISE) {
        float n = sinf(wx * 12.9898f + wz * 78.233f) * 43758.5453f;
        n -= floorf(n);
        w *= 0.15f + 0.85f * n * n;
    }
    return w;
}

static void apply_brush(BrushMode mode, float cx, float cz, float radius,
                        float dt, bool invert, float strength = 1.0f)
{
    dt *= strength;
    if (mode == BRUSH_RAISE || mode == BRUSH_SMOOTH || mode == BRUSH_FLATTEN) {
        float cell = 2.0f * TER_HALF / (HN - 1);
        int i0 = SDL_clamp((int)((cx - radius + TER_HALF) / cell), 0, HN - 1);
        int i1 = SDL_clamp((int)((cx + radius + TER_HALF) / cell) + 1, 0, HN - 1);
        int j0 = SDL_clamp((int)((cz - radius + TER_HALF) / cell), 0, HN - 1);
        int j1 = SDL_clamp((int)((cz + radius + TER_HALF) / cell) + 1, 0, HN - 1);
        std::vector<float> snap;
        if (mode == BRUSH_SMOOTH)
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
                float rate = SDL_min(1.0f, 10.0f * w * dt);
                auto blend = [rate](Uint8& v, float target) {
                    float nv = v + (target - v) * rate;
                    v = (Uint8)SDL_clamp((int)(nv + 0.5f), 0, 255);
                };
                if (mode == BRUSH_DIRT) {
                    blend(gMask[j * MASK_N + i], 255.0f);
                    blend(gMask2[j * MASK_N + i], 0.0f);
                } else if (mode == BRUSH_DIRT2) {
                    blend(gMask2[j * MASK_N + i], 255.0f);
                    blend(gMask[j * MASK_N + i], 0.0f);
                } else if (mode == BRUSH_ERASEDIRT) {
                    blend(gMask[j * MASK_N + i], 0.0f);
                    blend(gMask2[j * MASK_N + i], 0.0f);
                } else if (mode == BRUSH_KILLGRASS) {
                    blend(gKill[j * MASK_N + i], 255.0f);
                } else {   // grass blades only: density shaped by pattern,
                           // never touches the painted ground layers
                    float dens = gGrassDensity * grass_pattern(x, z);
                    blend(gKill[j * MASK_N + i], (1.0f - dens) * 255.0f);
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

static void save_map(const char* path)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        SDL_Log("save failed: %s", path);
        return;
    }
    const char magic[8] = { 'T','E','R','M','A','P','0','3' };
    fwrite(magic, 1, 8, f);
    fwrite(gHeights.data(), sizeof(float), gHeights.size(), f);
    fwrite(gMask.data(), 1, gMask.size(), f);
    fwrite(gKill.data(), 1, gKill.size(), f);
    fwrite(gMask2.data(), 1, gMask2.size(), f);
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
    fclose(f);
    gHeightsDirty = gMaskDirty = gMask2Dirty = gKillDirty = true;
    SDL_Log("loaded %s", path);
    return true;
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
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 6.0f;
    ImGui_ImplSDL3_InitForOpenGL(win, ctx);
    ImGui_ImplOpenGL3_Init("#version 330");

    GLuint terProg = make_program(TER_VS, TER_FS);
    GLuint grassProg = make_program(GRASS_VS, GRASS_FS);
    GLuint skyProg = make_program(SKY_VS, SKY_FS);

    // asset textures live next to the exe's source tree
    char base[512];
    SDL_snprintf(base, sizeof base, "%s", SDL_GetBasePath());
    // walk up from build dirs to terrain\ if needed: try a few candidates
    const char* candidates[] = { "assets/", "../assets/", "../../assets/",
                                 "../../../assets/" };
    GLuint grassTex = 0, dirtTex = 0, dirt2Tex = 0, cliffTex = 0;
    for (const char* c : candidates) {
        char p[600];
        SDL_snprintf(p, sizeof p, "%s%sgrass.bmp", base, c);
        grassTex = load_bmp_texture(p);
        if (grassTex) {
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
    float bladeDensity = 0.8f;
    BrushMode mode = BRUSH_RAISE;
    bool showGrass = true;
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
                if (e.key.key == SDLK_F5) save_map(mapPath);
                if (e.key.key == SDLK_F9) load_map(mapPath);
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
            if (painting) {
                if (!wasPainting && mode == BRUSH_FLATTEN)
                    gFlattenTarget = height_at(hit[0], hit[2]);
                apply_brush(mode, hit[0], hit[2], brushRadius, dt,
                            keys[SDL_SCANCODE_LSHIFT] != 0, brushStrength);
            }
            wasPainting = painting;
        }

        upload_dirty();

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
        glBindVertexArray(terVao);
        glDrawElements(GL_TRIANGLES, (GLsizei)idx.size(), GL_UNSIGNED_INT, nullptr);

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
            ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(240, 0), ImGuiCond_FirstUseEver);
            ImGui::Begin("Landscape", nullptr, ImGuiWindowFlags_NoCollapse);

            ImGui::SeparatorText("Sculpt");
            for (int i = 0; i <= BRUSH_FLATTEN; i++) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(kBrushColors[i][0], kBrushColors[i][1],
                           kBrushColors[i][2], 1.0f));
                if (ImGui::RadioButton(kBrushNames[i], (int)mode == i))
                    mode = (BrushMode)i;
                ImGui::PopStyleColor();
            }
            if (mode == BRUSH_RAISE)
                ImGui::TextDisabled("hold Shift to lower");

            ImGui::SeparatorText("Paint");
            for (int i = BRUSH_DIRT; i <= BRUSH_KILLGRASS; i++) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(kBrushColors[i][0], kBrushColors[i][1],
                           kBrushColors[i][2], 1.0f));
                if (ImGui::RadioButton(kBrushNames[i], (int)mode == i))
                    mode = (BrushMode)i;
                ImGui::PopStyleColor();
            }

            ImGui::SeparatorText("Brush");
            ImGui::SliderFloat("Radius", &brushRadius, 0.4f, 10.0f, "%.1f");
            ImGui::SliderFloat("Strength", &brushStrength, 0.1f, 3.0f, "%.1f");
            ImGui::Combo("Shape", &gShape, kShapeNames, 3);
            ImGui::Combo("Falloff", &gFalloff, kFalloffNames, 4);
            if (mode == BRUSH_GRASS) {
                ImGui::SeparatorText("Grass Brush");
                ImGui::SliderFloat("Density", &gGrassDensity, 0.0f, 1.0f, "%.2f");
                ImGui::Combo("Pattern", &gGrassPattern, kGrassPatterns, 3);
            }

            ImGui::SeparatorText("View");
            ImGui::Checkbox("Grass", &showGrass);
            ImGui::SliderFloat("Blade Density", &bladeDensity, 0.1f, 1.0f, "%.2f");

            ImGui::SeparatorText("Map");
            if (ImGui::Button("Save (F5)"))
                save_map(mapPath);
            ImGui::SameLine();
            if (ImGui::Button("Load (F9)"))
                load_map(mapPath);

            ImGui::TextDisabled("RMB look  WASD/QE fly  [ ] size");
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
