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
#include <queue>
#include <string>
#include <unordered_map>
#include <array>
#include <vector>
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// World extent and resolutions. These scale together: a bigger island
// gets proportionally more height samples, mask texels, mesh quads and
// grass, so sculpting and painting keep the same detail per world unit
// at any size.
static float TER_HALF = 24.0f;       // the ground is 2*TER_HALF on a side
static int   GRID_N   = 256;         // terrain quads per side
static int   HN       = 257;         // height samples per side
static int   MASK_N   = 512;         // splat mask resolution
static int   GRASS_N  = 320;         // grass instances per side

// resolution tiers picked from the map's world size
// Extra resolution on top of what the world size asks for. The grid is
// uniform, so this cannot be spent only where it is needed -- there is no
// refining one hillside -- but a road cut across a cliff is exactly where
// too few cells shows, and this is the lever that fixes it.
static float gDetailMult = 1.0f;

static void resolutions_for(float half, int* grid, int* hn, int* mask,
                            int* grass)
{
    // Everything scales PROPORTIONALLY with the world so a texel, a quad
    // and a blade cover the same ground at any map size. Stepped tiers
    // used to fall behind -- a mask texel went from 0.09 units at 48u to
    // 0.19 at 192u -- which is what made painted edges and the ground AO
    // coarsen every time the map grew. Caps keep the biggest maps sane.
    const float mult = half / 24.0f;        // 1 at the original 48u map
    auto snap = [](float v, int step, int lo, int hi) {
        int n = (int)(v / step + 0.5f) * step;
        return SDL_clamp(n, lo, hi);
    };
    const float dm = SDL_clamp(gDetailMult, 0.5f, 4.0f);
    *grid  = snap(256.0f * mult * dm, 64, 128, 2048);
    *hn    = *grid + 1;
    *mask  = snap(512.0f * mult * dm, 128, 256, 4096);
    *grass = snap(320.0f * mult, 32, 320, 800);   // blades stay per-area
}

// AO map resolution for a world size: the ground-AO pass renders blade
// footprints top-down and reads them back through mipmaps, so its texel
// density has to track the world too -- otherwise a bigger map spreads
// the same 1024 texels thinner and the AO radius (a mip level) covers
// more ground, which reads as blurrier and darker contact shadows.
static int ao_res_for(float half)
{
    int n = (int)(1024.0f * (half / 24.0f) / 256.0f + 0.5f) * 256;
    return SDL_clamp(n, 1024, 4096);
}

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
uniform sampler2D uAOMap;   // live top-down blade render (white = open)
uniform mat4 uLightMvp;
uniform int uShadowsOn;
uniform float uGrassAO;    // ground contact-darkening strength
uniform float uGrassAORad; // sampling mip level = occlusion radius
uniform float uShadowStr;  // sun shadow intensity

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

    // live AO: the blades are rendered top-down into uAOMap each frame,
    // so this darkening sits exactly where blades stand; the mip level
    // sets how far the contact shadow spreads
    float open = textureLod(uAOMap, maskUv, uGrassAORad).r;
    col *= 1.0 - uGrassAO * (1.0 - open) * (1.0 - cliffM);

    vec3 L = normalize(vec3(0.35, 0.8, -0.45));
    float diff = max(dot(normalize(vNormal), L), 0.0);
    diff *= mix(1.0, shadow_factor(vWorld), uShadowStr);
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
uniform sampler2D uShadowMap;
uniform mat4 uLightMvp;
uniform int uShadowsOn;
uniform float uShadowDark;
uniform float uHalf;
uniform float uTime;
uniform float uDensity;
uniform float uSwayAmp;   // 0 in the AO pass: occlusion stays at roots
uniform int uFlatten;     // AO pass: lay the blade flat as a footprint
out float vShadow;
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
                  * aBlade.y * aBlade.y * uSwayAmp;
    p.xz += vec2(0.85, 0.53) * sway;

    // AO pass: a vertical card has no area from above -- lay the blade
    // down as a small crossed footprint at its root instead
    if (uFlatten == 1) {
        vec2 perp = vec2(-dir.y, dir.x);
        float fs = wid * 3.0 * show;
        p = vec3(xz.x, ground, xz.y);
        p.xz += dir * aBlade.x * fs + perp * (aBlade.y - 0.5) * 2.0 * fs;
    }

    // one shadow probe per blade at its root: the blade inherits exactly
    // the shadow the ground under it has, so canopy patterns flow across
    // the grass without speckle
    vShadow = 1.0;
    if (uShadowsOn == 1) {
        vec4 lc = uLightMvp * vec4(xz.x, ground, xz.y, 1.0);
        vec3 sp = lc.xyz / lc.w * 0.5 + 0.5;
        if (sp.x > 0.0 && sp.x < 1.0 && sp.y > 0.0 && sp.y < 1.0 &&
            sp.z < 1.0) {
            float d = texture(uShadowMap, sp.xy).r;
            if (sp.z - 0.003 > d)
                vShadow = uShadowDark;
        }
    }
    vV = aBlade.y;
    vWorldXz = xz;
    vSeed = fract(aInst.w * 11.13);
    gl_Position = uMvp * vec4(p, 1.0);
}
)";

static const char* GRASS_FS = R"(#version 330 core
in float vShadow;
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
    vec3 col = mix(root, tip, vV * vV) * vShadow;
    fragColor = vec4(col, 1.0);
}
)";

static const char* PROP_VS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNorm;
layout(location = 2) in vec2 aUv;
layout(location = 3) in vec3 aVCol;
uniform mat4 uMvp;
uniform mat4 uModel;
out vec3 vNorm;
out vec2 vUv;
out float vLocalY;
out vec3 vWorld;
out vec3 vVCol;
void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vNorm = mat3(uModel) * aNorm;
    vUv = aUv;
    vLocalY = aPos.y;
    vWorld = world.xyz;
    vVCol = aVCol;
    gl_Position = uMvp * world;
}
)";

static const char* PROP_FS = R"(#version 330 core
in vec3 vNorm;
in vec2 vUv;
in float vLocalY;
in vec3 vWorld;
in vec3 vVCol;
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
uniform float uShadowStr;  // sun shadow intensity

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
    // albedo textures are authored final -- material Kd only drives the
    // gradient tint for mask textures
    vec3 col = uGrayMask == 1 ? grad * (0.55 + 0.9 * t.r) : t.rgb;
    col *= vVCol;   // vertex tint (bamboo segments etc.), white when absent
    vec3 L = normalize(vec3(0.35, 0.8, -0.45));
    vec3 n = normalize(vNorm);
    float sf = mix(1.0, shadow_factor(vWorld), uShadowStr);
    if (uGrayMask == 1) {
        // foliage: soft wrap lighting and mostly shadow-immune, so
        // canopies read as toon masses instead of speckled black --
        // they still CAST onto the ground
        float wrap = clamp(dot(n, L) * 0.55 + 0.45, 0.0, 1.0);
        col *= (0.42 + 0.62 * wrap) * mix(0.60, 1.0, sf);
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

// Outer extension: terrain generated BEYOND the map bounds, continuing
// each border height outward with fractal noise and sinking into the
// sea. A heightmap cannot be pulled sideways, so growing new topology
// past the rim is how an island gains an organic coast without losing
// any of the sculpted interior.
static const char* EXT_VS = R"(#version 330 core
layout(location = 0) in vec2 aBorder;   // point on the map rim
layout(location = 1) in vec2 aDir;      // outward normal at that point
layout(location = 2) in float aT;       // 0 at the rim .. 1 outermost
uniform mat4  uMvp;
uniform sampler2D uHeight;
uniform float uHalf;
uniform float uDist;      // how far out the land reaches
uniform float uNoise;     // coastline raggedness
uniform float uSea;       // waterline
uniform float uDrop;      // depth below the waterline at the outer edge
out vec3 vWorld;
out vec3 vNormal;

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
float fbm2(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * vnoise(p); p *= 2.03; a *= 0.5; }
    return v;
}

// world position + height for a (border, t) sample
vec3 ext_point(vec2 border, vec2 dir, float t) {
    // ragged reach: some stretches run out further than others
    float reach = uDist * (1.0 + (fbm2(border * 0.10) - 0.5) * 1.4 * uNoise);
    vec2 xz = border + dir * t * reach;
    vec2 uv = (border + vec2(uHalf)) / (2.0 * uHalf);
    float edgeH = texture(uHeight, uv).r;
    // fall from the rim height into the sea, with terrain detail on top
    float s = t * t * (3.0 - 2.0 * t);
    float h = mix(edgeH, uSea - uDrop, s);
    h += (fbm2(xz * 0.16) - 0.5) * 2.2 * uNoise * (1.0 - t) * (t + 0.15);
    return vec3(xz.x, h, xz.y);
}

void main() {
    vec3 p = ext_point(aBorder, aDir, aT);
    // normal from neighbouring samples of the same generator
    vec2 tang = vec2(-aDir.y, aDir.x);
    vec3 pu = ext_point(aBorder + tang * 0.35, aDir, aT);
    vec3 pv = ext_point(aBorder, aDir, min(aT + 0.02, 1.0));
    vNormal = normalize(cross(pv - p, pu - p));
    if (vNormal.y < 0.0) vNormal = -vNormal;
    vWorld = p;
    gl_Position = uMvp * vec4(p, 1.0);
}
)";

// island skirt: the terrain rim extruded down into a rocky underside
static const char* SKIRT_VS = R"(#version 330 core
layout(location = 0) in vec3 aData;   // x, z, t (0 rim, 1 bottom, 2 center)
uniform mat4 uMvp;
uniform sampler2D uHeight;
uniform float uHalf;
uniform float uDepth;
uniform float uFrill;   // lateral noise on the underside silhouette
uniform float uBulge;   // pushes the underside outward past the rim
uniform vec2  uCenter;  // the island's centre: bulge and the bottom cap
                        // pivot here, not on the world origin
out vec3 vWorld;
out vec3 vNormal;
out float vT;

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

void main() {
    vec2 xz = aData.xy;
    float t = aData.z;
    vec3 pos;
    if (t < 0.5) {
        vec2 uv = (xz + vec2(uHalf)) / (2.0 * uHalf);
        pos = vec3(xz.x, texture(uHeight, uv).r, xz.y);
    } else if (t < 1.5) {
        float n = vnoise(xz * 0.35);
        // frill: scalloped lobes pushing the underside in and out
        float f1 = vnoise(xz * 0.45 + 11.0) - 0.5;
        float f2 = vnoise(xz * 1.3 + 7.0) - 0.5;
        float taper = mix(0.82, 1.30, uBulge) + (f1 * 0.5 + f2 * 0.2) * uFrill;
        vec2 d = xz - uCenter;
        pos = vec3(uCenter.x + d.x * taper,
                   -uDepth * (0.55 + 0.5 * n + f2 * 0.5 * uFrill),
                   uCenter.y + d.y * taper);
    } else {
        pos = vec3(uCenter.x, -uDepth * 1.25, uCenter.y);
    }
    vWorld = pos;
    vNormal = normalize(vec3(xz.x - uCenter.x, uDepth * 0.02 + 6.0,
                             xz.y - uCenter.y));
    if (t > 1.5)
        vNormal = vec3(0.0, -1.0, 0.0);
    vT = min(t, 1.5);
    gl_Position = uMvp * vec4(pos, 1.0);
}
)";

static const char* SKIRT_FS = R"(#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in float vT;
out vec4 fragColor;
uniform sampler2D uCliffTex;

void main() {
    // triplanar rock over the underside, darkening toward the bottom
    vec3 n = normalize(vNormal);
    vec3 cx = texture(uCliffTex, vWorld.zy * 0.10).rgb;
    vec3 cz = texture(uCliffTex, vWorld.xy * 0.10).rgb;
    float wx = abs(n.x) / max(abs(n.x) + abs(n.z), 1e-4);
    vec3 col = mix(cz, cx, wx);
    vec3 L = normalize(vec3(0.35, 0.8, -0.45));
    float diff = max(dot(n, L), 0.0);
    col *= 0.62 + 0.38 * diff;
    col *= mix(1.0, 0.45, vT / 1.5);
    fragColor = vec4(col, 1.0);
}
)";

// scale reference: a Link-sized figure. He stands ~1.1 units tall in the
// client and the client scales editor islands 2x, so he is 0.55 units
// here -- drop him anywhere to judge island proportions.
static const char* DUMMY_VS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aCol;
uniform mat4 uMvp;
uniform vec3 uPos;
out vec3 vCol;
out vec3 vNrm;
void main() {
    vCol = aCol;
    vNrm = normalize(aPos - vec3(0.0, aPos.y, 0.0) + vec3(0.0, 0.35, 0.0));
    gl_Position = uMvp * vec4(aPos + uPos, 1.0);
}
)";

static const char* DUMMY_FS = R"(#version 330 core
in vec3 vCol;
in vec3 vNrm;
out vec4 fragColor;
void main() {
    vec3 L = normalize(vec3(0.35, 0.8, -0.45));
    float d = 0.72 + 0.38 * max(dot(normalize(vNrm), L), 0.0);
    fragColor = vec4(vCol * d, 1.0);
}
)";

// world waterline: a simple ocean plane marking sea level
static const char* WATER_VS = R"(#version 330 core
const vec2 verts[6] = vec2[6](vec2(-1,-1), vec2(1,-1), vec2(-1,1),
                              vec2(1,-1), vec2(1,1), vec2(-1,1));
uniform mat4 uMvp;
uniform float uLevel;
uniform float uExtent;
out vec2 vXz;
void main() {
    vec2 p = verts[gl_VertexID] * uExtent;
    vXz = p;
    gl_Position = uMvp * vec4(p.x, uLevel, p.y, 1.0);
}
)";

static const char* WATER_FS = R"(#version 330 core
in vec2 vXz;
out vec4 fragColor;
uniform vec4  uBrush;    // xz, radius, active
uniform vec3  uBrushCol;
uniform float uHalf;     // the quadrant's extent
uniform float uEdge;     // draw its boundary
void main() {
    // basic sea blue, slightly lighter toward the horizon distance
    float d = length(vXz);
    vec3 col = mix(vec3(0.10, 0.38, 0.72), vec3(0.30, 0.55, 0.85),
                   smoothstep(40.0, 160.0, d));
    // The canvas edge. Props can only be placed inside the quadrant, but
    // the sea is drawn far past it, so without this the boundary is
    // invisible and clicks outside look like they simply do nothing.
    if (uEdge > 0.5) {
        vec2 q = abs(vXz);
        float b = max(q.x, q.y);
        float line = (1.0 - smoothstep(uHalf - 0.35, uHalf, b)) *
                     smoothstep(uHalf - 0.9, uHalf - 0.4, b);
        col = mix(col, vec3(0.95, 0.95, 0.75), line * 0.85);
        if (q.x < uHalf && q.y < uHalf)
            col = mix(col, vec3(0.16, 0.44, 0.74), 0.25);
    }
    // and the brush cursor, which otherwise only ever drew on terrain
    if (uBrush.w > 0.5) {
        float bd = length(vXz - uBrush.xy);
        float ring = smoothstep(uBrush.z * 0.92, uBrush.z * 0.97, bd) *
                     (1.0 - smoothstep(uBrush.z * 1.03, uBrush.z * 1.08, bd));
        col = mix(col, uBrushCol, ring * 0.9);
    }
    fragColor = vec4(col, 1.0);
}
)";

// depth-only passes for the shadow map
static const char* DEPTH_FS = R"(#version 330 core
void main() {}
)";

// blades drawn black into the live ground-AO map
static const char* AO_FS = R"(#version 330 core
out vec4 fragColor;
void main() { fragColor = vec4(0.0, 0.0, 0.0, 1.0); }
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
static float gWaterline = -3.0f;   // sea level, world units
// A quadrant can hold nothing but props -- a level built entirely out of
// imported geometry wants no ground plane under it at all, and a flat
// plane at sea level is not "nothing", it is a floor in the way.
static bool gShowGround = false;
// How far the baked collision surface sits above the stamped vertices. A
// cell takes the highest vertex in it, which on a coarse face is below the
// surface between those vertices -- so without a lift you sink into the
// model you can see.
static float gFootLift = 0.15f;

// Where a prop sits at a spot. With no terrain this is the sea, not the
// bottom of a heightmap that was sunk out of the way -- placing read the
// heightmap directly and buried everything a hundred units under water.
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
                 BRUSH_EXPAND, BRUSH_CONTRACT,
                 BRUSH_SHAPE, BRUSH_TAPER,
                 BRUSH_DIRT, BRUSH_DIRT2, BRUSH_ERASEDIRT,
                 BRUSH_GRASS, BRUSH_KILLGRASS,
                 BRUSH_ROAD };   // stroke-level: handled outside apply_brush

static const float kBrushColors[15][3] = {
    { 1.0f, 0.85f, 0.3f },   // raise: yellow
    { 0.4f, 0.8f, 1.0f },    // smooth: blue
    { 0.9f, 0.5f, 0.9f },    // flatten: purple
    { 1.0f, 0.55f, 0.2f },   // sharpen edges: orange
    { 0.3f, 0.9f, 0.8f },    // terrace: teal
    { 0.6f, 1.0f, 0.6f },    // expand land: light green
    { 0.3f, 0.5f, 1.0f },    // contract land: deep blue
    { 1.0f, 0.4f, 0.6f },    // shape paint: pink
    { 0.55f, 0.85f, 1.0f },  // taper to sea: pale blue
    { 0.72f, 0.5f, 0.28f },  // path dirt: brown
    { 0.85f, 0.75f, 0.5f },  // soft dirt: sand
    { 0.35f, 0.8f, 0.35f },  // grass ground (erases dirt): deep green
    { 0.5f, 1.0f, 0.4f },    // grass blades: bright green
    { 0.9f, 0.35f, 0.3f },   // remove grass: red
    { 0.95f, 0.9f, 0.6f },   // path: sand
};
static const char* kBrushNames[15] = { "Sculpt", "Smooth", "Flatten",
                                       "Sharpen", "Terrace",
                                       "Expand", "Contract",
                                       "Shape Paint", "Taper to Sea",
                                       "Path Dirt", "Soft Dirt", "Grass Ground",
                                       "Grass Blades", "Remove Grass", "Path" };

// terrace step height (world units), set from the panel
static float gTerraceStep = 2.0f;

// stroke-level undo/redo: whole-state snapshots (~1 MB each, capped)
// A snapshot carries the map's dimensions as well as its contents, so
// operations that change the world size -- growing the canvas, resizing
// the map -- are undoable like any other stroke.
struct Snapshot {
    std::vector<float> h;
    std::vector<Uint8> m, m2, k;
    std::vector<PropInst> p;
    std::vector<float> shore, gen;   // live shoreline / generator bases
    float half = 0.0f, water = 0.0f;
    int hn = 0, mask = 0, grid = 0, grass = 0;
};
static std::vector<Snapshot> gUndoStack, gRedoStack;

static std::vector<float> gShoreBase;   // heights before any shoreline
static std::vector<float> gGenBase;     // heights before the generator
static std::vector<Uint8> gGenMask, gGenMask2, gGenKill;   // and its paint
static std::vector<PropInst> gGenProps;                    // and its props
static void mark_all_dirty();
static float cpu_vnoise(float x, float y);
static void push_undo();
static bool gMapResized = false;   // a load changed the world size

// Resize the world: resample every layer into the new resolution so the
// island scales with the map instead of being lost. Called from the Map
// tab and when a map with a different size is loaded.
// Empty slate for a quadrant with nothing in it yet: flat ground at sea
// level, no paint, no props, so a new island starts clean.
static void new_map()
{
    std::fill(gHeights.begin(), gHeights.end(), 0.0f);
    std::fill(gMask.begin(), gMask.end(), (Uint8)0);
    std::fill(gMask2.begin(), gMask2.end(), (Uint8)0);
    std::fill(gKill.begin(), gKill.end(), (Uint8)255);
    gProps.clear();
    gShoreBase.clear();
    gUndoStack.clear();
    gRedoStack.clear();
    mark_all_dirty();
    SDL_Log("new island");
}

static void resize_map(float newHalf)
{
    push_undo();   // snapshots carry the world size, so this is undoable
    int nGrid, nHN, nMask, nGrass;
    resolutions_for(newHalf, &nGrid, &nHN, &nMask, &nGrass);

    auto resample_f = [](const std::vector<float>& src, int sn, int dn) {
        std::vector<float> dst((size_t)dn * dn, 0.0f);
        for (int j = 0; j < dn; j++)
            for (int i = 0; i < dn; i++) {
                float u = (float)i / (dn - 1) * (sn - 1);
                float v = (float)j / (dn - 1) * (sn - 1);
                int i0 = SDL_clamp((int)u, 0, sn - 2);
                int j0 = SDL_clamp((int)v, 0, sn - 2);
                float fu = u - i0, fv = v - j0;
                float a = src[(size_t)j0 * sn + i0];
                float b = src[(size_t)j0 * sn + i0 + 1];
                float c = src[(size_t)(j0 + 1) * sn + i0];
                float d = src[(size_t)(j0 + 1) * sn + i0 + 1];
                dst[(size_t)j * dn + i] =
                    (a * (1 - fu) + b * fu) * (1 - fv) +
                    (c * (1 - fu) + d * fu) * fv;
            }
        return dst;
    };
    // bilinear, not nearest: the masks hold continuous coverage values,
    // and point-sampling them stair-steps every painted edge -- visibly,
    // and worse each time the map is resized
    auto resample_u8 = [](const std::vector<Uint8>& src, int sn, int dn) {
        std::vector<Uint8> dst((size_t)dn * dn, 0);
        for (int j = 0; j < dn; j++)
            for (int i = 0; i < dn; i++) {
                float u = ((i + 0.5f) / dn) * sn - 0.5f;
                float v = ((j + 0.5f) / dn) * sn - 0.5f;
                int i0 = SDL_clamp((int)floorf(u), 0, sn - 1);
                int j0 = SDL_clamp((int)floorf(v), 0, sn - 1);
                int i1 = SDL_min(i0 + 1, sn - 1), j1 = SDL_min(j0 + 1, sn - 1);
                float fu = SDL_clamp(u - i0, 0.0f, 1.0f);
                float fv = SDL_clamp(v - j0, 0.0f, 1.0f);
                float a = src[(size_t)j0 * sn + i0], b = src[(size_t)j0 * sn + i1];
                float c = src[(size_t)j1 * sn + i0], d = src[(size_t)j1 * sn + i1];
                float m = (a * (1 - fu) + b * fu) * (1 - fv) +
                          (c * (1 - fu) + d * fu) * fv;
                dst[(size_t)j * dn + i] = (Uint8)SDL_clamp(m + 0.5f, 0.0f, 255.0f);
            }
        return dst;
    };

    // heights carry their world scale: a map twice as wide wants hills
    // twice as tall to keep the same slopes
    const float hscale = newHalf / TER_HALF;
    std::vector<float> nh = resample_f(gHeights, HN, nHN);
    for (float& v : nh) v *= hscale;
    gHeights.swap(nh);
    if (!gShoreBase.empty()) {
        std::vector<float> nb = resample_f(gShoreBase, HN, nHN);
        for (float& v : nb) v *= hscale;
        gShoreBase.swap(nb);
    }
    if (!gGenBase.empty()) {
        std::vector<float> ng = resample_f(gGenBase, HN, nHN);
        for (float& v : ng) v *= hscale;
        gGenBase.swap(ng);
    }
    std::vector<Uint8> m1 = resample_u8(gMask, MASK_N, nMask);
    std::vector<Uint8> m2 = resample_u8(gMask2, MASK_N, nMask);
    std::vector<Uint8> mk = resample_u8(gKill, MASK_N, nMask);
    gMask.swap(m1); gMask2.swap(m2); gKill.swap(mk);
    // the generator's "paint before I ran" snapshot has to be resampled
    // too: left at the old resolution it stops matching, the restore is
    // skipped, and generated trails can never be painted back out
    if (gGenMask.size() == (size_t)MASK_N * MASK_N) {
        std::vector<Uint8> g1 = resample_u8(gGenMask, MASK_N, nMask);
        std::vector<Uint8> g2 = resample_u8(gGenMask2, MASK_N, nMask);
        std::vector<Uint8> g3 = resample_u8(gGenKill, MASK_N, nMask);
        gGenMask.swap(g1); gGenMask2.swap(g2); gGenKill.swap(g3);
    }

    // props sit in world units, so they move out with the island
    for (PropInst& pi : gProps) {
        pi.x *= hscale; pi.y *= hscale; pi.z *= hscale;
        pi.scale *= hscale;
    }
    gWaterline *= hscale;

    TER_HALF = newHalf;
    HN = nHN; MASK_N = nMask; GRID_N = nGrid; GRASS_N = nGrass;
    mark_all_dirty();
    SDL_Log("map resized to %.0f units: grid %d, heights %d, masks %d",
            newHalf * 2.0f, GRID_N, HN, MASK_N);
}

// Grow the map outward WITHOUT rescaling anything already sculpted: the
// island keeps its world position and size, and the new ring of ground
// continues each border height and tapers it down under the sea.
// A heightmap cannot be pushed sideways past its own bounds -- sculpt
// into the rim and the mesh is just sliced off flat -- so widening the
// canvas is the only way land keeps growing outward. Resolutions come
// from the same tier table as everything else, so the added ground has
// the same polygon density as the old.
static void grow_canvas(float newHalf, float seaLevel)
{
    if (newHalf <= TER_HALF + 0.01f)
        return;
    push_undo();   // one Ctrl+Z takes the map back to its old bounds
    const float oldHalf = TER_HALF;
    const float margin = newHalf - oldHalf;
    int nGrid, nHN, nMask, nGrass;
    resolutions_for(newHalf, &nGrid, &nHN, &nMask, &nGrass);

    // bilinear sample of an old-resolution field at a world position,
    // clamped to the border so outside lookups give the rim height
    auto sampleOld = [&](const std::vector<float>& src, float x, float z) {
        float fx = SDL_clamp((x + oldHalf) / (2.0f * oldHalf), 0.0f, 1.0f) *
                   (HN - 1);
        float fz = SDL_clamp((z + oldHalf) / (2.0f * oldHalf), 0.0f, 1.0f) *
                   (HN - 1);
        int i0 = SDL_clamp((int)fx, 0, HN - 2), j0 = SDL_clamp((int)fz, 0, HN - 2);
        float fu = fx - i0, fv = fz - j0;
        float a = src[(size_t)j0 * HN + i0], b = src[(size_t)j0 * HN + i0 + 1];
        float c = src[(size_t)(j0 + 1) * HN + i0];
        float d = src[(size_t)(j0 + 1) * HN + i0 + 1];
        return (a * (1 - fu) + b * fu) * (1 - fv) +
               (c * (1 - fu) + d * fu) * fv;
    };
    auto growField = [&](const std::vector<float>& src) {
        std::vector<float> dst((size_t)nHN * nHN, 0.0f);
        const float cell = 2.0f * newHalf / (nHN - 1);
        for (int j = 0; j < nHN; j++)
            for (int i = 0; i < nHN; i++) {
                float x = -newHalf + cell * i, z = -newHalf + cell * j;
                float edgeH = sampleOld(src, x, z);
                float out = SDL_max(fabsf(x) - oldHalf, fabsf(z) - oldHalf);
                if (out <= 0.0f) {
                    dst[(size_t)j * nHN + i] = edgeH;
                    continue;
                }
                // ragged reach so the new coast is not a square ring
                float wob = (cpu_vnoise(x * 0.11f, z * 0.11f) - 0.5f) * 0.55f +
                            (cpu_vnoise(x * 0.29f, z * 0.29f) - 0.5f) * 0.25f;
                float t = SDL_clamp(out / SDL_max(0.5f, margin * (1.0f + wob)),
                                    0.0f, 1.0f);
                float s = t * t * (3.0f - 2.0f * t);
                float h = edgeH * (1.0f - s) + (seaLevel - 2.0f) * s;
                h += (cpu_vnoise(x * 0.18f, z * 0.18f) - 0.5f) * 1.6f *
                     (1.0f - s);
                dst[(size_t)j * nHN + i] = SDL_min(h, edgeH);
            }
        return dst;
    };
    auto growMask = [&](const std::vector<Uint8>& src, Uint8 outside) {
        std::vector<Uint8> dst((size_t)nMask * nMask, outside);
        const float cell = 2.0f * newHalf / nMask;
        for (int j = 0; j < nMask; j++)
            for (int i = 0; i < nMask; i++) {
                float x = -newHalf + (i + 0.5f) * cell;
                float z = -newHalf + (j + 0.5f) * cell;
                if (fabsf(x) > oldHalf || fabsf(z) > oldHalf)
                    continue;
                // bilinear: point-sampling here stair-steps painted edges
                float u = (x + oldHalf) / (2.0f * oldHalf) * MASK_N - 0.5f;
                float v = (z + oldHalf) / (2.0f * oldHalf) * MASK_N - 0.5f;
                int i0 = SDL_clamp((int)floorf(u), 0, MASK_N - 1);
                int j0 = SDL_clamp((int)floorf(v), 0, MASK_N - 1);
                int i1 = SDL_min(i0 + 1, MASK_N - 1);
                int j1 = SDL_min(j0 + 1, MASK_N - 1);
                float fu = SDL_clamp(u - i0, 0.0f, 1.0f);
                float fv = SDL_clamp(v - j0, 0.0f, 1.0f);
                float a = src[(size_t)j0 * MASK_N + i0];
                float b = src[(size_t)j0 * MASK_N + i1];
                float c = src[(size_t)j1 * MASK_N + i0];
                float d = src[(size_t)j1 * MASK_N + i1];
                float m = (a * (1 - fu) + b * fu) * (1 - fv) +
                          (c * (1 - fu) + d * fu) * fv;
                dst[(size_t)j * nMask + i] =
                    (Uint8)SDL_clamp(m + 0.5f, 0.0f, 255.0f);
            }
        return dst;
    };

    std::vector<float> nh = growField(gHeights);
    std::vector<float> nShore, nGen;
    if (gShoreBase.size() == gHeights.size()) nShore = growField(gShoreBase);
    if (gGenBase.size() == gHeights.size())   nGen   = growField(gGenBase);
    std::vector<Uint8> m1 = growMask(gMask, 0);
    std::vector<Uint8> m2 = growMask(gMask2, 0);
    std::vector<Uint8> mk = growMask(gKill, 255);   // no blades on new ground
    std::vector<Uint8> g1, g2, g3;
    const bool haveGenPaint = gGenMask.size() == (size_t)MASK_N * MASK_N;
    if (haveGenPaint) {   // keep the generator's snapshot in step
        g1 = growMask(gGenMask, 0);
        g2 = growMask(gGenMask2, 0);
        g3 = growMask(gGenKill, 255);
    }

    gHeights.swap(nh);
    gShoreBase.swap(nShore);
    gGenBase.swap(nGen);
    gMask.swap(m1); gMask2.swap(m2); gKill.swap(mk);
    if (haveGenPaint) {
        gGenMask.swap(g1); gGenMask2.swap(g2); gGenKill.swap(g3);
    }
    // props and the waterline are in world units and do not move

    TER_HALF = newHalf;
    HN = nHN; MASK_N = nMask; GRID_N = nGrid; GRASS_N = nGrass;
    mark_all_dirty();
    SDL_Log("map grew to %.0f units (sculpt untouched): grid %d, heights %d",
            newHalf * 2.0f, GRID_N, HN);
}

// Rebuild the map at the resolution the current detail setting asks for,
// keeping its world size, its shape and everything painted on it. Heights
// and masks are resampled bilinearly, so raising detail interpolates what
// is there and lowering it averages -- neither invents nor discards work.
static void apply_detail()
{
    int nGrid, nHN, nMask, nGrass;
    resolutions_for(TER_HALF, &nGrid, &nHN, &nMask, &nGrass);
    if (nHN == HN && nMask == MASK_N && nGrid == GRID_N)
        return;
    push_undo();

    auto rs_f = [](const std::vector<float>& src, int sn, int dn) {
        std::vector<float> dst((size_t)dn * dn, 0.0f);
        for (int j = 0; j < dn; j++)
            for (int i = 0; i < dn; i++) {
                float u = (float)i / (dn - 1) * (sn - 1);
                float v = (float)j / (dn - 1) * (sn - 1);
                int i0 = SDL_clamp((int)u, 0, sn - 2);
                int j0 = SDL_clamp((int)v, 0, sn - 2);
                float fu = u - i0, fv = v - j0;
                float a = src[(size_t)j0 * sn + i0];
                float b = src[(size_t)j0 * sn + i0 + 1];
                float c = src[(size_t)(j0 + 1) * sn + i0];
                float d = src[(size_t)(j0 + 1) * sn + i0 + 1];
                dst[(size_t)j * dn + i] = (a * (1 - fu) + b * fu) * (1 - fv) +
                                          (c * (1 - fu) + d * fu) * fv;
            }
        return dst;
    };
    auto rs_u8 = [](const std::vector<Uint8>& src, int sn, int dn) {
        std::vector<Uint8> dst((size_t)dn * dn, 0);
        for (int j = 0; j < dn; j++)
            for (int i = 0; i < dn; i++) {
                float u = ((i + 0.5f) / dn) * sn - 0.5f;
                float v = ((j + 0.5f) / dn) * sn - 0.5f;
                int i0 = SDL_clamp((int)floorf(u), 0, sn - 1);
                int j0 = SDL_clamp((int)floorf(v), 0, sn - 1);
                int i1 = SDL_min(i0 + 1, sn - 1), j1 = SDL_min(j0 + 1, sn - 1);
                float fu = SDL_clamp(u - i0, 0.0f, 1.0f);
                float fv = SDL_clamp(v - j0, 0.0f, 1.0f);
                float a = src[(size_t)j0 * sn + i0], b = src[(size_t)j0 * sn + i1];
                float c = src[(size_t)j1 * sn + i0], d = src[(size_t)j1 * sn + i1];
                float m = (a * (1 - fu) + b * fu) * (1 - fv) +
                          (c * (1 - fu) + d * fu) * fv;
                dst[(size_t)j * dn + i] = (Uint8)SDL_clamp(m + 0.5f, 0.0f, 255.0f);
            }
        return dst;
    };

    std::vector<float> nh = rs_f(gHeights, HN, nHN);
    gHeights.swap(nh);
    if (gShoreBase.size() == (size_t)HN * HN) {
        std::vector<float> t = rs_f(gShoreBase, HN, nHN);
        gShoreBase.swap(t);
    }
    if (gGenBase.size() == (size_t)HN * HN) {
        std::vector<float> t = rs_f(gGenBase, HN, nHN);
        gGenBase.swap(t);
    }
    std::vector<Uint8> m1 = rs_u8(gMask, MASK_N, nMask);
    std::vector<Uint8> m2 = rs_u8(gMask2, MASK_N, nMask);
    std::vector<Uint8> mk = rs_u8(gKill, MASK_N, nMask);
    if (gGenMask.size() == (size_t)MASK_N * MASK_N) {
        std::vector<Uint8> g1 = rs_u8(gGenMask, MASK_N, nMask);
        std::vector<Uint8> g2 = rs_u8(gGenMask2, MASK_N, nMask);
        std::vector<Uint8> g3 = rs_u8(gGenKill, MASK_N, nMask);
        gGenMask.swap(g1); gGenMask2.swap(g2); gGenKill.swap(g3);
    }
    gMask.swap(m1); gMask2.swap(m2); gKill.swap(mk);

    HN = nHN; MASK_N = nMask; GRID_N = nGrid; GRASS_N = nGrass;
    gMapResized = true;
    mark_all_dirty();
    SDL_Log("detail x%.2f: grid %d, heights %d, masks %d",
            gDetailMult, GRID_N, HN, MASK_N);
}

// is there land near the map boundary that wants somewhere to go?
static bool land_at_edge(float seaLevel)
{
    int band = SDL_max(2, HN / 20);
    for (int j = 0; j < HN; j++) {
        bool rowEdge = (j < band || j >= HN - band);
        for (int i = 0; i < HN; i++) {
            if (!rowEdge && i >= band && i < HN - band) {
                i = HN - band - 1;   // skip the interior
                continue;
            }
            if (gHeights[(size_t)j * HN + i] > seaLevel + 0.3f)
                return true;
        }
    }
    return false;
}

static void mark_all_dirty()
{
    gHeightsDirty = gMaskDirty = gMask2Dirty = gKillDirty = true;
}

static Snapshot capture_state()
{
    Snapshot s;
    s.h = gHeights; s.m = gMask; s.m2 = gMask2; s.k = gKill; s.p = gProps;
    s.shore = gShoreBase; s.gen = gGenBase;
    s.half = TER_HALF; s.water = gWaterline;
    s.hn = HN; s.mask = MASK_N; s.grid = GRID_N; s.grass = GRASS_N;
    return s;
}

// restore a snapshot; flags a resize when the world size changed so the
// caller can rebuild the size-bound buffers
static void restore_state(const Snapshot& s)
{
    bool resized = (s.hn != HN || s.mask != MASK_N || s.grid != GRID_N ||
                    s.grass != GRASS_N || s.half != TER_HALF);
    gHeights = s.h; gMask = s.m; gMask2 = s.m2; gKill = s.k; gProps = s.p;
    gShoreBase = s.shore; gGenBase = s.gen;
    TER_HALF = s.half; gWaterline = s.water;
    HN = s.hn; MASK_N = s.mask; GRID_N = s.grid; GRASS_N = s.grass;
    if (resized)
        gMapResized = true;
    mark_all_dirty();
}

static void push_undo()
{
    gUndoStack.push_back(capture_state());
    if (gUndoStack.size() > 32)
        gUndoStack.erase(gUndoStack.begin());
    gRedoStack.clear();
}

static void do_undo()
{
    if (gUndoStack.empty())
        return;
    gRedoStack.push_back(capture_state());
    Snapshot s = gUndoStack.back();   // by value: restoring resizes globals
    gUndoStack.pop_back();
    restore_state(s);
}

static void do_redo()
{
    if (gRedoStack.empty())
        return;
    gUndoStack.push_back(capture_state());
    Snapshot s = gRedoStack.back();
    gRedoStack.pop_back();
    restore_state(s);
}

static float cpu_vnoise(float x, float y);

// Sink the map's rim below the waterline so the island ends in sea
// instead of a square plateau: a noisy radial falloff from the edge.
// (A square of land casts a square shadow and gets no foam ring.)
// Live, non-destructive shoreline: the sliders always rebuild the rim
// from a pristine snapshot, so dragging them never compounds.
struct ShoreParams {
    float width = 0.06f;    // rim fraction that becomes shore
    float drop = 1.0f;      // how far under the waterline the border sits
    float frill = 0.35f;    // raggedness of the coastline
    float smoothing = 0.5f; // 0 = abrupt cliff, 1 = gradual beach
    bool  radial = false;   // round the footprint instead of even inset
    bool  on = false;
};
static ShoreParams gShore;

static void apply_shoreline(float seaLevel)
{
    if (gShoreBase.size() != gHeights.size())
        return;
    if (!gShore.on) {
        gHeights = gShoreBase;
        gHeightsDirty = true;
        return;
    }
    const float cell = 2.0f * TER_HALF / (HN - 1);
    const float band = SDL_max(0.005f, gShore.width);
    // low smoothing keeps the drop near the border (cliffy), high
    // smoothing spreads it inland (beachy)
    const float curve = 0.25f + gShore.smoothing * 2.75f;
    for (int j = 0; j < HN; j++)
        for (int i = 0; i < HN; i++) {
            float x = -TER_HALF + cell * i;
            float z = -TER_HALF + cell * j;
            // rim distance: 0 at the boundary, 1 inland
            float rim;
            if (gShore.radial) {
                rim = 1.0f - sqrtf(x * x + z * z) / TER_HALF;
            } else {
                rim = SDL_min(1.0f - fabsf(x) / TER_HALF,
                              1.0f - fabsf(z) / TER_HALF);
            }
            // wobble scaled by the band so a narrow shore stays narrow
            float wob = ((cpu_vnoise(x * 0.09f, z * 0.09f) - 0.5f) * 1.6f +
                         (cpu_vnoise(x * 0.31f, z * 0.31f) - 0.5f) * 0.7f) *
                        gShore.frill;
            float t = SDL_clamp((rim + wob * band) / band, 0.0f, 1.0f);
            t = powf(t * t * (3.0f - 2.0f * t), curve);
            float base = gShoreBase[j * HN + i];
            gHeights[j * HN + i] =
                SDL_min(base, base * t + (seaLevel - gShore.drop) * (1.0f - t));
        }
    gHeightsDirty = true;
}

// ------------------------------------------------------- island generator
// A whole island from a seed and a set of shape controls. Everything is
// evaluated in normalized map coordinates (-1..1), so a generated island
// keeps its proportions at any map size, and every control is continuous
// -- dragging a slider re-runs the generator from a pristine snapshot the
// way the shoreline does, so nothing compounds and nothing is destroyed
// until you bake it.
struct GenParams {
    int   seed = 19430;
    float size = 1.10f;      // island radius, fraction of the map half
    float coast = 0.44f;     // width of the falloff into the sea
    float lumps = 0.11f;     // how much the coastline radius wanders
    float warp = 0.25f;      // domain warp: bends the whole shape organic
    float height = 28.3f;    // peak land height in world units
    float rough = 0.65f;     // terrain noise vs a smooth dome
    int   detail = 3;        // fbm octaves
    float fscale = 1.48f;    // feature frequency
    float ridge = 0.30f;      // billowy blobs .. sharp mountain spines
    int   peaks = 2;         // seeded summits on top of the noise
    float peakH = 1.17f;
    float peakSpread = 0.48f; // how far the summits sit from the centre
    float plateau = 0.46f;   // soft ceiling: mesa tops
    float terr = 3.2f;       // terrace step height, 0 = off
    float beach = 0.59f;     // widens the gentle land near the water
    float drop = 12.0f;       // sea floor depth outside the island
    // ---- level layout: the parts that make an island playable rather
    // ---- than just scenery
    int   flats = 2;         // flat clearings to build on
    float flatSize = 0.17f;  // clearing radius, fraction of island radius
    float flatFlat = 0.12f;   // how completely a clearing is levelled
    bool  paths = true;      // trails linking the clearings and the shore
    float pathWidth = 1.8f;  // world units
    float pathWander = 0.0f;
    float pathCut = 0.38f;   // how firmly a trail levels the ground it crosses
    float pathGrade = 0.56f; // steepest climb a trail will accept
    float pathBank = 3.0f;   // slope of the cut/fill banks beside the tread
    float pathCling = 0.28f;  // 0 = cut across the terrain, 1 = wind around it
    bool  pathPaint = true;  // lay dirt (and clear grass) along the trails
    int   pathLayer = 1;     // 0 = path dirt (brown), 1 = soft dirt (sand)
    bool  spiralRoad = true; // a road wrapping the terraces to the summit
    float spiralTurn = 0.18f; // turns of the hill gained per terrace
    float spiralInset = 1.26f;// how far back from each drop the road sits
    float spiralHug = 0.06f;  // 0 = cut straight through risers, 1 = follow
                             // the terrace treads exactly
    float pathFollow = 0.0f;    // 0 = hold one graded line, 1 = drape over
                                // the ground as it is
    bool  landmarks = true;     // caves, cairns and groves worth walking to
    float landmarkDens = 1.0f;  // how much of it gets placed
    bool  roadSupport = false;  // build the hillside out to carry the road
    float roadSupportW = 6.0f;  // how far the buttress reaches
    bool  shorePath = false;  // run one trail down to a landing beach
    bool  summitPath = true; // and one up to the island's high point
    bool  add = false;       // layer over the existing sculpt
    bool  on = false;
};
static GenParams gGen;

// value noise in [0,1] over an arbitrary seed offset
static float gen_fbm(float x, float y, int oct, float ridge)
{
    float v = 0.0f, a = 0.5f, f = 1.0f, norm = 0.0f;
    for (int i = 0; i < oct; i++) {
        float n = cpu_vnoise(x * f, y * f);
        if (ridge > 0.0f) {
            // ridged: fold the noise about its midpoint so the creases
            // become sharp crests instead of round lumps
            float r = 1.0f - fabsf(n * 2.0f - 1.0f);
            n = n * (1.0f - ridge) + r * r * ridge;
        }
        v += a * n;
        norm += a;
        f *= 2.03f;
        a *= 0.5f;
    }
    return v / SDL_max(0.0001f, norm);
}

// the terrain detail alone, centred on zero -- shared by the generator and
// the Shape Paint brush so painted deformations match generated ones
static float gen_detail(float u, float v)
{
    float so = gGen.seed * 0.7913f;
    float wx = cpu_vnoise(u * 1.7f + so, v * 1.7f - so) - 0.5f;
    float wy = cpu_vnoise(u * 1.7f + so + 37.0f, v * 1.7f - so + 11.0f) - 0.5f;
    float pu = u + wx * gGen.warp, pv = v + wy * gGen.warp;
    return gen_fbm(pu * gGen.fscale + so, pv * gGen.fscale - so,
                   SDL_clamp(gGen.detail, 1, 8), gGen.ridge) - 0.5f;
}

// Land height and coverage at a normalized position. mask is 1 inland,
// 0 out at sea, so callers can either replace the terrain (blend down to
// the sea floor) or add the land on top of what is already there.
static float gen_land_raw(float u, float v, float* maskOut)
{
    const GenParams& g = gGen;
    float so = g.seed * 0.7913f;
    // warp first: the coastline and the terrain bend together, which is
    // what stops generated islands from looking like circles with noise
    float wx = cpu_vnoise(u * 1.7f + so, v * 1.7f - so) - 0.5f;
    float wy = cpu_vnoise(u * 1.7f + so + 37.0f, v * 1.7f - so + 11.0f) - 0.5f;
    float pu = u + wx * g.warp, pv = v + wy * g.warp;

    float r = sqrtf(pu * pu + pv * pv);
    // radius varies with angle; sampling the noise ON the unit circle
    // keeps it seamless where the angle wraps
    float ang = atan2f(pv, pu);
    float lump = (cpu_vnoise(cosf(ang) * 2.2f + so * 3.0f,
                             sinf(ang) * 2.2f - so * 3.0f) - 0.5f) * 2.0f +
                 (cpu_vnoise(cosf(ang) * 5.1f - so,
                             sinf(ang) * 5.1f + so) - 0.5f);
    float R = SDL_max(0.05f, g.size * (1.0f + lump * g.lumps));
    float coastW = SDL_max(0.02f, g.coast * g.size);
    float t = SDL_clamp((R - r) / coastW, 0.0f, 1.0f);
    float mask = t * t * (3.0f - 2.0f * t);

    float h = gen_fbm(pu * g.fscale + so, pv * g.fscale - so,
                      SDL_clamp(g.detail, 1, 8), g.ridge);
    h = 0.45f + (h - 0.5f) * g.rough * 1.8f;

    for (int p = 0; p < g.peaks; p++) {
        float a = cpu_vnoise(g.seed * 0.031f + p * 4.7f, p * 2.3f) * 6.2831853f;
        float d = 0.25f + cpu_vnoise(p * 9.1f, g.seed * 0.017f) * 0.75f;
        float px = cosf(a) * d * g.peakSpread * g.size;
        float pz = sinf(a) * d * g.peakSpread * g.size;
        float dx = (pu - px) / (0.42f * g.size), dz = (pv - pz) / (0.42f * g.size);
        float dd = dx * dx + dz * dz;
        h += g.peakH * expf(-dd * 1.6f);
    }
    if (h < 0.0f) h = 0.0f;

    if (g.plateau > 0.0f) {
        // soft ceiling rather than a clamp, so the mesa top still reads
        // as terrain instead of a machined flat
        float cap = 1.0f - g.plateau * 0.55f;
        if (h > cap)
            h = cap + (h - cap) * (1.0f - g.plateau * 0.9f);
    }
    if (g.beach > 0.0f) {
        // stretch the shallow band: land near the water rises slower
        float b = SDL_clamp(mask / SDL_max(0.05f, g.beach * 0.7f), 0.0f, 1.0f);
        h *= b * b * (3.0f - 2.0f * b);
    }
    if (maskOut)
        *maskOut = mask;
    return h * g.height * mask;
}

// Clearings: seeded flat shelves that give an island somewhere to put a
// village, a shrine or a fight. Positions are stable for a seed.
struct GenFlat { float u, v, r; };
// cached for the duration of one regenerate: gen_land runs per cell, and
// rebuilding this list a million times would dominate the whole pass
static std::vector<GenFlat> gFlatCache;
static void gen_flats(std::vector<GenFlat>& out)
{
    out.clear();
    const GenParams& g = gGen;
    for (int i = 0; i < g.flats; i++) {
        float a = cpu_vnoise(g.seed * 0.013f + i * 3.1f, i * 5.7f) * 6.2831853f;
        float d = 0.18f + cpu_vnoise(i * 2.9f, g.seed * 0.023f) * 0.62f;
        float rr = 0.7f + cpu_vnoise(i * 7.3f, g.seed * 0.007f) * 0.6f;
        out.push_back({ cosf(a) * d * g.size * 0.8f,
                        sinf(a) * d * g.size * 0.8f,
                        SDL_max(0.02f, g.flatSize * g.size * rr) });
    }
}

// The land with the level layout applied. Clearings are levelled first,
// then terracing quantizes in WORLD height -- doing it before the coast
// mask (as this used to) meant multiplying the steps by a continuous
// falloff, which smoothed them straight back out. That is why terraces
// never read.
static float gen_land(float u, float v, float* maskOut)
{
    const GenParams& g = gGen;
    float mask = 0.0f;
    float land = gen_land_raw(u, v, &mask);

    if (g.flats > 0 && g.flatFlat > 0.0f) {
        for (const GenFlat& f : gFlatCache) {
            float dx = u - f.u, dz = v - f.v;
            float d = sqrtf(dx * dx + dz * dz) / f.r;
            if (d >= 1.0f)
                continue;
            float w = 1.0f - SDL_clamp((d - 0.55f) / 0.45f, 0.0f, 1.0f);
            w = w * w * (3.0f - 2.0f * w);
            float target = gen_land_raw(f.u, f.v, nullptr);
            land += (target - land) * w * g.flatFlat;
        }
    }
    if (g.terr > 0.01f && land > g.terr * 0.9f) {
        // leave the beach band continuous so the shoreline is not a wall
        land = floorf(land / g.terr) * g.terr;
    }
    if (maskOut)
        *maskOut = mask;
    return land;
}

// ---- road carving
// Cut and paint one polyline as a road: a flat tread at a grade-limited
// height, constant-slope banks either side, and dirt laid along it. Used
// both by the generator and by the Path brush, so a hand-drawn road is
// built exactly like a generated one.
static void road_carve(const std::vector<float>& rx,
                       const std::vector<float>& rz, float wander)
{
    const GenParams& g = gGen;
    if (rx.size() < 2 || gHeights.empty())
        return;
    const float cell = 2.0f * TER_HALF / (HN - 1);
    const float mcell = 2.0f * TER_HALF / MASK_N;
    const float halfW = SDL_max(0.4f, g.pathWidth * 0.5f);
    const float so = g.seed * 0.7913f;
        if (rx.size() < 2)
            return;
        // resample the route evenly so carving is uniform along it
        std::vector<float> cum(rx.size(), 0.0f);
        for (size_t k = 1; k < rx.size(); k++) {
            float sx = rx[k] - rx[k - 1], sz = rz[k] - rz[k - 1];
            cum[k] = cum[k - 1] + sqrtf(sx * sx + sz * sz);
        }
        float total = cum.back();
        if (total < cell * 2.0f)
            return;
        int steps = SDL_max(8, (int)(total / (cell * 0.5f)));
        std::vector<float> ptx(steps + 1), ptz(steps + 1), pth(steps + 1);
        size_t seg = 0;
        for (int i = 0; i <= steps; i++) {
            float t = (float)i / steps;
            float want = t * total;
            while (seg + 2 < rx.size() && cum[seg + 1] < want)
                seg++;
            float segLen = SDL_max(0.0001f, cum[seg + 1] - cum[seg]);
            float f = SDL_clamp((want - cum[seg]) / segLen, 0.0f, 1.0f);
            float x = rx[seg] + (rx[seg + 1] - rx[seg]) * f;
            float z = rz[seg] + (rz[seg + 1] - rz[seg]) * f;
            // wander across the route, tapering to nothing at both ends
            // so it still meets the clearings it was routed between
            float tx = rx[seg + 1] - rx[seg], tz = rz[seg + 1] - rz[seg];
            float tl = SDL_max(0.0001f, sqrtf(tx * tx + tz * tz));
            float env = sinf(t * 3.14159f);
            float w = ((cpu_vnoise(t * 3.1f + so, so * 2.0f) - 0.5f) * 2.0f +
                       (cpu_vnoise(t * 7.7f - so, so) - 0.5f)) *
                      wander * g.pathWidth * 1.2f * env;
            x += (-tz / tl) * w;
            z += (tx / tl) * w;
            ptx[i] = SDL_clamp(x, -TER_HALF + cell, TER_HALF - cell);
            ptz[i] = SDL_clamp(z, -TER_HALF + cell, TER_HALF - cell);
            pth[i] = height_at(ptx[i], ptz[i]);
        }
        // Grade-limit the profile instead of smoothing it. Averaging a
        // staircase gives a straight ramp, which then has to cut deep
        // into every tread and pile fill over every riser -- that is what
        // tore the scar across the terraces. Limiting slope from both
        // ends keeps flat ground exactly flat and only lowers the trail
        // where the climb is genuinely too steep, so a terraced hillside
        // gets its treads left alone and a notch cut through each riser.
        // Taking the minimum also means the trail never floats above the
        // land: it cuts in, the way a worn path does.
        {
            // The ground as it actually is, lightly eased -- this is the
            // line a path would take if it simply followed the hillside.
            // Smoothed hard: sampled raw, this is a staircase on terraced
            // ground, and a road that follows a staircase carves each disc
            // at a different height -- which shows up as vertical slivers
            // torn down the cliffs. What a path should follow is the shape
            // underneath the steps, not the steps.
            std::vector<float> ground = pth;
            for (int pass = 0; pass < 14; pass++) {
                std::vector<float> t = ground;
                for (int i = 1; i < steps; i++)
                    ground[i] = (t[i - 1] + 2.0f * t[i] + t[i + 1]) * 0.25f;
            }
            float ds = total / steps;
            float maxRise = SDL_max(0.02f, g.pathGrade) * ds;
            for (int pass = 0; pass < 2; pass++) {
                for (int i = 1; i <= steps; i++)
                    pth[i] = SDL_min(pth[i], pth[i - 1] + maxRise);
                for (int i = steps - 1; i >= 0; i--)
                    pth[i] = SDL_min(pth[i], pth[i + 1] + maxRise);
            }
            // round the corners where cuts begin and end
            for (int pass = 0; pass < 2; pass++) {
                std::vector<float> t = pth;
                for (int i = 1; i < steps; i++)
                    pth[i] = (t[i - 1] + 2.0f * t[i] + t[i + 1]) * 0.25f;
            }
            // Grade limiting takes the minimum along the WHOLE path, so a
            // single steep stretch drags everything after it down to one
            // graded line -- on a hillside that reads as a flat pad cut
            // into the slope rather than a path running across it. Follow
            // Ground blends back toward the hillside itself, so the road
            // rolls with the land and only departs from it where the
            // climb genuinely has to be eased.
            float follow = SDL_clamp(g.pathFollow, 0.0f, 1.0f);
            if (follow > 0.0f) {
                for (int i = 0; i <= steps; i++)
                    pth[i] += (ground[i] - pth[i]) * follow;
                // and take any residual steps back out of the result
                for (int pass = 0; pass < 3; pass++) {
                    std::vector<float> t = pth;
                    for (int i = 1; i < steps; i++)
                        pth[i] = (t[i - 1] + 2.0f * t[i] + t[i + 1]) * 0.25f;
                }
            }
        }

        for (int i = 0; i <= steps; i++) {
            float cx = ptx[i], cz = ptz[i], ch = pth[i];
            // Cut a bench, not a dip. A path reads as a path because it
            // has a FLAT tread of constant width, with the ground cut
            // away on the uphill side and filled on the downhill one --
            // easing the terrain toward the trail height across the whole
            // corridor just makes a soft trough that still rolls with the
            // land. Tread is levelled outright; the shoulder blends back
            // into whatever the hillside was doing.
            // Banks are a constant SLOPE, not a constant width: a bank of
            // fixed width has to go vertical when the cut is deep, which
            // is why a deep crossing came out as a landslide scar. Away
            // from the tread the ground may only differ from the trail by
            // what the bank angle allows at that distance, so a shallow
            // cut barely touches the hillside and a deep one opens a wide
            // batter -- the shape earthworks actually make.
            const float bank = SDL_max(0.15f, g.pathBank);
            // How far the earthworks reach is set by how deep this bit of
            // trail actually cuts: probe the ground just off each side of
            // the tread and give the batter only the width it needs to
            // meet grade. A fixed reach flattens the whole hillside even
            // where the trail is already sitting on the surface.
            int pi = SDL_max(0, i - 1), ni2 = SDL_min(steps, i + 1);
            float tgx = ptx[ni2] - ptx[pi], tgz = ptz[ni2] - ptz[pi];
            float tl2 = SDL_max(0.0001f, sqrtf(tgx * tgx + tgz * tgz));
            float nx = -tgz / tl2, nz = tgx / tl2;
            float e1 = height_at(SDL_clamp(cx + nx * halfW, -TER_HALF, TER_HALF),
                                 SDL_clamp(cz + nz * halfW, -TER_HALF, TER_HALF));
            float e2 = height_at(SDL_clamp(cx - nx * halfW, -TER_HALF, TER_HALF),
                                 SDL_clamp(cz - nz * halfW, -TER_HALF, TER_HALF));
            float depth = SDL_max(fabsf(e1 - ch), fabsf(e2 - ch));
            const float reach = halfW + SDL_min(10.0f, depth / bank);
            int i0 = SDL_clamp((int)((cx - reach + TER_HALF) / cell), 0, HN - 1);
            int i1 = SDL_clamp((int)((cx + reach + TER_HALF) / cell) + 1, 0, HN - 1);
            int j0 = SDL_clamp((int)((cz - reach + TER_HALF) / cell), 0, HN - 1);
            int j1 = SDL_clamp((int)((cz + reach + TER_HALF) / cell) + 1, 0, HN - 1);
            for (int j = j0; j <= j1; j++)
                for (int ii = i0; ii <= i1; ii++) {
                    float x = -TER_HALF + ii * cell, z = -TER_HALF + j * cell;
                    float dd = sqrtf((x - cx) * (x - cx) + (z - cz) * (z - cz));
                    if (dd >= reach)
                        continue;
                    float& h = gHeights[(size_t)j * HN + ii];
                    float want;
                    if (dd <= halfW) {
                        want = ch;                       // the tread itself
                    } else {
                        float allow = (dd - halfW) * bank;
                        want = SDL_clamp(h, ch - allow, ch + allow);
                    }
                    h += (want - h) * g.pathCut;
                }
            if (g.roadSupport) {
                // Build the hillside OUT to carry the road. Cutting alone
                // can only take material away, so on a face too steep to
                // hold a tread the road has nothing to sit on -- and two
                // laps of a spiral cannot pass each other at all, since a
                // heightmap keeps one height per spot. Filling underneath
                // grows a buttress out of the slope: the lower lap's fill
                // becomes the shelf the upper one needs, so the laps meet
                // the hill instead of cutting into each other.
                float sup = SDL_max(0.5f, g.roadSupportW);
                int si0 = SDL_clamp((int)((cx - sup + TER_HALF) / cell), 0, HN - 1);
                int si1 = SDL_clamp((int)((cx + sup + TER_HALF) / cell) + 1, 0, HN - 1);
                int sj0 = SDL_clamp((int)((cz - sup + TER_HALF) / cell), 0, HN - 1);
                int sj1 = SDL_clamp((int)((cz + sup + TER_HALF) / cell) + 1, 0, HN - 1);
                for (int j = sj0; j <= sj1; j++)
                    for (int ii = si0; ii <= si1; ii++) {
                        float x = -TER_HALF + ii * cell;
                        float z = -TER_HALF + j * cell;
                        float dd = sqrtf((x - cx) * (x - cx) + (z - cz) * (z - cz));
                        if (dd >= sup)
                            continue;
                        // never lower anything: this pass only adds land
                        float floorH = ch - SDL_max(0.0f, dd - halfW) * bank;
                        float& h = gHeights[(size_t)j * HN + ii];
                        if (h < floorH)
                            h = floorH;
                    }
            }
            if (!g.pathPaint)
                continue;
            // lay the dirt, and clear blades off the tread
            int mi0 = SDL_clamp((int)((cx - halfW + TER_HALF) / mcell), 0, MASK_N - 1);
            int mi1 = SDL_clamp((int)((cx + halfW + TER_HALF) / mcell) + 1, 0, MASK_N - 1);
            int mj0 = SDL_clamp((int)((cz - halfW + TER_HALF) / mcell), 0, MASK_N - 1);
            int mj1 = SDL_clamp((int)((cz + halfW + TER_HALF) / mcell) + 1, 0, MASK_N - 1);
            for (int j = mj0; j <= mj1; j++)
                for (int ii = mi0; ii <= mi1; ii++) {
                    float x = -TER_HALF + (ii + 0.5f) * mcell;
                    float z = -TER_HALF + (j + 0.5f) * mcell;
                    float d = sqrtf((x - cx) * (x - cx) + (z - cz) * (z - cz)) /
                              halfW;
                    if (d >= 1.0f)
                        continue;
                    // A splat mask is indexed by world XZ, so it belongs
                    // to a COLUMN of ground rather than to a surface: on a
                    // steep face one mask cell covers the whole height of
                    // that face and the paint smears down it instead of
                    // lying along the road. Leave steep ground unpainted
                    // -- it also stops the road bleeding over whatever was
                    // already painted either side of the cut.
                    float hc = height_at(x, z);
                    float sx = fabsf(height_at(x + cell, z) - hc);
                    float sz = fabsf(height_at(x, z + cell) - hc);
                    if (SDL_max(sx, sz) / cell > 0.9f)
                        continue;
                    // ragged edge, so the trail is not a clean stripe
                    float e = 1.0f - d + (cpu_vnoise(x * 0.7f, z * 0.7f) - 0.5f) * 0.5f;
                    if (e <= 0.05f)
                        continue;
                    Uint8 v = (Uint8)SDL_clamp(e * 320.0f, 0.0f, 255.0f);
                    Uint8& m = (g.pathLayer == 0 ? gMask : gMask2)
                                  [(size_t)j * MASK_N + ii];
                    if (v > m) m = v;
                    Uint8& k = gKill[(size_t)j * MASK_N + ii];
                    if (v > k) k = v;   // 255 = no blades
                }
        }
    
    gHeightsDirty = gMaskDirty = gMask2Dirty = gKillDirty = true;
}

// ---- trails
// Least-cost route between two points across the heightfield. A straight
// line with noise on it reads as a stripe drawn over the terrain; a real
// trail is the CHEAPEST way across it, so climbing is priced against the
// grade it costs. Below the grade limit slope is nearly free, past it the
// price climbs steeply -- which is what makes routes hug contours and
// switchback up a hillside instead of charging over the top.
// toShore: ignore the target and instead run to whichever piece of
// coastline is cheapest to reach, which is how a trail finds the gentle
// side of an island rather than falling off the steep one
static bool gen_route(float ax, float az, float bx, float bz, float seaLevel,
                      std::vector<float>& outX, std::vector<float>& outZ,
                      bool toShore = false)
{
    const int N = SDL_clamp(HN / 4, 48, 160);
    const float span = 2.0f * TER_HALF;
    auto wpos = [&](int i) { return -TER_HALF + span * i / (N - 1); };
    auto idx = [&](int i, int j) { return (size_t)j * N + i; };
    auto toGrid = [&](float w) {
        return SDL_clamp((int)((w + TER_HALF) / span * (N - 1) + 0.5f),
                         0, N - 1);
    };
    std::vector<float> raw((size_t)N * N);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            raw[idx(i, j)] = height_at(wpos(i), wpos(j));

    // Route over a SMOOTHED copy of the terrain. On terraced ground the
    // real heightfield is flat-or-wall, so a router reading it directly
    // finds every step impassable and runs along the rims forever looking
    // for a way round -- which is exactly what it did. Smoothing turns
    // each step back into the slope it stands for, so the route crosses
    // where the underlying ground is shallowest and the carve cuts the
    // ramp there.
    // Cling is the whole character of a trail. Routing on the RAW
    // terraced field makes every riser a wall, so a route runs along the
    // treads and winds around the hill -- the switchback road look.
    // Routing on a smoothed copy turns each riser back into the slope it
    // stands for, so a route crosses where the ground is shallowest and
    // cuts straight up. Blending between them gives both.
    const float cling = SDL_clamp(gGen.pathCling, 0.0f, 1.0f);
    std::vector<float> h = raw;
    for (int pass = 0; pass < 6; pass++) {
        std::vector<float> t = h;
        for (int j = 1; j < N - 1; j++)
            for (int i = 1; i < N - 1; i++)
                h[idx(i, j)] = (t[idx(i, j)] * 4.0f +
                                t[idx(i - 1, j)] + t[idx(i + 1, j)] +
                                t[idx(i, j - 1)] + t[idx(i, j + 1)]) * 0.125f;
    }
    for (size_t k = 0; k < h.size(); k++)
        h[k] = h[k] + (raw[k] - h[k]) * cling;
    // local ruggedness: how broken the ground is around each node, so
    // trails keep a little clearance from cliff edges instead of riding
    // along the very lip of one
    std::vector<float> rough((size_t)N * N, 0.0f);
    for (int j = 1; j < N - 1; j++)
        for (int i = 1; i < N - 1; i++) {
            float c = raw[idx(i, j)];
            float m = 0.0f;
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++)
                    m = SDL_max(m, fabsf(raw[idx(i + di, j + dj)] - c));
            rough[idx(i, j)] = m;
        }

    const float cell = span / (N - 1);
    const float maxG = SDL_max(0.02f, gGen.pathGrade);
    std::vector<float> dist((size_t)N * N, 1e30f);
    std::vector<int> prev((size_t)N * N, -1);
    int si = toGrid(ax), sj = toGrid(az), ti = toGrid(bx), tj = toGrid(bz);
    // shoreline nodes: land with open water next door
    std::vector<char> isShore((size_t)N * N, 0);
    if (toShore) {
        for (int j = 1; j < N - 1; j++)
            for (int i = 1; i < N - 1; i++) {
                if (raw[idx(i, j)] <= seaLevel + 0.2f)
                    continue;
                if (raw[idx(i - 1, j)] <= seaLevel + 0.2f ||
                    raw[idx(i + 1, j)] <= seaLevel + 0.2f ||
                    raw[idx(i, j - 1)] <= seaLevel + 0.2f ||
                    raw[idx(i, j + 1)] <= seaLevel + 0.2f)
                    isShore[idx(i, j)] = 1;
            }
    }
    typedef std::pair<float, int> QE;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    dist[idx(si, sj)] = 0.0f;
    pq.push({ 0.0f, (int)idx(si, sj) });
    int goal = (int)idx(ti, tj);
    while (!pq.empty()) {
        QE top = pq.top();
        pq.pop();
        int cur = top.second;
        if (top.first > dist[cur] + 0.0001f)
            continue;
        if (toShore) {
            if (isShore[cur]) {   // first shore reached is the cheapest
                goal = cur;
                break;
            }
        } else if (cur == goal) {
            break;
        }
        int ci = cur % N, cj = cur / N;
        for (int dj = -1; dj <= 1; dj++)
            for (int di = -1; di <= 1; di++) {
                if (!di && !dj)
                    continue;
                int ni = ci + di, nj = cj + dj;
                if (ni < 0 || nj < 0 || ni >= N || nj >= N)
                    continue;
                float d = cell * ((di && dj) ? 1.4142f : 1.0f);
                float dh = h[idx(ni, nj)] - h[idx(ci, cj)];
                float grade = fabsf(dh) / d;
                float c = d;
                // Steepness is priced, but the price is CAPPED. Without a
                // cap one steep step costs more than a detour of any
                // length, which is what produced the long sweeping arcs
                // that never got anywhere.
                float over = grade / maxG;
                float pen = 1.0f + over * over * 2.0f;
                // clinging trails refuse the climb almost outright, which
                // is what sends them round the contour instead
                c *= SDL_min(pen, 30.0f + cling * 570.0f);
                // climbing costs by how much height it gains, not just how
                // steeply -- so a route prefers to gain its height once
                // rather than repeatedly rolling over ridges
                if (dh > 0.0f)
                    c += dh * (1.6f - cling * 1.45f);
                // keep off cliff lips -- but a clinging trail is meant to
                // run along them, so this eases off as cling rises
                c += SDL_min(rough[idx(ni, nj)], 6.0f) * cell * 0.9f *
                     (1.0f - cling * 0.85f);
                if (raw[idx(ni, nj)] < seaLevel + 0.2f)
                    c += span * 4.0f;                    // stay on land
                size_t nk = idx(ni, nj);
                if (dist[cur] + c < dist[nk]) {
                    dist[nk] = dist[cur] + c;
                    prev[nk] = cur;
                    pq.push({ dist[nk], (int)nk });
                }
            }
    }
    if (prev[goal] < 0 && goal != (int)idx(si, sj))
        return false;
    std::vector<int> rev;
    for (int c = goal; c >= 0; c = prev[c]) {
        rev.push_back(c);
        if (c == (int)idx(si, sj))
            break;
    }
    if (rev.size() < 2)
        return false;
    outX.clear();
    outZ.clear();
    for (size_t k = rev.size(); k-- > 0;) {
        outX.push_back(wpos(rev[k] % N));
        outZ.push_back(wpos(rev[k] / N));
    }
    // the route came off an 8-direction grid, so it staircases: relax it
    // into a curve the way a walked trail would wear
    for (int pass = 0; pass < 6; pass++)
        for (size_t k = 1; k + 1 < outX.size(); k++) {
            outX[k] = (outX[k - 1] + 2.0f * outX[k] + outX[k + 1]) * 0.25f;
            outZ[k] = (outZ[k - 1] + 2.0f * outZ[k] + outZ[k + 1]) * 0.25f;
        }
    return true;
}

// A path is routed between the clearings, then down to a landing beach.
// Rather than cutting a straight ramp between two heights -- which gouges
// a trench across anything in the way -- it samples the ground it crosses
// and SMOOTHS that profile, so the trail follows the land at a walkable
// grade and only cuts where the ground is genuinely too steep.
static void gen_carve_paths(float seaLevel)
{
    const GenParams& g = gGen;
    if (!g.paths || gHeights.empty())
        return;
    std::vector<GenFlat> flats;
    gen_flats(flats);

    struct Node { float x, z; };
    std::vector<Node> nodes;
    for (const GenFlat& f : flats)
        nodes.push_back({ f.u * TER_HALF, f.v * TER_HALF });
    if (nodes.size() < 2 && !g.shorePath)
        return;
    if (g.summitPath) {
        // A trail that has to reach the top is what produces the spiral:
        // with Cling up, every riser is a wall, so the only way up is to
        // keep going round the hill gaining a terrace at a time.
        const float hcell = 2.0f * TER_HALF / (HN - 1);
        float bestH = -1e9f, bx2 = 0.0f, bz2 = 0.0f;
        for (int j = 0; j < HN; j++)
            for (int i = 0; i < HN; i++) {
                float hh = gHeights[(size_t)j * HN + i];
                if (hh > bestH) {
                    bestH = hh;
                    bx2 = -TER_HALF + i * hcell;
                    bz2 = -TER_HALF + j * hcell;
                }
            }
        if (bestH > seaLevel + 0.5f)
            nodes.insert(nodes.begin(), { bx2, bz2 });
    }
    if (g.shorePath && !nodes.empty()) {
        // let the router choose the landing: the cheapest coastline to
        // walk to, rather than whatever lies straight outward -- which
        // was as likely to be the island's steepest face as its beach
        std::vector<float> sx, sz;
        if (gen_route(nodes.back().x, nodes.back().z, 0.0f, 0.0f, seaLevel,
                      sx, sz, true) && !sx.empty())
            nodes.push_back({ sx.back(), sz.back() });
    }
    if (nodes.size() < 2)
        return;

    const float cell = 2.0f * TER_HALF / (HN - 1);
    const float mcell = 2.0f * TER_HALF / MASK_N;
    const float halfW = SDL_max(0.4f, g.pathWidth * 0.5f);
    const float so = g.seed * 0.7913f;

    auto carve_route = [&](const std::vector<float>& rx,
                           const std::vector<float>& rz) {
        road_carve(rx, rz, g.pathWander);
    };

    if (g.spiralRoad) {
        // A road that wraps the hill cannot be asked for by routing from
        // A to B: a router's whole job is to find the SHORT way, and a
        // terrace road is deliberately the long one. So it is generated
        // directly -- follow the contour of each terrace for part of a
        // turn, step out and down onto the next, and keep going round.
        float bestH = -1e9f, cx0 = 0.0f, cz0 = 0.0f;
        for (int j = 0; j < HN; j++)
            for (int i = 0; i < HN; i++) {
                float hh = gHeights[(size_t)j * HN + i];
                if (hh > bestH) {
                    bestH = hh;
                    cx0 = -TER_HALF + i * cell;
                    cz0 = -TER_HALF + j * cell;
                }
            }
        float stepH = g.terr > 0.05f ? g.terr
                                     : SDL_max(0.5f, g.height / 7.0f);
        // One continuous helix rather than a ring per terrace. Built per
        // level, consecutive rings sit at whatever radius their contour
        // happens to be, so on a steep face they land on top of each
        // other and the carve smears them into the slope -- which is what
        // made the road merge into the cliff instead of wrapping it.
        // Here the road's HEIGHT falls smoothly as it goes round, and the
        // radius is wherever the ground currently stands at that height,
        // so it climbs the hill as one unbroken shelf.
        // Read the contour off a SMOOTHED copy of the terrain. On a
        // terraced hill a riser is near vertical, so the radius at which
        // the ground stands at a given height barely moves through it --
        // the road gets pinned to the rock face and reads as part of the
        // cliff. Smoothed, each riser is the slope it stands for, the
        // radius advances steadily, and the carve notches the road
        // through the lip the way a real hill road crosses a bench.
        const int SN = SDL_clamp(HN / 3, 64, 192);
        std::vector<float> sm((size_t)SN * SN);
        for (int j = 0; j < SN; j++)
            for (int i = 0; i < SN; i++) {
                float x = -TER_HALF + 2.0f * TER_HALF * i / (SN - 1);
                float z = -TER_HALF + 2.0f * TER_HALF * j / (SN - 1);
                sm[(size_t)j * SN + i] = height_at(x, z);
            }
        // How much the terrain is smoothed before the contour is read
        // IS the road's character: unsmoothed, every riser is vertical
        // and the road clings to the treads and the rock face; heavily
        // smoothed, risers become slopes and the road cuts across them.
        const float hug = SDL_clamp(gGen.spiralHug, 0.0f, 1.0f);
        const int passes = (int)((1.0f - hug) * 16.0f + 0.5f);
        for (int pass = 0; pass < passes; pass++) {
            std::vector<float> t = sm;
            for (int j = 1; j < SN - 1; j++)
                for (int i = 1; i < SN - 1; i++)
                    sm[(size_t)j * SN + i] =
                        (t[(size_t)j * SN + i] * 4.0f +
                         t[(size_t)j * SN + i - 1] + t[(size_t)j * SN + i + 1] +
                         t[(size_t)(j - 1) * SN + i] + t[(size_t)(j + 1) * SN + i]) *
                        0.125f;
        }
        auto smooth_at = [&](float x, float z) {
            float u = (x + TER_HALF) / (2.0f * TER_HALF) * (SN - 1);
            float v = (z + TER_HALF) / (2.0f * TER_HALF) * (SN - 1);
            int i0 = SDL_clamp((int)u, 0, SN - 2), j0 = SDL_clamp((int)v, 0, SN - 2);
            float fu = SDL_clamp(u - i0, 0.0f, 1.0f);
            float fv = SDL_clamp(v - j0, 0.0f, 1.0f);
            float a = sm[(size_t)j0 * SN + i0], b = sm[(size_t)j0 * SN + i0 + 1];
            float c = sm[(size_t)(j0 + 1) * SN + i0];
            float d = sm[(size_t)(j0 + 1) * SN + i0 + 1];
            return (a * (1 - fu) + b * fu) * (1 - fv) +
                   (c * (1 - fu) + d * fu) * fv;
        };
        float smTop = -1e9f;
        for (float v2 : sm)
            if (v2 > smTop) smTop = v2;

        std::vector<float> sxs, szs;
        const float turn = SDL_max(0.05f, g.spiralTurn);
        const float dropPerRad = stepH / (turn * 6.2831853f);
        const float dA = 0.05f;
        float ang = 0.0f, lv = smTop - stepH * 0.35f;
        int guard = 0;
        while (lv > seaLevel + stepH * 0.4f && guard++ < 40000) {
            float dx = cosf(ang), dz = sinf(ang);
            float found = 0.0f;
            // coarse enough that a high-resolution map does not make
            // this thousands of samples per degree of arc
            const float rstep = SDL_max(cell * 0.6f, TER_HALF / 400.0f);
            for (float r = 0.0f; r < TER_HALF * 1.6f; r += rstep) {
                float x = cx0 + dx * r, z = cz0 + dz * r;
                if (fabsf(x) > TER_HALF || fabsf(z) > TER_HALF)
                    break;
                if (smooth_at(x, z) < lv)
                    break;              // the ground has fallen below us
                found = r;
            }
            if (found > 0.0f) {
                float rr = SDL_max(0.3f,
                                   found - halfW * 1.3f - g.spiralInset);
                sxs.push_back(SDL_clamp(cx0 + dx * rr, -TER_HALF + cell,
                                        TER_HALF - cell));
                szs.push_back(SDL_clamp(cz0 + dz * rr, -TER_HALF + cell,
                                        TER_HALF - cell));
            }
            ang += dA;
            lv -= dropPerRad * dA;
        }
        carve_route(sxs, szs);
        // and a way down to the water from where the road runs out
        if (g.shorePath && !sxs.empty()) {
            std::vector<float> bx2, bz2;
            if (gen_route(sxs.back(), szs.back(), 0.0f, 0.0f, seaLevel,
                          bx2, bz2, true))
                carve_route(bx2, bz2);
        }
    } else {
        for (size_t n = 0; n + 1 < nodes.size(); n++) {
            float ax = nodes[n].x, az = nodes[n].z;
            float bx = nodes[n + 1].x, bz = nodes[n + 1].z;
            float dx = bx - ax, dz = bz - az;
            if (sqrtf(dx * dx + dz * dz) < cell * 2.0f)
                continue;
            // route it across the terrain; a straight line is only the
            // fallback for when no walkable way exists at all
            std::vector<float> rx, rz;
            if (!gen_route(ax, az, bx, bz, seaLevel, rx, rz)) {
                rx = { ax, bx };
                rz = { az, bz };
            }
            carve_route(rx, rz);
        }
    }
    gHeightsDirty = gMaskDirty = gMask2Dirty = gKillDirty = true;
}

static void gen_landmarks(float seaLevel);   // defined with the props

// Rebuild the terrain from the generator. Non-destructive: always starts
// from the snapshot taken when the generator was switched on.
static void apply_generator(float seaLevel)
{
    if (gGenBase.size() != gHeights.size())
        return;
    // paint is restored too: the generator lays trails into the masks,
    // so regenerating has to start from clean ones or every drag of a
    // slider would stack another set of paths on the last
    auto restore_paint = [&]() {
        if (gGenMask.size() != gMask.size()) {
            // Nothing sane to put back -- rather than silently leaving
            // generated trails painted on forever, re-snapshot here so
            // the next regenerate has a clean base to restore.
            SDL_Log("generator: paint snapshot was stale, re-taken");
            gGenMask = gMask; gGenMask2 = gMask2; gGenKill = gKill;
            return;
        }
        gMask = gGenMask;
        gMask2 = gGenMask2;
        gKill = gGenKill;
        gMaskDirty = gMask2Dirty = gKillDirty = true;
    };
    if (!gGen.on) {
        gHeights = gGenBase;
        restore_paint();
        gHeightsDirty = true;
        return;
    }
    restore_paint();
    gen_flats(gFlatCache);
    const float cell = 2.0f * TER_HALF / (HN - 1);
    for (int j = 0; j < HN; j++)
        for (int i = 0; i < HN; i++) {
            float u = (-TER_HALF + cell * i) / TER_HALF;
            float v = (-TER_HALF + cell * j) / TER_HALF;
            float mask = 0.0f;
            float land = gen_land(u, v, &mask);
            float base = gGenBase[j * HN + i];
            gHeights[j * HN + i] =
                gGen.add ? base + land
                         : land * mask + (seaLevel - gGen.drop) * (1.0f - mask);
        }
    gen_carve_paths(seaLevel);
    gen_landmarks(seaLevel);
    gHeightsDirty = true;
}

// A hand-drawn road. The stroke is collected as a polyline and the whole
// road is rebuilt from a snapshot every frame, so the grade limiting and
// the banks are computed over the WHOLE path rather than compounding
// piece by piece as the cursor moves -- dragging back over a stretch
// re-cuts it rather than digging it deeper.
static std::vector<float> gRoadX, gRoadZ;
static std::vector<float> gRoadBaseH;
static std::vector<Uint8> gRoadBaseM, gRoadBaseM2, gRoadBaseK;

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
    // Whether this part of the model is ground. The footprint takes the
    // highest vertex over each cell, so a tree's canopy wins and you end
    // up standing on the leaves -- the parts that are scenery have to say
    // so.
    bool collide = true;
    unsigned tex = 0;
    bool grayMask = false;
    float kd[3] = { 1, 1, 1 };   // top/main color
    float ka[3] = { 1, 1, 1 };   // bottom color (gradient)
    std::string name;
    std::string texName;
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
    std::vector<float> pts;   // xyz per vertex, kept for footprint baking
    std::vector<int> ptMat;   // which material each of those came from
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
            m.mats.back().name = name;
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
            m.mats.back().texName = name;
        }
    }
    fclose(f);
}

// ---- style presets: per-model, per-role (Trunk / Leaves / Other)
// texture + gradient color overrides, persisted in props/styles.txt so
// they survive pack re-exports
struct StyleOverride {
    std::string tex;             // "-" or empty = keep the MTL texture
    float kd[3] = { 1, 1, 1 };
    float ka[3] = { 1, 1, 1 };
};
static std::unordered_map<std::string, StyleOverride> gStyles; // "id|Role"
static std::vector<std::string> gPropTexFiles;
static const char* kRoleNames[3] = { "Trunk", "Leaves", "Other" };

static int mat_role(const std::string& name)
{
    std::string l;
    for (char c : name)
        l += (char)tolower((unsigned char)c);
    if (l.find("leaf") != std::string::npos ||
        l.find("leaves") != std::string::npos ||
        l.find("foliage") != std::string::npos ||
        l.find("frond") != std::string::npos ||
        l.find("needle") != std::string::npos ||
        l.find("petal") != std::string::npos)
        return 1;
    if (l.find("bark") != std::string::npos ||
        l.find("trunk") != std::string::npos ||
        l.find("stem") != std::string::npos ||
        l.find("branch") != std::string::npos ||
        l.find("wood") != std::string::npos ||
        l.find("root") != std::string::npos)
        return 0;
    return 2;
}

static std::string mesh_id(const PropMesh& m)
{
    return m.category + "/" + m.label;
}

static void apply_styles(PropMesh& m)
{
    for (PropMaterial& mat : m.mats) {
        auto it = gStyles.find(mesh_id(m) + "|" +
                               kRoleNames[mat_role(mat.name)]);
        if (it == gStyles.end())
            continue;
        const StyleOverride& s = it->second;
        memcpy(mat.kd, s.kd, sizeof mat.kd);
        memcpy(mat.ka, s.ka, sizeof mat.ka);
        if (!s.tex.empty() && s.tex != "-") {
            PropTex pt = prop_texture("textures/" + s.tex);
            if (pt.tex) {
                mat.tex = pt.tex;
                mat.grayMask = pt.gray;
                mat.texName = "textures/" + s.tex;
            }
        }
    }
}

static void save_styles()
{
    FILE* f = fopen((gPropsDir + "/styles.txt").c_str(), "wb");
    if (!f)
        return;
    for (const auto& kv : gStyles) {
        const StyleOverride& s = kv.second;
        fprintf(f, "style %s %s %f %f %f %f %f %f\n", kv.first.c_str(),
                s.tex.empty() ? "-" : s.tex.c_str(),
                s.kd[0], s.kd[1], s.kd[2], s.ka[0], s.ka[1], s.ka[2]);
    }
    fclose(f);
}

static void load_styles()
{
    FILE* f = fopen((gPropsDir + "/styles.txt").c_str(), "rb");
    if (!f)
        return;
    char line[512], key[256], tex[256];
    while (fgets(line, sizeof line, f)) {
        StyleOverride s;
        if (sscanf(line, "style %255s %255s %f %f %f %f %f %f", key, tex,
                   &s.kd[0], &s.kd[1], &s.kd[2],
                   &s.ka[0], &s.ka[1], &s.ka[2]) == 8) {
            s.tex = tex;
            gStyles[key] = s;
        }
    }
    fclose(f);
    SDL_Log("loaded %d style presets", (int)gStyles.size());
}

static bool import_glb_into(PropMesh& m, const std::string& path);

static bool load_prop(int idx)
{
    PropMesh& m = gPropMeshes[idx];
    if (m.loaded || m.failed)
        return m.loaded;
    if (m.objPath.size() > 4 &&
        m.objPath.compare(m.objPath.size() - 4, 4, ".glb") == 0) {
        if (import_glb_into(m, m.objPath))
            return true;
        m.failed = true;
        return false;
    }
    FILE* f = fopen(m.objPath.c_str(), "rb");
    if (!f) {
        m.failed = true;
        return false;
    }
    std::vector<float> vs, ns, ts, vcs;
    std::vector<float> data;   // interleaved pos3 norm3 uv2 col3
    std::unordered_map<std::string, int> matIndex;
    int curMat = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        float a, b, c, r, g, bl;
        char name[256];
        int vn = sscanf(line, "v %f %f %f %f %f %f", &a, &b, &c, &r, &g, &bl);
        if (line[0] == 'v' && line[1] == ' ' && vn >= 3) {
            vs.insert(vs.end(), { a, b, c });
            if (vn == 6)
                vcs.insert(vcs.end(), { r, g, bl });
            else
                vcs.insert(vcs.end(), { 1, 1, 1 });
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
                m.subs.push_back({ (int)(data.size() / 11), 0, curMat });
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
                    data.insert(data.end(),
                                { vcs[i * 3], vcs[i * 3 + 1], vcs[i * 3 + 2] });
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
    apply_styles(m);
    glGenVertexArrays(1, &m.vao);
    glGenBuffers(1, &m.vbo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float),
                          (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float),
                          (void*)(8 * sizeof(float)));
    glBindVertexArray(0);
    m.loaded = true;
    return true;
}

// ---- landmarks
// Places worth walking to, built from props where the terrain cannot do
// the job: a heightmap holds one height per spot, so a cave mouth or an
// overhang has to be geometry standing on the ground rather than shaped
// out of it. The alcove is carved, the roof is propped.
static int find_prop(const char* category, const char* nameHas, int nth)
{
    int seen = 0;
    for (int i = 0; i < (int)gPropMeshes.size(); i++) {
        const PropMesh& m = gPropMeshes[i];
        if (m.category != category)
            continue;
        if (nameHas && m.label.find(nameHas) == std::string::npos)
            continue;
        if (seen++ == nth)
            return i;
    }
    // fall back to anything in the category
    for (int i = 0; i < (int)gPropMeshes.size(); i++)
        if (gPropMeshes[i].category == category)
            return i;
    return -1;
}

// wantR is the radius the thing should END UP, in world units. The pack's
// meshes vary enormously -- a cliff is tens of units across, a bush under
// one -- so a fixed scale gives either pebbles or mountains.
static void place_prop(int mesh, float x, float z, float yaw, float wantR,
                       float sinkFrac)
{
    if (mesh < 0 || !load_prop(mesh))
        return;
    const PropMesh& m = gPropMeshes[mesh];
    const float sc = wantR / SDL_max(0.05f, m.boundR);
    gProps.push_back({ mesh, x,
                       height_at(x, z) - m.boundH * sc * sinkFrac, z,
                       yaw, sc });
}

static void gen_landmarks(float seaLevel)
{
    const GenParams& g = gGen;
    if (!g.landmarks || gPropMeshes.empty())
        return;
    const float cell = 2.0f * TER_HALF / (HN - 1);
    auto rnd = [&](int k) {
        unsigned n = (unsigned)(g.seed * 2654435761u + k * 2246822519u);
        n ^= n >> 13; n *= 2654435761u; n ^= n >> 16;
        return (n & 0xFFFFFF) / 16777216.0f;
    };

    // --- the summit: a cairn of boulders, visible from the water
    float bestH = -1e9f, sx = 0.0f, sz = 0.0f;
    for (int j = 0; j < HN; j++)
        for (int i = 0; i < HN; i++) {
            float hh = gHeights[(size_t)j * HN + i];
            if (hh > bestH) {
                bestH = hh;
                sx = -TER_HALF + i * cell;
                sz = -TER_HALF + j * cell;
            }
        }
    if (bestH > seaLevel + 1.0f) {
        const int boulder = find_prop("Rocks", "BoulderClassic", 0);
        for (int i = 0; i < 5; i++) {
            const float a = 6.2831853f * i / 5 + rnd(i) * 0.8f;
            const float r = 1.6f + rnd(i + 40) * 1.4f;
            place_prop(boulder, sx + cosf(a) * r, sz + sinf(a) * r,
                       rnd(i + 80) * 6.28f, 0.9f + rnd(i + 120) * 0.7f, 0.2f);
        }
    }

    // --- a cave: find a steep face partway up and cut into it, then
    // stand cliff props over the mouth so it reads as a roof rather than
    // a dent. This is the bit a heightmap cannot do on its own.
    {
        float bx = 0.0f, bz = 0.0f, bestSlope = 0.0f;
        for (int j = 2; j < HN - 2; j += 3)
            for (int i = 2; i < HN - 2; i += 3) {
                const float h = gHeights[(size_t)j * HN + i];
                if (h < seaLevel + 2.0f || h > bestH * 0.75f)
                    continue;      // want the flanks, not the peak or shore
                const float dx = gHeights[(size_t)j * HN + i + 2] -
                                 gHeights[(size_t)j * HN + i - 2];
                const float dz = gHeights[(size_t)(j + 2) * HN + i] -
                                 gHeights[(size_t)(j - 2) * HN + i];
                const float sl = sqrtf(dx * dx + dz * dz);
                if (sl > bestSlope) {
                    bestSlope = sl;
                    bx = -TER_HALF + i * cell;
                    bz = -TER_HALF + j * cell;
                }
            }
        if (bestSlope > 0.5f) {
            const float mouthR = 3.4f;
            const float floorH = height_at(bx, bz) - 0.4f;
            // carve the alcove: a flat floor cut back into the slope
            const int i0 = SDL_clamp((int)((bx - mouthR * 2.0f + TER_HALF) / cell), 0, HN - 1);
            const int i1 = SDL_clamp((int)((bx + mouthR * 2.0f + TER_HALF) / cell) + 1, 0, HN - 1);
            const int j0 = SDL_clamp((int)((bz - mouthR * 2.0f + TER_HALF) / cell), 0, HN - 1);
            const int j1 = SDL_clamp((int)((bz + mouthR * 2.0f + TER_HALF) / cell) + 1, 0, HN - 1);
            for (int j = j0; j <= j1; j++)
                for (int i = i0; i <= i1; i++) {
                    const float x = -TER_HALF + i * cell;
                    const float z = -TER_HALF + j * cell;
                    const float d = sqrtf((x - bx) * (x - bx) +
                                          (z - bz) * (z - bz)) / mouthR;
                    if (d >= 1.0f)
                        continue;
                    float w = 1.0f - SDL_clamp((d - 0.4f) / 0.6f, 0.0f, 1.0f);
                    w = w * w * (3.0f - 2.0f * w);
                    float& h = gHeights[(size_t)j * HN + i];
                    if (h > floorH)
                        h += (floorH - h) * w;
                }
            gHeightsDirty = true;
            // the roof and jambs, in props
            const int cliff = find_prop("Rocks", "CliffClassic", 0);
            const int cliff2 = find_prop("Rocks", "CliffClassic", 2);
            const int boulder = find_prop("Rocks", "Boulder", 1);
            for (int i = 0; i < 7; i++) {
                const float a = 3.14159f * (0.15f + 0.7f * i / 6.0f);
                const float r = mouthR * 1.15f;
                place_prop(i & 1 ? cliff : cliff2, bx + cosf(a) * r,
                           bz + sinf(a) * r, a + 1.57f,
                           mouthR * (0.55f + rnd(i + 200) * 0.25f), 0.35f);
            }
            for (int i = 0; i < 4; i++)
                place_prop(boulder, bx + (rnd(i + 300) - 0.5f) * mouthR * 1.6f,
                           bz + (rnd(i + 340) - 0.5f) * mouthR * 1.6f,
                           rnd(i + 380) * 6.28f,
                           0.7f + rnd(i + 420) * 0.6f, 0.15f);
        }
    }

    // --- groves on the clearings, so a flat shelf reads as somewhere
    for (size_t f = 0; f < gFlatCache.size(); f++) {
        const GenFlat& fl = gFlatCache[f];
        const float fx = fl.u * TER_HALF, fz = fl.v * TER_HALF;
        const int n = (int)(6 * g.landmarkDens) + 2;
        const int tree = find_prop("Trees", nullptr, (int)f);
        const int bush = find_prop("Foliage", "Bush", 0);
        for (int i = 0; i < n; i++) {
            const float a = rnd((int)f * 100 + i) * 6.2831853f;
            const float r = fl.r * TER_HALF * (0.35f + rnd(i + 500) * 0.8f);
            const float x = SDL_clamp(fx + cosf(a) * r, -TER_HALF + 2.0f,
                                      TER_HALF - 2.0f);
            const float z = SDL_clamp(fz + sinf(a) * r, -TER_HALF + 2.0f,
                                      TER_HALF - 2.0f);
            if (height_at(x, z) < seaLevel + 0.8f)
                continue;
            place_prop(i % 3 == 0 ? bush : tree, x, z,
                       rnd(i + 600) * 6.28f,
                       (i % 3 == 0 ? 0.5f : 1.1f) + rnd(i + 640) * 0.5f,
                       0.06f);
        }
    }
}

// ---- glb import
// Brings an outside model in as a prop. The pack's props are OBJ, but a
// glb is what everything else exports, so this reads one into the same
// interleaved layout (pos3 norm3 uv2 col3) the OBJ loader produces and
// registers it under an "Imported" category. Colour comes from each
// material's base-colour factor; textures are not unpacked yet, so a
// textured model arrives in its flat material colours.
// A glb carries its images inside it (or beside it), so a material's base
// colour map can be unpacked and bound like any pack texture -- the props
// then show up with their real materials rather than flat colours, and
// the same Trunk/Leaves style overrides apply to them.
static unsigned glb_texture(const cgltf_image* img, const std::string& dir,
                            bool* grayOut)
{
    if (!img)
        return 0;
    std::vector<unsigned char> file;
    const unsigned char* bytes = nullptr;
    size_t len = 0;
    if (img->buffer_view && img->buffer_view->buffer &&
        img->buffer_view->buffer->data) {
        bytes = (const unsigned char*)img->buffer_view->buffer->data +
                img->buffer_view->offset;
        len = img->buffer_view->size;
    } else if (img->uri && strncmp(img->uri, "data:", 5) != 0) {
        std::string up = dir + "/" + img->uri;
        FILE* tf = fopen(up.c_str(), "rb");
        if (!tf)
            return 0;
        fseek(tf, 0, SEEK_END);
        file.resize((size_t)ftell(tf));
        fseek(tf, 0, SEEK_SET);
        if (fread(file.data(), 1, file.size(), tf) != file.size()) {
            fclose(tf);
            return 0;
        }
        fclose(tf);
        bytes = file.data();
        len = file.size();
    }
    if (!bytes || !len)
        return 0;
    int w = 0, h = 0, ch = 0;
    stbi_uc* px = stbi_load_from_memory(bytes, (int)len, &w, &h, &ch, 4);
    if (!px)
        return 0;
    // Is it a mask rather than a picture? A grayscale image is the pack's
    // way of saying "tint me with the material gradient", and the shader
    // needs telling -- otherwise it shows the mask itself, which is why
    // the leaves came out grey while their Top and Bottom were right.
    if (grayOut) {
        bool gray = true;
        const int step = SDL_max(1, (w * h) / 4096);
        for (int i = 0; i < w * h && gray; i += step) {
            const stbi_uc* q = px + (size_t)i * 4;
            if (abs((int)q[0] - (int)q[1]) > 6 || abs((int)q[1] - (int)q[2]) > 6)
                gray = false;
        }
        *grayOut = gray;
    }
    unsigned tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, px);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    stbi_image_free(px);
    return tex;
}

static bool import_glb(const std::string& path);
static bool import_glb_into(PropMesh& m, const std::string& path);

// A .blend is Blender's own working format -- there is no reading it from
// outside. Blender itself will convert one though, so this runs it in the
// background to export a glb beside the editor and imports that, which
// also means the model arrives with the materials Blender assigned.
static bool import_blend(const std::string& path)
{
    std::string blender;
    const char* env = SDL_getenv("BLENDER");
    if (env && *env) {
        blender = env;
    } else {
        const char* guesses[] = {
            "C:/Program Files/Blender Foundation/Blender 5.0/blender.exe",
            "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe",
            "C:/Program Files/Blender Foundation/Blender 4.0/blender.exe",
        };
        for (const char* g : guesses)
            if (std::filesystem::exists(g)) { blender = g; break; }
    }
    if (blender.empty()) {
        SDL_Log("import: no blender.exe found -- set BLENDER to its path");
        return false;
    }
    const std::string base = SDL_GetBasePath();
    const std::string out = base + "blend_import.glb";
    const std::string script = base + "blend_import.py";
    // Written to a file rather than passed as --python-expr: the quoting
    // does not survive a command line, and this needs real code.
    //
    // glTF can only carry a Principled BSDF's base colour and a texture
    // linked straight into it. These materials run the leaf texture
    // through a Multiply against a height Color Ramp, and an exporter
    // that cannot trace that link exports NO texture at all -- which is
    // why the leaves arrived white. So before exporting: find the image
    // feeding the graph and wire it directly to Base Color, and write the
    // ramp's two ends out beside the glb as the gradient the editor's
    // materials already understand.
    {
        FILE* sf = fopen(script.c_str(), "wb");
        if (!sf) {
            SDL_Log("import: could not write %s", script.c_str());
            return false;
        }
        fprintf(sf,
            "import bpy, os\n"
            "grad = []\n"
            "for m in bpy.data.materials:\n"
            "    if not m.use_nodes or not m.node_tree: continue\n"
            "    nt = m.node_tree\n"
            "    bsdf = next((n for n in nt.nodes if n.type=='BSDF_PRINCIPLED'), None)\n"
            "    img = next((n for n in nt.nodes if n.type=='TEX_IMAGE' and n.image), None)\n"
            "    if bsdf and img:\n"
            "        bc = bsdf.inputs['Base Color']\n"
            "        for l in list(bc.links): nt.links.remove(l)\n"
            "        nt.links.new(img.outputs['Color'], bc)\n"
            "        al = bsdf.inputs.get('Alpha')\n"
            "        if al is not None and 'Alpha' in img.outputs:\n"
            "            for l in list(al.links): nt.links.remove(l)\n"
            "            nt.links.new(img.outputs['Alpha'], al)\n"
            "    ramp = next((n for n in nt.nodes if n.type=='VALTORGB'), None)\n"
            "    if ramp:\n"
            "        e = ramp.color_ramp.elements\n"
            "        a = e[0].color; b = e[len(e)-1].color\n"
            "        grad.append('%%s %%f %%f %%f %%f %%f %%f' %% (m.name,\n"
            "            b[0],b[1],b[2], a[0],a[1],a[2]))\n"
            "bpy.ops.export_scene.gltf(filepath=r'%s', export_format='GLB',\n"
            "    export_apply=True, export_materials='EXPORT')\n"
            "open(r'%s','w').write('\\n'.join(grad))\n",
            out.c_str(), (out + ".grad").c_str());
        fclose(sf);
    }
    char cmd[2048];
    SDL_snprintf(cmd, sizeof cmd, "\"\"%s\" -b \"%s\" --python \"%s\"\"",
                 blender.c_str(), path.c_str(), script.c_str());
    SDL_Log("import: converting %s via blender...", path.c_str());
    const int rc = system(cmd);
    if (rc != 0 || !std::filesystem::exists(out)) {
        SDL_Log("import: blender conversion failed (%d)", rc);
        return false;
    }
    return import_glb(out);
}

// Fills a mesh from a glb. Split out so the library can reload one it
// already knows about, not only bring a new one in.
static bool import_glb_into(PropMesh& m, const std::string& path)
{
    cgltf_options opt{};
    cgltf_data* d = nullptr;
    if (cgltf_parse_file(&opt, path.c_str(), &d) != cgltf_result_success)
        return false;
    if (cgltf_load_buffers(&opt, d, path.c_str()) != cgltf_result_success) {
        cgltf_free(d);
        return false;
    }

    // top/bottom colours Blender pulled out of each Color Ramp, if the
    // model came in through the .blend path
    std::unordered_map<std::string, std::array<float, 6>> grads;
    {
        FILE* gf = fopen((path + ".grad").c_str(), "rb");
        if (gf) {
            char nm[256];
            float t0, t1, t2, b0, b1, b2;
            while (fscanf(gf, "%255s %f %f %f %f %f %f", nm, &t0, &t1, &t2,
                          &b0, &b1, &b2) == 7)
                grads[nm] = { t0, t1, t2, b0, b1, b2 };
            fclose(gf);
        }
    }
    std::unordered_map<const cgltf_image*, unsigned> texCache;
    std::unordered_map<const cgltf_image*, bool> grayCache;
    std::string dir = path;
    {
        size_t a = dir.find_last_of("/\\");
        dir = a == std::string::npos ? std::string(".") : dir.substr(0, a);
    }
    m.category = "Imported";
    {
        size_t a = path.find_last_of("/\\");
        m.label = a == std::string::npos ? path : path.substr(a + 1);
        size_t dot = m.label.find_last_of('.');
        if (dot != std::string::npos)
            m.label = m.label.substr(0, dot);
    }
    m.objPath = path;
    std::vector<float> data;
    float lo[3] = { 1e9f, 1e9f, 1e9f }, hi[3] = { -1e9f, -1e9f, -1e9f };

    // Walk NODES, not meshes: a glb positions its objects with node
    // transforms, and a mesh can be instanced by several of them. Reading
    // the mesh list alone drops every placement, so a whole scene collapses
    // onto the origin and you see only whatever is biggest.
    for (cgltf_size ni = 0; ni < d->nodes_count; ni++) {
        const cgltf_node& nd = d->nodes[ni];
        if (!nd.mesh)
            continue;
        cgltf_float xf[16];
        cgltf_node_transform_world(&nd, xf);
        const cgltf_mesh& me = *nd.mesh;
        for (cgltf_size pi = 0; pi < me.primitives_count; pi++) {
            const cgltf_primitive& pr = me.primitives[pi];
            if (pr.type != cgltf_primitive_type_triangles)
                continue;
            const cgltf_accessor* pos = nullptr;
            const cgltf_accessor* nrm = nullptr;
            const cgltf_accessor* uv = nullptr;
            // Blender bakes a lot of look into vertex colours -- leaf
            // tints especially -- and the exporter carries them, so a
            // model that arrives white was usually coloured this way
            const cgltf_accessor* col = nullptr;
            for (cgltf_size ai = 0; ai < pr.attributes_count; ai++) {
                const cgltf_attribute& at = pr.attributes[ai];
                if (at.type == cgltf_attribute_type_position) pos = at.data;
                else if (at.type == cgltf_attribute_type_normal) nrm = at.data;
                else if (at.type == cgltf_attribute_type_texcoord && !uv)
                    uv = at.data;
                else if (at.type == cgltf_attribute_type_color && !col)
                    col = at.data;
            }
            if (!pos)
                continue;

            PropMaterial mat;
            mat.name = m.label;
            if (pr.material && pr.material->has_pbr_metallic_roughness) {
                const cgltf_pbr_metallic_roughness& pbr =
                    pr.material->pbr_metallic_roughness;
                const float* bc = pbr.base_color_factor;
                // A texture carries the colour; the gradient multiplies
                // it, so leave it white unless a Color Ramp said otherwise.
                // Tinting by the base-colour factor and darkening the
                // bottom as well shaded the whole model down twice.
                const bool hasTex = pbr.base_color_texture.texture != nullptr;
                for (int k = 0; k < 3; k++) {
                    mat.kd[k] = hasTex ? 1.0f : bc[k];
                    mat.ka[k] = hasTex ? 1.0f : bc[k] * 0.72f;
                }
                if (pr.material->name)
                    mat.name = pr.material->name;
                auto gi = grads.find(mat.name);
                if (gi != grads.end()) {
                    // the ramp IS the gradient these materials paint with
                    for (int k = 0; k < 3; k++) {
                        mat.kd[k] = gi->second[k];
                        mat.ka[k] = gi->second[3 + k];
                    }
                }
                if (pbr.base_color_texture.texture) {
                    const cgltf_image* im = pbr.base_color_texture.texture->image;
                    auto it = texCache.find(im);
                    if (it == texCache.end()) {
                        bool gray = false;
                        const unsigned t = glb_texture(im, dir, &gray);
                        texCache[im] = t;
                        grayCache[im] = gray;
                        mat.tex = t;
                        // a mask only if a Color Ramp tints it: grey rock is
                        // a picture, not a mask, and drawing it as one makes
                        // it read far darker than it should
                        mat.grayMask = gray && grads.find(mat.name) != grads.end();
                    } else {
                        mat.tex = it->second;
                        mat.grayMask = grayCache[im] &&
                                       grads.find(mat.name) != grads.end();
                    }
                    if (mat.tex) {
                        // the texture carries the colour; the factor is a
                        // tint on top of it, which is usually white
                        mat.texName = im && im->uri ? im->uri : mat.name;
                    }
                }
            }
            const int matIdx = (int)m.mats.size();
            m.mats.push_back(mat);

            PropSubmesh sub;
            sub.mat = matIdx;
            sub.first = (int)(data.size() / 11);

            const cgltf_size n = pr.indices ? pr.indices->count : pos->count;
            for (cgltf_size k = 0; k < n; k++) {
                const cgltf_size v =
                    pr.indices ? cgltf_accessor_read_index(pr.indices, k) : k;
                float p3[3] = { 0, 0, 0 }, n3[3] = { 0, 1, 0 }, t2[2] = { 0, 0 };
                cgltf_accessor_read_float(pos, v, p3, 3);
                if (nrm) cgltf_accessor_read_float(nrm, v, n3, 3);
                if (uv)  cgltf_accessor_read_float(uv, v, t2, 2);
                // into the node's world space
                const float wx = xf[0] * p3[0] + xf[4] * p3[1] +
                                 xf[8] * p3[2] + xf[12];
                const float wy = xf[1] * p3[0] + xf[5] * p3[1] +
                                 xf[9] * p3[2] + xf[13];
                const float wz = xf[2] * p3[0] + xf[6] * p3[1] +
                                 xf[10] * p3[2] + xf[14];
                p3[0] = wx; p3[1] = wy; p3[2] = wz;
                // normals by the rotation part only, then renormalised
                const float nx2 = xf[0] * n3[0] + xf[4] * n3[1] + xf[8] * n3[2];
                const float ny2 = xf[1] * n3[0] + xf[5] * n3[1] + xf[9] * n3[2];
                const float nz2 = xf[2] * n3[0] + xf[6] * n3[1] + xf[10] * n3[2];
                const float nl = sqrtf(nx2 * nx2 + ny2 * ny2 + nz2 * nz2);
                if (nl > 1e-6f) {
                    n3[0] = nx2 / nl; n3[1] = ny2 / nl; n3[2] = nz2 / nl;
                }
                for (int c = 0; c < 3; c++) {
                    lo[c] = SDL_min(lo[c], p3[c]);
                    hi[c] = SDL_max(hi[c], p3[c]);
                }
                float c4[4] = { 1, 1, 1, 1 };
                // only where no texture carries the colour: multiplying a
                // texture by baked vertex shading darkens the whole model
                if (col && !mat.tex)
                    cgltf_accessor_read_float(col, v, c4, 4);
                data.insert(data.end(), { p3[0], p3[1], p3[2],
                                          n3[0], n3[1], n3[2],
                                          t2[0], t2[1],
                                          c4[0], c4[1], c4[2] });
                m.pts.insert(m.pts.end(), { p3[0], p3[1], p3[2] });
                m.ptMat.push_back(matIdx);
            }
            sub.count = (int)(data.size() / 11) - sub.first;
            if (sub.count > 0)
                m.subs.push_back(sub);
        }
    }
    cgltf_free(d);
    if (data.empty())
        return false;

    m.boundR = SDL_max(0.05f, SDL_max(hi[0] - lo[0], hi[2] - lo[2]) * 0.5f);
    m.boundH = SDL_max(0.05f, hi[1] - lo[1]);
    glGenVertexArrays(1, &m.vao);
    glBindVertexArray(m.vao);
    glGenBuffers(1, &m.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(),
                 GL_STATIC_DRAW);
    for (int a = 0; a < 4; a++)
        glEnableVertexAttribArray(a);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float),
                          (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float),
                          (void*)(6 * sizeof(float)));
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float),
                          (void*)(8 * sizeof(float)));
    glBindVertexArray(0);
    m.loaded = true;

    return true;
}

static bool import_glb(const std::string& path)
{
    PropMesh m;
    if (!import_glb_into(m, path))
        return false;
    // Keep the model in the prop library on disk, not just in memory.
    // A map stores props by "category/label" and the client resolves that
    // against files it can find -- an import that lives only in this
    // session is dropped when the island is loaded in game.
    {
        std::error_code ec;
        const std::string dir = gPropsDir + "/Imported";
        std::filesystem::create_directories(dir, ec);
        const std::string dst = dir + "/" + m.label + ".glb";
        if (path != dst)
            std::filesystem::copy_file(path, dst,
                std::filesystem::copy_options::overwrite_existing, ec);
        if (ec)
            SDL_Log("import: could not keep a copy in %s (%s)", dir.c_str(),
                    ec.message().c_str());
        else
            m.objPath = dst;
    }
    gPropMeshes.push_back(m);
    const int idx = (int)gPropMeshes.size() - 1;
    bool placed = false;
    for (PropCategory& c : gPropCats)
        if (c.name == "Imported") { c.meshes.push_back(idx); placed = true; }
    if (!placed) {
        PropCategory c;
        c.name = "Imported";
        c.meshes.push_back(idx);
        gPropCats.push_back(c);
    }
    SDL_Log("imported %s: %d parts", m.label.c_str(), (int)m.subs.size());
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
        // glb as well as obj: an imported model is kept in the library
        // as a glb, and indexing only obj meant it vanished from the
        // editor on the next run -- nothing to select, nothing to place
        for (const auto& e : fs::directory_iterator(dir + "/" + c, ec))
            if (e.path().extension() == ".obj" ||
                e.path().extension() == ".glb")
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
    for (const auto& e : fs::directory_iterator(dir + "/textures", ec))
        if (e.path().extension() == ".bmp")
            gPropTexFiles.push_back(e.path().filename().string());
    std::sort(gPropTexFiles.begin(), gPropTexFiles.end());
    load_styles();
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
    if (mode <= BRUSH_TAPER) {
        dt *= strength;
        float cell = 2.0f * TER_HALF / (HN - 1);
        int i0 = SDL_clamp((int)((cx - radius + TER_HALF) / cell), 0, HN - 1);
        int i1 = SDL_clamp((int)((cx + radius + TER_HALF) / cell) + 1, 0, HN - 1);
        int j0 = SDL_clamp((int)((cz - radius + TER_HALF) / cell), 0, HN - 1);
        int j1 = SDL_clamp((int)((cz + radius + TER_HALF) / cell) + 1, 0, HN - 1);
        std::vector<float> snap;
        if (mode == BRUSH_SMOOTH || mode == BRUSH_SHARPEN ||
            mode == BRUSH_EXPAND || mode == BRUSH_CONTRACT ||
            mode == BRUSH_TAPER)
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
                } else if (mode == BRUSH_EXPAND ||
                           mode == BRUSH_CONTRACT) {
                    // morphological dilate/erode: taking the max (or min)
                    // of the neighbourhood slides the whole silhouette --
                    // coastlines, cliff bases, plateau rims -- sideways,
                    // which a plain up/down brush cannot do
                    float best = snap[j * HN + i];
                    for (int dj = -2; dj <= 2; dj++)
                        for (int di = -2; di <= 2; di++) {
                            int ii = SDL_clamp(i + di, 0, HN - 1);
                            int jj = SDL_clamp(j + dj, 0, HN - 1);
                            float v = snap[jj * HN + ii];
                            best = (mode == BRUSH_EXPAND) ? SDL_max(best, v)
                                                          : SDL_min(best, v);
                        }
                    h += (best - h) * SDL_min(1.0f, 6.0f * w * dt);
                } else if (mode == BRUSH_SHAPE) {
                    // paint the generator's own terrain detail in by hand,
                    // so hand-sculpted ground matches generated ground
                    float n = gen_detail(x / TER_HALF, z / TER_HALF);
                    h += (invert ? -1.0f : 1.0f) * n * 2.2f *
                         SDL_max(1.0f, gGen.height) * w * dt;
                } else if (mode == BRUSH_TAPER) {
                    // roll the land down into the water: lowering alone
                    // leaves a step, so this averages as it sinks and the
                    // edge comes out as a beach rather than a cut
                    float sum = 0.0f;
                    int n = 0;
                    for (int dj = -3; dj <= 3; dj++)
                        for (int di = -3; di <= 3; di++) {
                            int ii = SDL_clamp(i + di, 0, HN - 1);
                            int jj = SDL_clamp(j + dj, 0, HN - 1);
                            sum += snap[jj * HN + ii];
                            n++;
                        }
                    float sea = gWaterline - 0.6f;
                    float target = invert ? sum / n
                                          : sum / n * 0.45f + sea * 0.55f;
                    h += (target - h) * SDL_min(1.0f, 5.0f * w * dt);
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
    float grassShadowDark = 0.55f, groundAO = 0.0f;
    float aoRadius = 2.5f, shadowStrength = 1.0f;
    char propSelId[96] = { 0 };   // selected prop as "category/label"
    float islandDepth = 0.0f;
    float waterline = -3.0f;
    int showWater = 1;
    float islandFrill = 0.0f;
    float islandBulge = 0.0f;
    // island generator (appended: older maps just get the defaults)
    // Every generator default comes from GenParams itself. Written out
    // by hand they went stale the moment the generator's own defaults
    // moved, and since loading a map fills the sliders from here, that
    // handed back the OLD defaults on any map saved before these fields
    // existed.
    int   genSeed = GenParams().seed;
    int   genDetail = GenParams().detail;
    int   genPeaks = GenParams().peaks;
    int   genAdd = GenParams().add ? 1 : 0;
    int   genOn = 0;
    float genSize = GenParams().size;
    float genCoast = GenParams().coast;
    float genLumps = GenParams().lumps;
    float genWarp = GenParams().warp;
    float genHeight = GenParams().height;
    float genRough = GenParams().rough;
    float genScale = GenParams().fscale;
    float genRidge = GenParams().ridge;
    float genPeakH = GenParams().peakH;
    float genSpread = GenParams().peakSpread;
    float genPlateau = GenParams().plateau;
    float genTerr = GenParams().terr;
    float genBeach = GenParams().beach;
    float genDrop = GenParams().drop;
    int   autoGrow = 1;
    int   trimSkirt = 0;
    float detailMult = 1.0f;
    float footLift = 0.15f;
    int   propsOnly = 0;   // quadrant built from props, no ground
    int   genFlats = GenParams().flats;
    int   genPaths = GenParams().paths ? 1 : 0;
    int   genPathPaint = GenParams().pathPaint ? 1 : 0;
    int   genShorePath = GenParams().shorePath ? 1 : 0;
    int   genSummitPath = GenParams().summitPath ? 1 : 0;
    int   genPathLayer = GenParams().pathLayer;
    int   genSpiral = GenParams().spiralRoad ? 1 : 0;
    float genSpiralTurn = GenParams().spiralTurn;
    float genSpiralInset = GenParams().spiralInset;
    float genSpiralHug = GenParams().spiralHug;
    float genPathFollow = GenParams().pathFollow;
    int   genRoadSupport = GenParams().roadSupport ? 1 : 0;
    float genRoadSupportW = GenParams().roadSupportW;
    float genFlatSize = GenParams().flatSize;
    float genFlatFlat = GenParams().flatFlat;
    float genPathWidth = GenParams().pathWidth;
    float genPathWander = GenParams().pathWander;
    float genPathCut = GenParams().pathCut;
    float genPathGrade = GenParams().pathGrade;
    float genPathBank = GenParams().pathBank;
    float genPathCling = GenParams().pathCling;
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
    // '9' marks a quadrant built from props: same layout as '8', but its
    // heightmap is a footprint stamped under the props for the sea and
    // collision to read, not ground meant to be drawn.
    const char magic[8] = { 'T','E','R','M','A','P','0',
                            gShowGround ? '8' : '9' };
    fwrite(magic, 1, 8, f);
    // world size and the resolutions that scale with it, so a map opens
    // at the size it was authored
    { float h = TER_HALF; Uint32 r[3] = { (Uint32)HN, (Uint32)MASK_N,
                                          (Uint32)GRID_N };
      fwrite(&h, 4, 1, f); fwrite(r, 4, 3, f); }
    // A props-only quadrant has no sculpted ground, but the client reads
    // the heightmap for three things it still needs: the shore field that
    // draws foam and ripples, the sea shading at range, and collision.
    // The built-in test island works exactly this way -- a mesh with a
    // baked heightfield that exists only for those. So rather than saving
    // the sunk plane, stamp the props' own footprint into it: land under
    // what was placed, open water everywhere else. Nothing downstream has
    // to change, because it is the same data the client always consumed.
    if (!gShowGround && !gProps.empty()) {
        // Deep enough to read as absent, not merely underwater: the
        // client treats ground above -50 as solid, so a footprint whose
        // sea was only a few units down made the whole quadrant walkable.
        std::vector<float> foot(gHeights.size(), -100.0f);
        const float cell = 2.0f * TER_HALF / (HN - 1);
        for (const PropInst& pi : gProps) {
            const PropMesh& pm = gPropMeshes[pi.mesh];
            const float r = pm.boundR * pi.scale;
            // only things big enough to be ground: a tree is scenery
            // standing ON the island, not part of its footprint
            if (r < 2.5f)
                continue;
            // Stamp the model itself where we have its geometry: a
            // bounding circle leaves you walking on invisible ground well
            // past the edge, and puts the surface at an average height
            // rather than the one you can see. Each vertex claims its own
            // cell at its own height; the mesh is dense enough that the
            // outline comes out as the model's own.
            if (!pm.pts.empty()) {
                // Rasterise the model's TRIANGLES, the way the built-in
                // test island's heightfield was baked. Stamping vertices
                // only samples the corners: a cell between them keeps
                // whatever the neighbours had, which reads as sinking into
                // a face you can see, and no amount of lifting the result
                // fixes a surface that was never measured. Interpolating
                // across each triangle gives the real height at each cell.
                const float cs = cosf(pi.yaw), sn = sinf(pi.yaw);
                auto place = [&](size_t v, float* o) {
                    const float lx = pm.pts[v] * pi.scale;
                    const float ly = pm.pts[v + 1] * pi.scale;
                    const float lz = pm.pts[v + 2] * pi.scale;
                    // Exactly the rotation model_trs draws with. This had
                    // the signs the other way round -- the transpose, i.e.
                    // a turn of -yaw -- so the collision and the shore
                    // field it feeds were rotated away from the model you
                    // can see. On a prop turned most of the way round that
                    // is ground and foam in entirely the wrong place.
                    o[0] = pi.x + lx * cs + lz * sn;
                    o[1] = pi.y + ly;
                    o[2] = pi.z - lx * sn + lz * cs;
                };
                for (size_t t = 0; t + 8 < pm.pts.size(); t += 9) {
                    const size_t vi = t / 3;
                    if (vi < pm.ptMat.size()) {
                        const int mi = pm.ptMat[vi];
                        if (mi >= 0 && mi < (int)pm.mats.size() &&
                            !pm.mats[mi].collide)
                            continue;   // scenery, not ground
                    }
                    float a[3], b[3], c[3];
                    place(t, a); place(t + 3, b); place(t + 6, c);
                    const float minx = SDL_min(a[0], SDL_min(b[0], c[0]));
                    const float maxx = SDL_max(a[0], SDL_max(b[0], c[0]));
                    const float minz = SDL_min(a[2], SDL_min(b[2], c[2]));
                    const float maxz = SDL_max(a[2], SDL_max(b[2], c[2]));
                    const int i0 = SDL_clamp((int)((minx + TER_HALF) / cell), 0, HN - 1);
                    const int i1 = SDL_clamp((int)((maxx + TER_HALF) / cell) + 1, 0, HN - 1);
                    const int j0 = SDL_clamp((int)((minz + TER_HALF) / cell), 0, HN - 1);
                    const int j1 = SDL_clamp((int)((maxz + TER_HALF) / cell) + 1, 0, HN - 1);
                    const float d = (b[2] - c[2]) * (a[0] - c[0]) +
                                    (c[0] - b[0]) * (a[2] - c[2]);
                    if (fabsf(d) < 1e-9f)
                        continue;      // edge-on, contributes no surface
                    bool covered = false;
                    for (int j = j0; j <= j1; j++)
                        for (int i = i0; i <= i1; i++) {
                            const float x = -TER_HALF + i * cell;
                            const float z = -TER_HALF + j * cell;
                            const float w0 = ((b[2] - c[2]) * (x - c[0]) +
                                              (c[0] - b[0]) * (z - c[2])) / d;
                            const float w1 = ((c[2] - a[2]) * (x - c[0]) +
                                              (a[0] - c[0]) * (z - c[2])) / d;
                            const float w2 = 1.0f - w0 - w1;
                            if (w0 < -0.001f || w1 < -0.001f || w2 < -0.001f)
                                continue;
                            const float y = a[1] * w0 + b[1] * w1 + c[1] * w2;
                            float& hh = foot[(size_t)j * HN + i];
                            if (y > hh)
                                hh = y;
                            covered = true;
                        }
                    if (!covered) {
                        // A triangle smaller than a cell can miss every
                        // cell centre and write nothing at all -- which is
                        // a pinhole in the surface, and exactly where you
                        // drop through. Claim the cells its corners land in
                        // so no face goes unrecorded however fine it is.
                        const float* vs[3] = { a, b, c };
                        for (const float* v : vs) {
                            const int i = (int)((v[0] + TER_HALF) / cell + 0.5f);
                            const int j = (int)((v[2] + TER_HALF) / cell + 0.5f);
                            if (i < 0 || j < 0 || i >= HN || j >= HN)
                                continue;
                            float& hh = foot[(size_t)j * HN + i];
                            if (v[1] > hh)
                                hh = v[1];
                        }
                    }
                }
                continue;
            }
            const float top = pi.y + pm.boundH * pi.scale * 0.55f;
            const int i0 = SDL_clamp((int)((pi.x - r + TER_HALF) / cell), 0, HN - 1);
            const int i1 = SDL_clamp((int)((pi.x + r + TER_HALF) / cell) + 1, 0, HN - 1);
            const int j0 = SDL_clamp((int)((pi.z - r + TER_HALF) / cell), 0, HN - 1);
            const int j1 = SDL_clamp((int)((pi.z + r + TER_HALF) / cell) + 1, 0, HN - 1);
            for (int j = j0; j <= j1; j++)
                for (int i = i0; i <= i1; i++) {
                    const float x = -TER_HALF + i * cell;
                    const float z = -TER_HALF + j * cell;
                    const float dx = x - pi.x, dz = z - pi.z;
                    if (dx * dx + dz * dz > r * r)
                        continue;
                    float& h = foot[(size_t)j * HN + i];
                    if (top > h)
                        h = top;
                }
        }
        // Close whatever the rasteriser still left open. Even with every
        // triangle recorded, a cell can end up empty where faces meet at an
        // angle, and a single empty cell in the middle of a surface is a
        // hole you fall through -- the client samples this bilinearly, so
        // one -100 neighbour drags the interpolated ground far below.
        for (int pass = 0; pass < 6; pass++) {
            std::vector<float> prev = foot;
            int fixed = 0;
            for (int j = 1; j < HN - 1; j++)
                for (int i = 1; i < HN - 1; i++) {
                    const size_t k = (size_t)j * HN + i;
                    if (prev[k] > -50.0f)
                        continue;
                    float best = -100.0f;
                    int n = 0;
                    for (int dj = -1; dj <= 1; dj++)
                        for (int di = -1; di <= 1; di++) {
                            const float v = prev[(size_t)(j + dj) * HN + i + di];
                            if (v > -50.0f) { n++; best = SDL_max(best, v); }
                        }
                    // five of eight neighbours means this is inside the
                    // surface, not along its edge
                    if (n >= 5) { foot[k] = best; fixed++; }
                }
            if (!fixed)
                break;
        }
        // Grow the surface a little past the model's rim. A cell counts as
        // surface only when its centre is covered, and the client samples
        // this field bilinearly -- so ground next to open water reads lower
        // than it is, and you fall a cell or two before the edge you can
        // see. Two rings of the nearest height put the drop where the model
        // actually ends.
        for (int pass = 0; pass < 2; pass++) {
            std::vector<float> prev = foot;
            for (int j = 1; j < HN - 1; j++)
                for (int i = 1; i < HN - 1; i++) {
                    const size_t k = (size_t)j * HN + i;
                    if (prev[k] > -50.0f)
                        continue;
                    float best = -100.0f;
                    for (int dj = -1; dj <= 1; dj++)
                        for (int di = -1; di <= 1; di++) {
                            const float v = prev[(size_t)(j + dj) * HN + i + di];
                            if (v > best)
                                best = v;
                        }
                    if (best > -50.0f)
                        foot[k] = best;
                }
        }
        int landCells = 0;
        for (float v : foot)
            if (v > -50.0f)
                landCells++;
        fwrite(foot.data(), sizeof(float), foot.size(), f);
        SDL_Log("saved a props footprint instead of terrain (%d land cells)",
                landCells);
    } else {
        fwrite(gHeights.data(), sizeof(float), gHeights.size(), f);
    }
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
    // size-prefixed settings blob: adding sliders later never breaks
    // older or newer saves
    Uint32 tsz = (Uint32)sizeof gTune;
    fwrite(&tsz, 4, 1, f);
    fwrite(&gTune, tsz, 1, f);
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
    if (magic[7] >= '8') {
        float h = 24.0f; Uint32 r[3] = { 257, 512, 256 };
        fread(&h, 4, 1, f); fread(r, 4, 3, f);
        TER_HALF = h; HN = (int)r[0]; MASK_N = (int)r[1]; GRID_N = (int)r[2];
        int g, hn, mk, gr;
        resolutions_for(TER_HALF, &g, &hn, &mk, &gr);
        GRASS_N = gr;
        gMapResized = true;      // the caller refreshes size-bound buffers
    }
    gHeights.assign((size_t)HN * HN, 0.0f);
    gMask.assign((size_t)MASK_N * MASK_N, 0);
    gMask2.assign((size_t)MASK_N * MASK_N, 0);
    gKill.assign((size_t)MASK_N * MASK_N, 255);
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
        // instances only render once their mesh is loaded -- placement
        // does this lazily, imports must do it here
        for (const PropInst& pi : gProps)
            load_prop(pi.mesh);
    }
    gLoadedSettings = false;
    gLoadedTune = false;
    if (magic[7] >= '5') {
        if (fread(&gSetBlade, sizeof(float), 1, f) == 1 &&
            fread(&gSetEdge, sizeof(float), 1, f) == 1 &&
            fread(gSetCam, sizeof(float), 5, f) == 5)
            gLoadedSettings = true;
    }
    if (magic[7] == '6') {
        // legacy fixed-size blob (best effort)
        if (fread(&gTune, sizeof gTune, 1, f) == 1)
            gLoadedTune = true;
    } else if (magic[7] >= '7') {
        Uint32 tsz = 0;
        if (fread(&tsz, 4, 1, f) == 1 && tsz > 0) {
            gTune = TuneBlob();   // defaults for fields the file predates
            Uint32 take = SDL_min(tsz, (Uint32)sizeof gTune);
            if (fread(&gTune, 1, take, f) == take) {
                if (tsz > take)
                    fseek(f, tsz - take, SEEK_CUR);
                gLoadedTune = true;
            }
        }
    }
    fclose(f);
    gHeightsDirty = gMaskDirty = gMask2Dirty = gKillDirty = true;
    // Ground is something a map either has or does not. A sculpted island
    // brings its terrain with it; a quadrant built out of props has no
    // land above the water and gets none drawn, no sculpt tools and
    // nothing under the cursor but the sea.
    gShowGround = false;
    for (float h : gHeights)
        if (h > gWaterline + 0.05f) { gShowGround = true; break; }
    SDL_Log("loaded %s (%s)", path,
            gShowGround ? "has terrain" : "props only");
    return true;
}

// ------------------------------------------------------------- world chart
// WW-style sea chart: an N x N looping grid of quadrants, each optionally
// holding an exported .wmap island. Saved as a simple text .wworld file.
static const int WORLD_MAX = 8;
static int gWorldSize = 7;               // Wind Waker's chart is 7x7
static std::string gWorldCells[WORLD_MAX][WORLD_MAX];
static int gWorldSel[2] = { 0, 0 };
static int gTestCell[2] = { 1, 0 };   // built-in test island quadrant
static int gSpawnCell[2] = { -1, -1 };  // where the game starts you; -1 = unset
static bool gShowWater = true;

// Every quadrant is a slot on disk under the editor, so islands persist
// between sessions and switching cells always has somewhere to save to.
// Whether the open map has anything in it: browsing quadrants should not
// litter the chart with blank islands, so only sculpted or painted work
// claims a slot.
static bool map_has_content()
{
    if (!gProps.empty())
        return true;
    for (float h : gHeights)
        if (fabsf(h) > 0.001f)
            return true;
    for (Uint8 v : gMask)
        if (v)
            return true;
    for (Uint8 v : gMask2)
        if (v)
            return true;
    for (Uint8 v : gKill)
        if (v != 255)
            return true;
    return false;
}

static std::string slot_path(int cx, int cy)
{
    std::string dir = std::string(SDL_GetBasePath()) + "islands";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    char name[64];
    SDL_snprintf(name, sizeof name, "/%c%d.wmap", 'A' + cx, cy + 1);
    return dir + name;
}

// the editor's own chart, reloaded next launch
static std::string editor_chart_path()
{
    return std::string(SDL_GetBasePath()) + "editor.wworld";
}

static void save_world(const char* path)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        SDL_Log("world save failed: %s", path);
        return;
    }
    fprintf(f, "wworld 1\nsize %d\nloop 1\nwaterline %f\n", gWorldSize,
            gWaterline);
    fprintf(f, "testisland %d %d\n", gTestCell[0], gTestCell[1]);
    for (int y = 0; y < gWorldSize; y++)
        for (int x = 0; x < gWorldSize; x++)
            if (!gWorldCells[y][x].empty())
                fprintf(f, "cell %d %d %s\n", x, y,
                        gWorldCells[y][x].c_str());
    fclose(f);
    SDL_Log("saved world %s", path);
}

static bool load_world(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;
    // merge rather than replace: a chart that only mentions one quadrant
    // should not wipe the islands sitting in the others
    char line[1200];
    while (fgets(line, sizeof line, f)) {
        int x, y, n;
        char p[1024];
        float wl;
        if (sscanf(line, "size %d", &n) == 1)
            gWorldSize = SDL_clamp(n, 2, WORLD_MAX);
        else if (sscanf(line, "waterline %f", &wl) == 1)
            gWaterline = wl;
        else if (sscanf(line, "spawn %d %d", &x, &y) == 2) {
            gSpawnCell[0] = SDL_clamp(x, 0, WORLD_MAX - 1);
            gSpawnCell[1] = SDL_clamp(y, 0, WORLD_MAX - 1);
        }
        else if (sscanf(line, "testisland %d %d", &x, &y) == 2) {
            gTestCell[0] = x;
            gTestCell[1] = y;
        }
        else if (sscanf(line, "editing %d %d", &x, &y) == 2) {
            gWorldSel[0] = SDL_clamp(x, 0, WORLD_MAX - 1);
            gWorldSel[1] = SDL_clamp(y, 0, WORLD_MAX - 1);
        }
        else if (sscanf(line, "cell %d %d %1023[^\n]", &x, &y, p) == 3)
            if (x >= 0 && x < WORLD_MAX && y >= 0 && y < WORLD_MAX)
                gWorldCells[y][x] = p;
    }
    fclose(f);
    SDL_Log("loaded world %s", path);
    return true;
}

// async file-dialog plumbing (SDL may invoke the callback off-thread:
// stash the result, act on it from the main loop)
static volatile int gDialogAction = 0;   // 1..5, see handler
static char gDialogFile[1024];
static const SDL_DialogFileFilter kMapFilters[] = {
    { "Windward map", "wmap" },
};
static const SDL_DialogFileFilter kGlbFilters[] = {
    { "glTF binary", "glb" },
};
static const SDL_DialogFileFilter kBlendFilters[] = {
    { "Blender file", "blend" },
};
static const SDL_DialogFileFilter kWorldFilters[] = {
    { "Windward world", "wworld" },
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
    bool genShot = false;
    if (argc >= 3 && SDL_strcmp(argv[1], "--shot") == 0)
        shotPath = argv[2];
    // --genshot <file.bmp> [seed]: run the island generator with the
    // built-in defaults and photograph it from above. Trail routing is
    // only judgeable from the top, and this keeps the judgement repeatable.
    if (argc >= 3 && SDL_strcmp(argv[1], "--genshot") == 0) {
        shotPath = argv[2];
        genShot = true;
        if (argc >= 4)
            gGen.seed = SDL_atoi(argv[3]);
        if (argc >= 5)   // sweep the road-hug control from the shell
            gGen.spiralHug = (float)SDL_atof(argv[4]);
        if (argc >= 6)
            gGen.spiralTurn = (float)SDL_atof(argv[5]);
    }

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
    GLuint aoProg = make_program(GRASS_VS, AO_FS);
    make_stamps();

    // live ground-AO map: blades rendered top-down, mipmapped for radius.
    // Its resolution follows the world size so a texel always covers the
    // same ground -- the AO radius is a mip level, so a map that grew
    // without this got wider, blurrier, darker contact shadows.
    int AO_N = ao_res_for(TER_HALF);
    GLuint aoTex = 0, aoFbo = 0;
    glGenTextures(1, &aoTex);
    auto alloc_ao = [&]() {
        glBindTexture(GL_TEXTURE_2D, aoTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, AO_N, AO_N, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
    };
    alloc_ao();
    glGenFramebuffers(1, &aoFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, aoFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, aoTex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        SDL_Log("AO FBO incomplete");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // top-down projection: world xz -> clip xy over the terrain extent
    // top-down projection for the AO pass. Rebuilt per frame from the
    // CURRENT world size: baking it once meant that after the map grew,
    // the AO map still covered the old extent while the terrain looked
    // AO up across the new one, so every blade's contact shadow landed
    // in the wrong place -- including on the dirt paths, which have no
    // blades and should stay open.
    Mat4 topDown{};
    auto build_topdown = [&]() {
        topDown = Mat4{};
        topDown.m[0] = 1.0f / TER_HALF;
        topDown.m[9] = 1.0f / TER_HALF;
        topDown.m[15] = 1.0f;
    };
    build_topdown();

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

    // Terrain mesh, rebuilt whenever the map is resized so the quad
    // count tracks the world size and detail per unit stays put.
    GLsizei terIdxCount = 0;
    GLuint terVao = 0, terVbo = 0, terIbo = 0;
    auto rebuild_terrain_mesh = [&]() {
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
        if (terVao) { glDeleteVertexArrays(1, &terVao); glDeleteBuffers(1, &terVbo); glDeleteBuffers(1, &terIbo); }
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

        terIdxCount = (GLsizei)idx.size();
    };
    rebuild_terrain_mesh();

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
    // grass instances: rebuilt on resize so density per unit holds
    std::vector<float> inst;
    int instCount = 0;
    auto rebuild_grass_instances = [&]() {
        inst.clear();
    // instances: jittered grid, deterministic
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

        instCount = GRASS_N * GRASS_N;
    };
    rebuild_grass_instances();

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

    // Resize the world and refresh every buffer that depends on its
    // size: the mesh, the grass field, and the shader-placed rings.
    // keep the AO map's texel density matched to the world size
    auto resync_ao = [&]() {
        int want = ao_res_for(TER_HALF);
        if (want == AO_N)
            return;
        AO_N = want;
        alloc_ao();   // same texture object, so the FBO stays attached
    };
    auto apply_map_resize = [&](float newHalf) {
        resize_map(newHalf);
        resync_ao();
        rebuild_terrain_mesh();
        rebuild_grass_instances();
        glBindBuffer(GL_ARRAY_BUFFER, instVbo);
        glBufferData(GL_ARRAY_BUFFER, inst.size() * sizeof(float),
                     inst.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        gShoreBase.clear();
        gShore.on = false;
    };
    // an undo/redo that crossed a resize needs the size-bound buffers back
    auto refresh_if_resized = [&]() {
        if (!gMapResized)
            return;
        gMapResized = false;
        resync_ao();
        rebuild_terrain_mesh();
        rebuild_grass_instances();
        glBindBuffer(GL_ARRAY_BUFFER, instVbo);
        glBufferData(GL_ARRAY_BUFFER, inst.size() * sizeof(float),
                     inst.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    };
    // widen the canvas around the sculpt, keeping it exactly where it is
    auto apply_canvas_grow = [&](float newHalf) {
        grow_canvas(newHalf, gWaterline);
        resync_ao();
        rebuild_terrain_mesh();
        rebuild_grass_instances();
        glBindBuffer(GL_ARRAY_BUFFER, instVbo);
        glBufferData(GL_ARRAY_BUFFER, inst.size() * sizeof(float),
                     inst.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    };

    GLuint emptyVao = 0;
    glGenVertexArrays(1, &emptyVao);

    // Link-scale reference dummy (0.55 units = his client height / 2)
    GLuint dummyProg = make_program(DUMMY_VS, DUMMY_FS);
    GLuint dummyVao = 0, dummyVbo = 0;
    int dummyVerts = 0, dummyMarkerVerts = 0;
    {
        std::vector<float> dv;
        auto box = [&dv](float x0, float y0, float z0, float x1, float y1,
                         float z1, float r, float g, float b) {
            const float c[8][3] = {
                { x0, y0, z0 }, { x1, y0, z0 }, { x1, y1, z0 }, { x0, y1, z0 },
                { x0, y0, z1 }, { x1, y0, z1 }, { x1, y1, z1 }, { x0, y1, z1 },
            };
            const int f[12][3] = {
                {0,1,2},{0,2,3}, {5,4,7},{5,7,6}, {4,0,3},{4,3,7},
                {1,5,6},{1,6,2}, {3,2,6},{3,6,7}, {4,5,1},{4,1,0},
            };
            for (auto& t : f)
                for (int k = 0; k < 3; k++) {
                    dv.insert(dv.end(), c[t[k]], c[t[k]] + 3);
                    dv.insert(dv.end(), { r, g, b });
                }
        };
        const float H = 0.55f;              // total height
        box(-0.085f, 0.0f, -0.05f, -0.02f, H * 0.42f, 0.05f,
            0.93f, 0.93f, 0.90f);           // legs
        box(0.02f, 0.0f, -0.05f, 0.085f, H * 0.42f, 0.05f,
            0.93f, 0.93f, 0.90f);
        box(-0.10f, H * 0.40f, -0.07f, 0.10f, H * 0.72f, 0.07f,
            0.20f, 0.55f, 0.22f);           // tunic
        box(-0.075f, H * 0.72f, -0.06f, 0.075f, H * 0.90f, 0.06f,
            0.95f, 0.85f, 0.70f);           // head
        box(-0.08f, H * 0.88f, -0.065f, 0.08f, H, 0.065f,
            0.25f, 0.62f, 0.28f);           // cap
        dummyVerts = (int)dv.size() / 6;
        // marker pole (drawn separately) so a 0.55u figure is findable
        box(-0.02f, H, -0.02f, 0.02f, H + 6.0f, 0.02f, 1.0f, 0.3f, 0.8f);
        dummyMarkerVerts = (int)dv.size() / 6 - dummyVerts;
        glGenVertexArrays(1, &dummyVao);
        glGenBuffers(1, &dummyVbo);
        glBindVertexArray(dummyVao);
        glBindBuffer(GL_ARRAY_BUFFER, dummyVbo);
        glBufferData(GL_ARRAY_BUFFER, dv.size() * sizeof(float), dv.data(),
                     GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, (void*)12);
        glBindVertexArray(0);
    }

    // outer extension: perimeter x outward steps, placed by the shader
    GLuint extProg = make_program(EXT_VS, TER_FS);
    GLuint extVao = 0, extVbo = 0, extIbo = 0;
    GLsizei extIdxCount = 0;
    {
        const int P = 256;      // points around the rim
        const int M = 24;       // outward steps
        std::vector<float> ev;  // borderX, borderZ, dirX, dirZ, t
        auto push = [&ev](float bx, float bz, float t) {
            float dx = 0.0f, dz = 0.0f;
            if (fabsf(fabsf(bx) - TER_HALF) < 0.001f) dx = bx > 0 ? 1.0f : -1.0f;
            if (fabsf(fabsf(bz) - TER_HALF) < 0.001f) dz = bz > 0 ? 1.0f : -1.0f;
            float l = sqrtf(dx * dx + dz * dz);
            if (l > 0.0f) { dx /= l; dz /= l; }
            ev.insert(ev.end(), { bx, bz, dx, dz, t });
        };
        std::vector<float> ring;
        for (int i = 0; i < P / 4; i++)
            ring.insert(ring.end(),
                { -TER_HALF + 2.0f * TER_HALF * i / (P / 4), -TER_HALF });
        for (int i = 0; i < P / 4; i++)
            ring.insert(ring.end(),
                { TER_HALF, -TER_HALF + 2.0f * TER_HALF * i / (P / 4) });
        for (int i = P / 4; i > 0; i--)
            ring.insert(ring.end(),
                { -TER_HALF + 2.0f * TER_HALF * i / (P / 4), TER_HALF });
        for (int i = P / 4; i > 0; i--)
            ring.insert(ring.end(),
                { -TER_HALF, -TER_HALF + 2.0f * TER_HALF * i / (P / 4) });
        const int RP = (int)ring.size() / 2;
        for (int p = 0; p < RP; p++)
            for (int m = 0; m <= M; m++)
                push(ring[p * 2], ring[p * 2 + 1], (float)m / M);
        std::vector<unsigned> ei;
        for (int p = 0; p < RP; p++) {
            int pn = (p + 1) % RP;
            for (int m = 0; m < M; m++) {
                unsigned a = p * (M + 1) + m, b = a + 1;
                unsigned c = pn * (M + 1) + m, d = c + 1;
                ei.insert(ei.end(), { a, b, c, c, b, d });
            }
        }
        extIdxCount = (GLsizei)ei.size();
        glGenVertexArrays(1, &extVao);
        glGenBuffers(1, &extVbo);
        glGenBuffers(1, &extIbo);
        glBindVertexArray(extVao);
        glBindBuffer(GL_ARRAY_BUFFER, extVbo);
        glBufferData(GL_ARRAY_BUFFER, ev.size() * sizeof(float), ev.data(),
                     GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, extIbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, ei.size() * sizeof(unsigned),
                     ei.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 20, (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, (void*)8);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 20, (void*)16);
        glBindVertexArray(0);
    }

    // island skirt geometry: perimeter ring (rim + bottom verts) + cap
    GLuint skirtProg = make_program(SKIRT_VS, SKIRT_FS);
    // trim the underside to the island's own coastline instead of the
    // whole map square
    bool trimSkirt = false;
    GLuint skirtVao = 0, skirtVbo = 0, skirtIbo = 0;
    int skirtIdxCount = 0;
    // The skirt rings the map's perimeter, so it has to follow the world
    // size: built once, a grown map kept an underside sized for its old
    // bounds. Rebuilt whenever the extent or the grid changes.
    float skirtHalf = 0.0f;
    int skirtGrid = 0;
    float skirtCenter[2] = { 0.0f, 0.0f };
    bool skirtWasTrimmed = false;
    auto build_skirt = [&]() {
        skirtHalf = TER_HALF;
        skirtGrid = GRID_N;
        skirtWasTrimmed = trimSkirt;
        skirtCenter[0] = skirtCenter[1] = 0.0f;
        std::vector<float> ring;   // perimeter xz positions, CCW
        bool trimmed = false;
        if (trimSkirt) {
            // Trace the island's own coastline and hang the underside off
            // THAT instead of the map's square perimeter: the shape below
            // the water then matches the shape above it, with no block of
            // rock filling the rest of the map.
            const float cell = 2.0f * TER_HALF / (HN - 1);
            const float sea = gWaterline + 0.05f;
            double cx = 0, cz = 0;
            int n = 0;
            for (int j = 0; j < HN; j++)
                for (int i = 0; i < HN; i++)
                    if (gHeights[(size_t)j * HN + i] > sea) {
                        cx += -TER_HALF + i * cell;
                        cz += -TER_HALF + j * cell;
                        n++;
                    }
            if (n > 32) {
                cx /= n; cz /= n;
                // radial contour: the outermost land along each bearing.
                // The skirt is a ring plus a centre fan, so it wants a
                // star-shaped outline anyway -- deep bays get bridged.
                const int P = SDL_clamp(GRID_N, 96, 512);
                std::vector<float> rad((size_t)P, 0.0f);
                const float step = cell * 0.75f;
                const float rmax = TER_HALF * 2.2f;
                for (int p = 0; p < P; p++) {
                    float a = 6.2831853f * p / P;
                    float dx = cosf(a), dz = sinf(a);
                    float found = 0.0f;
                    for (float r = 0.0f; r < rmax; r += step) {
                        float x = (float)cx + dx * r, z = (float)cz + dz * r;
                        if (fabsf(x) > TER_HALF || fabsf(z) > TER_HALF)
                            break;
                        if (height_at(x, z) > sea)
                            found = r;
                    }
                    rad[p] = found;
                }
                // smooth around the ring so the silhouette is not jagged
                for (int pass = 0; pass < 2; pass++) {
                    std::vector<float> t = rad;
                    for (int p = 0; p < P; p++)
                        rad[p] = (t[(p + P - 1) % P] + 2.0f * t[p] +
                                  t[(p + 1) % P]) * 0.25f;
                }
                for (int p = 0; p < P; p++) {
                    float a = 6.2831853f * p / P;
                    float r = rad[p] + cell * 1.5f;   // meet the shore
                    ring.insert(ring.end(), {
                        SDL_clamp((float)cx + cosf(a) * r, -TER_HALF, TER_HALF),
                        SDL_clamp((float)cz + sinf(a) * r, -TER_HALF, TER_HALF) });
                }
                skirtCenter[0] = (float)cx;
                skirtCenter[1] = (float)cz;
                trimmed = true;
            }
        }
        if (!trimmed) {
            for (int i = 0; i < GRID_N; i++)
                ring.insert(ring.end(),
                    { -TER_HALF + 2.0f * TER_HALF * i / GRID_N, -TER_HALF });
            for (int j = 0; j < GRID_N; j++)
                ring.insert(ring.end(),
                    { TER_HALF, -TER_HALF + 2.0f * TER_HALF * j / GRID_N });
            for (int i = GRID_N; i > 0; i--)
                ring.insert(ring.end(),
                    { -TER_HALF + 2.0f * TER_HALF * i / GRID_N, TER_HALF });
            for (int j = GRID_N; j > 0; j--)
                ring.insert(ring.end(),
                    { -TER_HALF, -TER_HALF + 2.0f * TER_HALF * j / GRID_N });
        }
        int P = (int)ring.size() / 2;
        std::vector<float> sv;          // x, z, t
        for (int p = 0; p < P; p++) {
            sv.insert(sv.end(), { ring[p * 2], ring[p * 2 + 1], 0.0f });
            sv.insert(sv.end(), { ring[p * 2], ring[p * 2 + 1], 1.0f });
        }
        int centerIdx = P * 2;
        sv.insert(sv.end(), { skirtCenter[0], skirtCenter[1], 2.0f });
        std::vector<unsigned> si;
        for (int p = 0; p < P; p++) {
            unsigned a = p * 2, b = a + 1;
            unsigned c = ((p + 1) % P) * 2, d = c + 1;
            si.insert(si.end(), { a, b, c, c, b, d });       // side quad
            si.insert(si.end(), { b, (unsigned)centerIdx, d }); // cap fan
        }
        skirtIdxCount = (int)si.size();
        if (!skirtVao) {
            glGenVertexArrays(1, &skirtVao);
            glGenBuffers(1, &skirtVbo);
            glGenBuffers(1, &skirtIbo);
        }
        glBindVertexArray(skirtVao);
        glBindBuffer(GL_ARRAY_BUFFER, skirtVbo);
        glBufferData(GL_ARRAY_BUFFER, sv.size() * sizeof(float), sv.data(),
                     GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, skirtIbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, si.size() * sizeof(unsigned),
                     si.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindVertexArray(0);
    };
    build_skirt();

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
    bool autoGrow = true;   // widen the map when a sculpt reaches the rim
    // generated land beyond the map bounds
    bool extendOn = false;
    float extendDist = 10.0f, extendNoise = 0.6f, extendDrop = 3.0f;
    bool showDummy = false;             // Link-scale reference figure
    bool dummyMarker = true;            // tall pole so he is findable
    float dummyPos[2] = { 0.0f, 0.0f }; // placed with the brush cursor
    float grassShadowDark = 0.55f;   // blade brightness inside shadow
    float groundAO = 0.0f;           // contact AO strength under blades
    float aoRadius = 2.5f;           // AO spread (mip level)
    float shadowStrength = 1.0f;     // sun shadow intensity
    float islandDepth = 0.0f;        // skirt extrusion below the terrain
    float islandFrill = 0.0f;        // scalloped underside silhouette
    float islandBulge = 0.0f;        // underside pushed outward past rim
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
        gTune.grassShadowDark = grassShadowDark;
        gTune.groundAO = groundAO;
        gTune.aoRadius = aoRadius;
        gTune.shadowStrength = shadowStrength;
        gTune.islandDepth = islandDepth;
        gTune.waterline = gWaterline;
        gTune.showWater = gShowWater ? 1 : 0;
        gTune.islandFrill = islandFrill;
        gTune.islandBulge = islandBulge;
        gTune.genSeed = gGen.seed;      gTune.genDetail = gGen.detail;
        gTune.genPeaks = gGen.peaks;    gTune.genAdd = gGen.add ? 1 : 0;
        gTune.genOn = gGen.on ? 1 : 0;  gTune.genSize = gGen.size;
        gTune.genCoast = gGen.coast;    gTune.genLumps = gGen.lumps;
        gTune.genWarp = gGen.warp;      gTune.genHeight = gGen.height;
        gTune.genRough = gGen.rough;    gTune.genScale = gGen.fscale;
        gTune.genRidge = gGen.ridge;    gTune.genPeakH = gGen.peakH;
        gTune.genSpread = gGen.peakSpread;
        gTune.genPlateau = gGen.plateau; gTune.genTerr = gGen.terr;
        gTune.genBeach = gGen.beach;    gTune.genDrop = gGen.drop;
        gTune.autoGrow = autoGrow ? 1 : 0;
        gTune.trimSkirt = trimSkirt ? 1 : 0;
        gTune.detailMult = gDetailMult;
        gTune.footLift = gFootLift;
        gTune.propsOnly = gShowGround ? 0 : 1;
        gTune.genFlats = gGen.flats;
        gTune.genPaths = gGen.paths ? 1 : 0;
        gTune.genPathPaint = gGen.pathPaint ? 1 : 0;
        gTune.genShorePath = gGen.shorePath ? 1 : 0;
        gTune.genFlatSize = gGen.flatSize;
        gTune.genFlatFlat = gGen.flatFlat;
        gTune.genPathWidth = gGen.pathWidth;
        gTune.genPathWander = gGen.pathWander;
        gTune.genPathCut = gGen.pathCut;
        gTune.genPathGrade = gGen.pathGrade;
        gTune.genPathBank = gGen.pathBank;
        gTune.genPathCling = gGen.pathCling;
        gTune.genSummitPath = gGen.summitPath ? 1 : 0;
        gTune.genPathLayer = gGen.pathLayer;
        gTune.genSpiral = gGen.spiralRoad ? 1 : 0;
        gTune.genSpiralTurn = gGen.spiralTurn;
        gTune.genSpiralInset = gGen.spiralInset;
        gTune.genSpiralHug = gGen.spiralHug;
        gTune.genPathFollow = gGen.pathFollow;
        gTune.genRoadSupport = gGen.roadSupport ? 1 : 0;
        gTune.genRoadSupportW = gGen.roadSupportW;
        memset(gTune.propSelId, 0, sizeof gTune.propSelId);
        if (propSel >= 0 && propSel < (int)gPropMeshes.size())
            SDL_strlcpy(gTune.propSelId,
                        mesh_id(gPropMeshes[propSel]).c_str(),
                        sizeof gTune.propSelId);
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
            sculptTool = SDL_clamp(gTune.sculptTool, 0, 8);
            paintLayer = SDL_clamp(gTune.paintLayer, 0, 2);
            detailTool = SDL_clamp(gTune.detailTool, 0, 1);
            propTool = SDL_clamp(gTune.propTool, 0, 3);
            propCat = gTune.propCat;
            shadowsOn = gTune.shadows != 0;
            grassShadowDark = gTune.grassShadowDark;
            groundAO = gTune.groundAO;
            aoRadius = gTune.aoRadius;
            shadowStrength = gTune.shadowStrength;
            islandDepth = gTune.islandDepth;
            gWaterline = gTune.waterline;
            gShowWater = gTune.showWater != 0;
            islandFrill = gTune.islandFrill;
            islandBulge = gTune.islandBulge;
            // The generator's sliders are deliberately NOT restored from
            // the map. They are a tool preset, not map content -- whatever
            // the generator produced is already baked into the heights --
            // and restoring them meant any map saved with older values
            // handed those back every time it was opened, quietly
            // overriding the built-in defaults. Reset to Defaults in the
            // Shape tab is the way back to them.
            gGen.on = false;
            gGenBase.clear();
            autoGrow = gTune.autoGrow != 0;
            trimSkirt = gTune.trimSkirt != 0;
            gDetailMult = SDL_clamp(gTune.detailMult, 0.5f, 4.0f);
            gFootLift = SDL_clamp(gTune.footLift, 0.0f, 1.0f);
            gShowGround = gTune.propsOnly == 0;
            if (gTune.propSelId[0]) {
                for (int mi = 0; mi < (int)gPropMeshes.size(); mi++)
                    if (mesh_id(gPropMeshes[mi]) == gTune.propSelId) {
                        propSel = mi;
                        break;
                    }
            }
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

    if (genShot) {
        gGenBase = gHeights;
        gGenMask = gMask; gGenMask2 = gMask2; gGenKill = gKill;
        gGen.on = true;
        apply_generator(gWaterline);
        simTime = 7.0;
        yaw = 0.0f;
        pitch = -1.45f;            // very nearly straight down
        camPos[0] = 0.0f;
        camPos[1] = TER_HALF * 2.05f + gGen.height * 1.6f;
        camPos[2] = 0.6f;
    } else if (shotPath) {
        stamp_demo_scene();
        simTime = 7.0;
        yaw = 0.0f;
        pitch = -0.30f;
        camPos[0] = 4.5f; camPos[1] = 7.0f; camPos[2] = 33.0f;
    } else {
        // the chart first: it says which quadrant we were working in and
        // where each island lives, so the workspace comes back as left
        if (load_world(editor_chart_path().c_str())) {
            const std::string& cur = gWorldCells[gWorldSel[1]][gWorldSel[0]];
            if (!cur.empty() && load_map(cur.c_str()))
                applySettingsIn();
            else if (load_map(mapPath))
                applySettingsIn();
        } else if (load_map(mapPath)) {
            applySettingsIn();   // startup load must restore the sliders too
        }
        gMapResized = false;
    }
    SDL_Log("gen defaults in effect: seed %d size %.2f height %.1f terr %.2f "
            "detail %d fscale %.2f plateau %.2f", gGen.seed, gGen.size,
            gGen.height, gGen.terr, gGen.detail, gGen.fscale, gGen.plateau);

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
                    refresh_if_resized();
                }
                if (e.key.key == SDLK_Y && (e.key.mod & SDL_KMOD_CTRL)) {
                    do_redo();
                    refresh_if_resized();
                }
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
        } else if (gDialogAction == 3) {
            gDialogAction = 0;
            gWorldCells[gWorldSel[1]][gWorldSel[0]] = gDialogFile;
        } else if (gDialogAction == 4) {
            gDialogAction = 0;
            std::string p = gDialogFile;
            if (p.size() < 7 || p.substr(p.size() - 7) != ".wworld")
                p += ".wworld";
            // A chart is only a list of paths to islands, so the island
            // on screen has to reach disk first -- otherwise the cell
            // still points at whatever was saved last and everything
            // since (props, grass, paint) is missing from the world.
            if (map_has_content()) {
                if (gWorldCells[gWorldSel[1]][gWorldSel[0]].empty())
                    gWorldCells[gWorldSel[1]][gWorldSel[0]] =
                        slot_path(gWorldSel[0], gWorldSel[1]);
                syncSettingsOut();
                save_map(gWorldCells[gWorldSel[1]][gWorldSel[0]].c_str());
            }
            save_world(p.c_str());
        } else if (gDialogAction == 6) {
            gDialogAction = 0;
            if (!import_glb(gDialogFile))
                SDL_Log("could not import %s", gDialogFile);
        } else if (gDialogAction == 7) {
            gDialogAction = 0;
            if (!import_blend(gDialogFile))
                SDL_Log("could not import %s", gDialogFile);
        } else if (gDialogAction == 5) {
            gDialogAction = 0;
            if (load_world(gDialogFile)) {
                // a chart on its own changes nothing on screen: open the
                // island of whichever quadrant it says we were editing
                const std::string& cur =
                    gWorldCells[gWorldSel[1]][gWorldSel[0]];
                if (!cur.empty() && load_map(cur.c_str()))
                    applySettingsIn();
                else if (cur.empty())
                    new_map();
                rebuild_terrain_mesh();
                rebuild_grass_instances();
                glBindBuffer(GL_ARRAY_BUFFER, instVbo);
                glBufferData(GL_ARRAY_BUFFER, inst.size() * sizeof(float),
                             inst.data(), GL_STATIC_DRAW);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                gMapResized = false;
                save_world(editor_chart_path().c_str());
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
            // With no ground, do not raycast the terrain at all: it is
            // still there, sunk far below the sea, and the ray happily
            // hits it -- which put props a hundred units under the water
            // instead of on the waterline.
            // The waterline is a real surface to work on, not a
            // fallback. A quadrant built out of props has no terrain at
            // all, so the cursor has to land somewhere regardless: the sea
            // plane is the one thing every quadrant has, and it is where a
            // prop-built level sits. Ground, when there is any, simply
            // wins wherever it stands above the water.
            hasHit = false;
            if (rd[1] < -0.0001f) {
                const float t = (gWaterline - camPos[1]) / rd[1];
                if (t > 0.0f) {
                    // Inside the quadrant only. Letting a prop sit
                    // anywhere the sea does put islands out beyond the map
                    // -- nothing to bake a footprint from, and in game they
                    // draw a chart-length away from where you spawn.
                    hit[0] = SDL_clamp(camPos[0] + rd[0] * t,
                                       -TER_HALF, TER_HALF);
                    hit[1] = gWaterline;
                    hit[2] = SDL_clamp(camPos[2] + rd[2] * t,
                                       -TER_HALF, TER_HALF);
                    // clamped, never refused: rejecting a click that met
                    // the sea beyond the quadrant meant most of the screen
                    // did nothing, since the camera sits well back
                    hasHit = true;
                }
            }
            if (gShowGround) {
                float gh[3] = { 0, 0, 0 };
                if (ray_terrain(camPos, rd, gh) && gh[1] > gWaterline) {
                    hit[0] = gh[0]; hit[1] = gh[1]; hit[2] = gh[2];
                    hasHit = true;
                }
            }
            if (ImGui::GetIO().WantCaptureMouse)
                hasHit = false;   // cursor over the panel: never paint through
            bool painting = hasHit && (mb & SDL_BUTTON_LMASK) && !shotPath;
            bool clickEdge = painting && !wasPainting;
            if (painting && activeTab != 3 && mode == BRUSH_ROAD) {
                if (!wasPainting) {
                    push_undo();
                    gRoadX.clear();
                    gRoadZ.clear();
                    gRoadBaseH = gHeights;
                    gRoadBaseM = gMask;
                    gRoadBaseM2 = gMask2;
                    gRoadBaseK = gKill;
                }
                float cellW = 2.0f * TER_HALF / (HN - 1);
                bool added = gRoadX.empty();
                if (!added) {
                    float ddx = hit[0] - gRoadX.back();
                    float ddz = hit[2] - gRoadZ.back();
                    added = sqrtf(ddx * ddx + ddz * ddz) > cellW * 1.5f;
                }
                if (added) {
                    gRoadX.push_back(hit[0]);
                    gRoadZ.push_back(hit[2]);
                    // rebuild from the snapshot so the road is cut once,
                    // over the whole stroke, however slowly it is drawn
                    gHeights = gRoadBaseH;
                    gMask = gRoadBaseM;
                    gMask2 = gRoadBaseM2;
                    gKill = gRoadBaseK;
                    road_carve(gRoadX, gRoadZ, 0.0f);
                }
            } else if (painting && activeTab != 3) {
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
                                       gShowGround
                                           ? height_at(hit[0], hit[2])
                                           : hit[1],
                                       hit[2], yaw, sc });
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
                        gProps.push_back({ propSel, px,
                                           gShowGround ? height_at(px, pz)
                                                       : gWaterline,
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
            // A stroke that pushed land into the rim has nowhere left to
            // go, so grow the canvas around it: same sculpt, same scale,
            // more ground to keep pushing into.
            if (autoGrow && wasPainting && !painting && activeTab == 0 &&
                mode <= BRUSH_TAPER && TER_HALF < 400.0f &&
                land_at_edge(gWaterline))
                apply_canvas_grow(TER_HALF * 1.3f);
            wasPainting = painting;
            if (selInst >= (int)gProps.size())
                selInst = -1;
        }

        const bool heightsChanged = gHeightsDirty;
        upload_dirty();

        // live ground-AO pass: blades top-down into the AO map
        {
            build_topdown();   // the map may have grown since last frame
            // the trimmed outline follows the terrain, so rebuild it when
            // the map changes shape -- but not mid-stroke, which would
            // retrace the coastline every frame
            if (skirtHalf != TER_HALF || skirtGrid != GRID_N ||
                skirtWasTrimmed != trimSkirt ||
                (trimSkirt && heightsChanged && !wasPainting))
                build_skirt();
            glBindFramebuffer(GL_FRAMEBUFFER, aoFbo);
            glViewport(0, 0, AO_N, AO_N);
            glDisable(GL_DEPTH_TEST);
            glClearColor(1, 1, 1, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            if (showGrass && groundAO > 0.001f) {
                glUseProgram(aoProg);
                glUniformMatrix4fv(glGetUniformLocation(aoProg, "uMvp"), 1,
                                   GL_FALSE, topDown.m);
                glUniform1f(glGetUniformLocation(aoProg, "uHalf"), TER_HALF);
                glUniform1f(glGetUniformLocation(aoProg, "uTime"),
                            (float)simTime);
                glUniform1f(glGetUniformLocation(aoProg, "uDensity"),
                            bladeDensity);
                glUniform1i(glGetUniformLocation(aoProg, "uShadowsOn"), 0);
                glUniform1f(glGetUniformLocation(aoProg, "uSwayAmp"), 0.0f);
                glUniform1i(glGetUniformLocation(aoProg, "uFlatten"), 1);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, gHeightTex);
                glUniform1i(glGetUniformLocation(aoProg, "uHeight"), 0);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, gMaskTex);
                glUniform1i(glGetUniformLocation(aoProg, "uMask"), 1);
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_2D, gKillTex);
                glUniform1i(glGetUniformLocation(aoProg, "uKill"), 3);
                glActiveTexture(GL_TEXTURE4);
                glBindTexture(GL_TEXTURE_2D, gMask2Tex);
                glUniform1i(glGetUniformLocation(aoProg, "uMask2"), 4);
                glBindVertexArray(grassVao);
                glDrawArraysInstanced(GL_TRIANGLES, 0, 12, instCount);
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, aoTex);
            glGenerateMipmap(GL_TEXTURE_2D);
            glEnable(GL_DEPTH_TEST);
        }

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
            glDrawElements(GL_TRIANGLES, terIdxCount, GL_UNSIGNED_INT,
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
        if (gShowGround) {
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
        glActiveTexture(GL_TEXTURE9);
        glBindTexture(GL_TEXTURE_2D, aoTex);
        glUniform1i(glGetUniformLocation(terProg, "uAOMap"), 9);
        glUniform1f(glGetUniformLocation(terProg, "uGrassAO"), groundAO);
        glUniform1f(glGetUniformLocation(terProg, "uGrassAORad"), aoRadius);
        glUniform1f(glGetUniformLocation(terProg, "uShadowStr"),
                    shadowStrength);
        glUniformMatrix4fv(glGetUniformLocation(terProg, "uLightMvp"), 1,
                           GL_FALSE, lightMvp.m);
        glUniform1i(glGetUniformLocation(terProg, "uShadowsOn"),
                    shadowsOn ? 1 : 0);
        glBindVertexArray(terVao);
        glDrawElements(GL_TRIANGLES, terIdxCount, GL_UNSIGNED_INT, nullptr);

        // generated land beyond the map bounds
        if (extendOn) {
            glUseProgram(extProg);
            glUniformMatrix4fv(glGetUniformLocation(extProg, "uMvp"), 1,
                               GL_FALSE, mvp.m);
            glUniform1f(glGetUniformLocation(extProg, "uHalf"), TER_HALF);
            glUniform1f(glGetUniformLocation(extProg, "uDist"), extendDist);
            glUniform1f(glGetUniformLocation(extProg, "uNoise"), extendNoise);
            glUniform1f(glGetUniformLocation(extProg, "uSea"), gWaterline);
            glUniform1f(glGetUniformLocation(extProg, "uDrop"), extendDrop);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, gHeightTex);
            glUniform1i(glGetUniformLocation(extProg, "uHeight"), 0);
            // same painted materials as the terrain proper
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, gMaskTex);
            glUniform1i(glGetUniformLocation(extProg, "uMask"), 1);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, grassTex);
            glUniform1i(glGetUniformLocation(extProg, "uGrassTex"), 2);
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, dirtTex);
            glUniform1i(glGetUniformLocation(extProg, "uDirtTex"), 3);
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, cliffTex);
            glUniform1i(glGetUniformLocation(extProg, "uCliffTex"), 4);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, gMask2Tex);
            glUniform1i(glGetUniformLocation(extProg, "uMask2"), 5);
            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_2D, dirt2Tex);
            glUniform1i(glGetUniformLocation(extProg, "uDirt2Tex"), 6);
            glActiveTexture(GL_TEXTURE8);
            glBindTexture(GL_TEXTURE_2D, shadowTex);
            glUniform1i(glGetUniformLocation(extProg, "uShadowMap"), 8);
            glUniformMatrix4fv(glGetUniformLocation(extProg, "uLightMvp"), 1,
                               GL_FALSE, lightMvp.m);
            glUniform1i(glGetUniformLocation(extProg, "uShadowsOn"),
                        shadowsOn ? 1 : 0);
            glUniform1f(glGetUniformLocation(extProg, "uShadowStr"),
                        shadowStrength);
            glUniform1f(glGetUniformLocation(extProg, "uEdgeBreak"),
                        edgeBreak);
            glUniform1f(glGetUniformLocation(extProg, "uGrassAO"), 0.0f);
            glUniform1f(glGetUniformLocation(extProg, "uGrassAORad"), 0.0f);
            glActiveTexture(GL_TEXTURE9);
            glBindTexture(GL_TEXTURE_2D, aoTex);
            glUniform1i(glGetUniformLocation(extProg, "uAOMap"), 9);
            float noBrush[4] = { 0, 0, 1, 0 };
            glUniform4fv(glGetUniformLocation(extProg, "uBrush"), 1, noBrush);
            glDisable(GL_CULL_FACE);
            glBindVertexArray(extVao);
            glDrawElements(GL_TRIANGLES, extIdxCount, GL_UNSIGNED_INT,
                           nullptr);
        }

        }   // gShowGround
        // island skirt underside
        if (gShowGround && islandDepth > 0.05f) {
            glUseProgram(skirtProg);
            glUniformMatrix4fv(glGetUniformLocation(skirtProg, "uMvp"), 1,
                               GL_FALSE, mvp.m);
            glUniform1f(glGetUniformLocation(skirtProg, "uHalf"), TER_HALF);
            glUniform1f(glGetUniformLocation(skirtProg, "uDepth"),
                        islandDepth);
            glUniform1f(glGetUniformLocation(skirtProg, "uFrill"),
                        islandFrill);
            glUniform1f(glGetUniformLocation(skirtProg, "uBulge"),
                        islandBulge);
            glUniform2f(glGetUniformLocation(skirtProg, "uCenter"),
                        skirtCenter[0], skirtCenter[1]);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, gHeightTex);
            glUniform1i(glGetUniformLocation(skirtProg, "uHeight"), 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, cliffTex);
            glUniform1i(glGetUniformLocation(skirtProg, "uCliffTex"), 1);
            glDisable(GL_CULL_FACE);
            glBindVertexArray(skirtVao);
            glDrawElements(GL_TRIANGLES, skirtIdxCount, GL_UNSIGNED_INT,
                           nullptr);
        }

        // Link-scale reference figure
        if (showDummy) {
            glUseProgram(dummyProg);
            glUniformMatrix4fv(glGetUniformLocation(dummyProg, "uMvp"), 1,
                               GL_FALSE, mvp.m);
            float dp[3] = { dummyPos[0],
                            height_at(dummyPos[0], dummyPos[1]),
                            dummyPos[1] };
            glUniform3fv(glGetUniformLocation(dummyProg, "uPos"), 1, dp);
            glBindVertexArray(dummyVao);
            glDrawArrays(GL_TRIANGLES, 0, dummyVerts);
            if (dummyMarker) {
                glDrawArrays(GL_TRIANGLES, dummyVerts, dummyMarkerVerts);
            }
        }

        // ocean plane at the waterline
        if (gShowWater) {
            static GLuint waterProg = make_program(WATER_VS, WATER_FS);
            glUseProgram(waterProg);
            glUniformMatrix4fv(glGetUniformLocation(waterProg, "uMvp"), 1,
                               GL_FALSE, mvp.m);
            glUniform1f(glGetUniformLocation(waterProg, "uLevel"),
                        gWaterline);
            glUniform1f(glGetUniformLocation(waterProg, "uExtent"),
                        TER_HALF * 8.0f);
            glUniform1f(glGetUniformLocation(waterProg, "uHalf"), TER_HALF);
            glUniform1f(glGetUniformLocation(waterProg, "uEdge"),
                        gShowGround ? 0.0f : 1.0f);
            glUniform4f(glGetUniformLocation(waterProg, "uBrush"),
                        hit[0], hit[2], brushRadius,
                        hasHit && !shotPath ? 1.0f : 0.0f);
            glUniform3fv(glGetUniformLocation(waterProg, "uBrushCol"), 1,
                         kBrushColors[mode]);
            glBindVertexArray(emptyVao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }

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
            glUniform1f(glGetUniformLocation(propProg, "uShadowStr"),
                        shadowStrength);
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
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, shadowTex);
            glUniform1i(glGetUniformLocation(grassProg, "uShadowMap"), 5);
            glUniformMatrix4fv(glGetUniformLocation(grassProg, "uLightMvp"),
                               1, GL_FALSE, lightMvp.m);
            glUniform1i(glGetUniformLocation(grassProg, "uShadowsOn"),
                        shadowsOn ? 1 : 0);
            glUniform1f(glGetUniformLocation(grassProg, "uShadowDark"),
                        grassShadowDark);
            glUniform1f(glGetUniformLocation(grassProg, "uSwayAmp"), 1.0f);
            glUniform1i(glGetUniformLocation(grassProg, "uFlatten"), 0);
            glDisable(GL_CULL_FACE);
            glBindVertexArray(grassVao);
            if (gShowGround)
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
                if (gShowGround && ImGui::BeginTabItem("Sculpt")) {
                    activeTab = 0;
                    const char* tools[] = { "Raise or Lower Terrain",
                                            "Smooth Height", "Flatten",
                                            "Sharpen Edges", "Terrace",
                                            "Expand Land", "Contract Land",
                                            "Shape Paint", "Taper to Sea",
                                            "Path" };
                    ImGui::Combo("##sculpttool", &sculptTool, tools, 10);
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
                        "Push the coastline and cliff bases OUTWARD -- "
                        "grows land sideways without changing its shape.",
                        "Pull the coastline and cliff bases INWARD -- "
                        "carves land back sideways, opening water.",
                        "Paint the Shape tab's terrain detail in by hand -- "
                        "same noise, seed and ridges as the generator.\n"
                        "Hold Shift to carve it in instead.",
                        "Roll the land down into the water, smoothing as it "
                        "sinks, so edges end as beaches instead of cliffs.\n"
                        "Hold Shift to just smooth without lowering.",
                        "Drag to lay a road: it cuts a flat tread at a "
                        "walkable grade, banks the ground either side and "
                        "paints the surface.\nUse it where the generated "
                        "road cannot go -- a heightmap holds one height "
                        "per spot, so a spiral cannot cross above itself.",
                    };
                    ImGui::TextWrapped("%s", helps[sculptTool]);
                    static const BrushMode toolModes[] = {
                        BRUSH_RAISE, BRUSH_SMOOTH, BRUSH_FLATTEN,
                        BRUSH_SHARPEN, BRUSH_TERRACE,
                        BRUSH_EXPAND, BRUSH_CONTRACT,
                        BRUSH_SHAPE, BRUSH_TAPER, BRUSH_ROAD,
                    };
                    mode = toolModes[sculptTool];
                    brushGallery();
                    if (mode == BRUSH_ROAD) {
                        ImGui::SliderFloat("Road Width", &gGen.pathWidth,
                                           0.6f, 8.0f, "%.1f");
                        ImGui::SliderFloat("Road Cut", &gGen.pathCut,
                                           0.0f, 1.0f, "%.2f");
                        ImGui::SliderFloat("Bank Slope", &gGen.pathBank,
                                           0.2f, 3.0f, "%.2f");
                        ImGui::SliderFloat("Max Grade", &gGen.pathGrade,
                                           0.05f, 1.2f, "%.2f");
                        ImGui::SliderFloat("Follow Ground", &gGen.pathFollow,
                                           0.0f, 1.0f, "%.2f");
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip(
                                "1: the road rolls with the hillside.\n"
                                "0: it holds one graded line and cuts the "
                                "ground to meet it, which on a slope comes "
                                "out as a flat pad.");
                        ImGui::Checkbox("Paint Surface", &gGen.pathPaint);
                        if (gGen.pathPaint) {
                            const char* surf[] = { "Path Dirt (brown)",
                                                   "Soft Dirt (sand)" };
                            ImGui::Combo("Surface", &gGen.pathLayer, surf, 2);
                        }
                        ImGui::Checkbox("Build Hillside for Road",
                                        &gGen.roadSupport);
                        if (gGen.roadSupport)
                            ImGui::SliderFloat("Buttress Reach",
                                               &gGen.roadSupportW,
                                               1.0f, 20.0f, "%.1f");
                        ImGui::TextDisabled("shares the generator's road\n"
                                            "settings, so hand-drawn and\n"
                                            "generated roads match");
                    } else {
                        ImGui::SliderFloat("Strength", &brushStrength,
                                           0.1f, 3.0f, "%.1f");
                    }
                    if (mode == BRUSH_TERRACE)
                        ImGui::SliderFloat("Step Height", &gTerraceStep,
                                           0.5f, 6.0f, "%.1f");
                    ImGui::SliderFloat("Island Extrusion", &islandDepth,
                                       0.0f, 30.0f, "%.1f");
                    ImGui::SliderFloat("Island Frill", &islandFrill,
                                       0.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("Island Bulge", &islandBulge,
                                       0.0f, 1.0f, "%.2f");
                    ImGui::Checkbox("Trim Underside", &trimSkirt);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Hangs the underside off the island's own "
                            "coastline instead of the whole map square, so "
                            "below the water it is the same shape as above "
                            "it.\nTraces the shore wherever the land meets "
                            "the waterline; deep bays get bridged, since the "
                            "underside is a single solid mass.");
                    ImGui::SeparatorText("Shoreline");
                    bool shoreDirty = false;
                    if (ImGui::Checkbox("Shoreline", &gShore.on)) {
                        if (gShoreBase.size() != gHeights.size()) {
                            push_undo();
                            gShoreBase = gHeights;
                        }
                        shoreDirty = true;
                    }
                    if (gShore.on) {
                        shoreDirty |= ImGui::SliderFloat(
                            "Width", &gShore.width, 0.01f, 0.6f, "%.3f",
                            ImGuiSliderFlags_Logarithmic);
                        shoreDirty |= ImGui::SliderFloat(
                            "Frill", &gShore.frill, 0.0f, 1.5f, "%.2f");
                        shoreDirty |= ImGui::SliderFloat(
                            "Smooth", &gShore.smoothing, 0.0f, 1.0f, "%.2f");
                        shoreDirty |= ImGui::SliderFloat(
                            "Drop", &gShore.drop, 0.3f, 8.0f, "%.1f");
                        shoreDirty |= ImGui::Checkbox("Round Footprint",
                                                      &gShore.radial);
                        if (ImGui::Button("Bake into Terrain")) {
                            gShoreBase.clear();   // keep it, drop the undo
                            gShore.on = false;
                        }
                        ImGui::TextDisabled("live: sliders rebuild the rim\n"
                                            "from the original heights");
                    }
                    if (shoreDirty)
                        apply_shoreline(gWaterline);
                    ImGui::SeparatorText("Canvas");
                    ImGui::Checkbox("Auto-Grow Map", &autoGrow);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Sculpt into the map edge and the heightmap has "
                            "nowhere to go -- the mesh gets sliced off flat.\n"
                            "With this on, a stroke that reaches the rim "
                            "widens the map by 30%% and tapers the new ground "
                            "into the sea.\nYour island keeps its exact size "
                            "and position; only the canvas around it grows, "
                            "at the same polygon density.");
                    ImGui::SameLine();
                    if (ImGui::Button("Grow Now"))
                        apply_canvas_grow(TER_HALF * 1.3f);
                    ImGui::TextDisabled("map is %.0f x %.0f units",
                                        TER_HALF * 2.0f, TER_HALF * 2.0f);
                    ImGui::SeparatorText("Outer Extension");
                    ImGui::Checkbox("Extend Land Outward", &extendOn);
                    if (extendOn) {
                        ImGui::SliderFloat("Reach", &extendDist,
                                           1.0f, 30.0f, "%.1f");
                        ImGui::SliderFloat("Ragged", &extendNoise,
                                           0.0f, 1.5f, "%.2f");
                        ImGui::SliderFloat("Sink", &extendDrop,
                                           0.3f, 10.0f, "%.1f");
                        ImGui::TextDisabled("grows new coast past the map\n"
                                            "edge -- keeps your sculpt whole");
                    }
                    ImGui::SeparatorText("Scale Reference");
                    if (ImGui::Checkbox("Link Dummy", &showDummy) &&
                        showDummy) {
                        // drop him where you are looking, else at the
                        // camera's ground position (he is small: 0.55u)
                        if (hasHit) {
                            dummyPos[0] = hit[0];
                            dummyPos[1] = hit[2];
                        } else {
                            dummyPos[0] = SDL_clamp(camPos[0], -TER_HALF,
                                                    TER_HALF);
                            dummyPos[1] = SDL_clamp(camPos[2], -TER_HALF,
                                                    TER_HALF);
                        }
                    }
                    if (showDummy) {
                        ImGui::TextDisabled("0.55u tall = Link in game");
                        if (ImGui::Button("Move to Cursor") && hasHit) {
                            dummyPos[0] = hit[0];
                            dummyPos[1] = hit[2];
                        }
                        ImGui::SameLine();
                        ImGui::Checkbox("Marker", &dummyMarker);
                    }
                    ImGui::EndTabItem();
                }
                if (gShowGround && ImGui::BeginTabItem("Shape")) {
                    activeTab = 0;
                    ImGui::TextWrapped(
                        "Generates a whole island from a seed. Live: every "
                        "slider rebuilds from the heights you had before "
                        "the generator was switched on, so nothing is lost "
                        "until you bake.");
                    bool genDirty = false;
                    if (ImGui::Checkbox("Generate Island", &gGen.on)) {
                        if (gGenBase.size() != gHeights.size()) {
                            push_undo();
                            gGenBase = gHeights;
                            gGenMask = gMask; gGenMask2 = gMask2;
                            gGenKill = gKill; gGenProps = gProps;
                        }
                        genDirty = true;
                    }
                    if (gGen.on) {
                        ImGui::SeparatorText("Resolution");
                        {
                            bool x2 = gDetailMult > 1.5f && gDetailMult < 3.0f;
                            bool x4 = gDetailMult >= 3.0f;
                            if (ImGui::Checkbox("Detail x2", &x2)) {
                                gDetailMult = x2 ? 2.0f : 1.0f;
                                apply_detail();
                                refresh_if_resized();
                                genDirty = true;
                            }
                            ImGui::SameLine();
                            if (ImGui::Checkbox("x4", &x4)) {
                                gDetailMult = x4 ? 4.0f : 1.0f;
                                apply_detail();
                                refresh_if_resized();
                                genDirty = true;
                            }
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip(
                                    "Four times the cells. Sixteen times the "
                                    "heightmap and mask memory, so expect it "
                                    "to be slow to regenerate and to sculpt "
                                    "on.");
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip(
                                    "Twice the cells for the same world, so "
                                    "road treads, ramps and painted edges "
                                    "stop being a few cells wide.\nCosts "
                                    "memory and frame time everywhere -- the "
                                    "grid is uniform, so it cannot be spent "
                                    "only where it is needed.");
                            ImGui::SameLine();
                            ImGui::TextDisabled("grid %d", GRID_N);
                        }
                        ImGui::SeparatorText("Seed");
                        genDirty |= ImGui::InputInt("Seed", &gGen.seed);
                        if (ImGui::Button("Randomize")) {
                            gGen.seed = (int)(SDL_GetTicks() * 2654435761u
                                              % 100000u);
                            genDirty = true;
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Next")) { gGen.seed++; genDirty = true; }
                        ImGui::SameLine();
                        if (ImGui::Button("Prev")) { gGen.seed--; genDirty = true; }

                        ImGui::SeparatorText("Footprint");
                        genDirty |= ImGui::SliderFloat("Island Size",
                                                       &gGen.size, 0.15f, 1.1f,
                                                       "%.2f");
                        genDirty |= ImGui::SliderFloat("Coast Falloff",
                                                       &gGen.coast, 0.05f, 1.0f,
                                                       "%.2f");
                        genDirty |= ImGui::SliderFloat("Coast Wander",
                                                       &gGen.lumps, 0.0f, 1.2f,
                                                       "%.2f");
                        genDirty |= ImGui::SliderFloat("Warp", &gGen.warp,
                                                       0.0f, 1.5f, "%.2f");
                        genDirty |= ImGui::SliderFloat("Sea Floor",
                                                       &gGen.drop, 0.5f, 12.0f,
                                                       "%.1f");

                        ImGui::SeparatorText("Terrain");
                        genDirty |= ImGui::SliderFloat("Height", &gGen.height,
                                                       1.0f, 40.0f, "%.1f");
                        genDirty |= ImGui::SliderFloat("Roughness",
                                                       &gGen.rough, 0.0f, 1.5f,
                                                       "%.2f");
                        genDirty |= ImGui::SliderInt("Detail", &gGen.detail,
                                                     1, 8);
                        genDirty |= ImGui::SliderFloat("Feature Scale",
                                                       &gGen.fscale, 0.5f,
                                                       14.0f, "%.2f");
                        genDirty |= ImGui::SliderFloat("Ridges", &gGen.ridge,
                                                       0.0f, 1.0f, "%.2f");

                        ImGui::SeparatorText("Summits");
                        genDirty |= ImGui::SliderInt("Peaks", &gGen.peaks, 0, 6);
                        genDirty |= ImGui::SliderFloat("Peak Height",
                                                       &gGen.peakH, 0.0f, 2.0f,
                                                       "%.2f");
                        genDirty |= ImGui::SliderFloat("Peak Spread",
                                                       &gGen.peakSpread, 0.0f,
                                                       1.2f, "%.2f");

                        ImGui::SeparatorText("Profile");
                        genDirty |= ImGui::SliderFloat("Plateau",
                                                       &gGen.plateau, 0.0f,
                                                       1.0f, "%.2f");
                        genDirty |= ImGui::SliderFloat("Terrace Step",
                                                       &gGen.terr, 0.0f, 8.0f,
                                                       "%.1f");
                        genDirty |= ImGui::SliderFloat("Beach Width",
                                                       &gGen.beach, 0.0f, 1.5f,
                                                       "%.2f");

                        ImGui::SeparatorText("Level Layout");
                        genDirty |= ImGui::SliderInt("Clearings",
                                                     &gGen.flats, 0, 6);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Flat shelves levelled into the "
                                              "terrain -- somewhere to put a "
                                              "village, a shrine, a fight.");
                        genDirty |= ImGui::SliderFloat("Clearing Size",
                                                       &gGen.flatSize, 0.05f,
                                                       0.5f, "%.2f");
                        genDirty |= ImGui::SliderFloat("Clearing Flatness",
                                                       &gGen.flatFlat, 0.0f,
                                                       1.0f, "%.2f");
                        genDirty |= ImGui::Checkbox("Trails", &gGen.paths);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Routes a path between the "
                                              "clearings, following the "
                                              "ground at a walkable grade "
                                              "instead of cutting a ramp "
                                              "straight through it.");
                        if (gGen.paths) {
                            genDirty |= ImGui::SliderFloat("Trail Width",
                                                           &gGen.pathWidth,
                                                           0.6f, 8.0f, "%.1f");
                            genDirty |= ImGui::SliderFloat("Trail Wander",
                                                           &gGen.pathWander,
                                                           0.0f, 1.5f, "%.2f");
                            genDirty |= ImGui::SliderFloat("Trail Cut",
                                                           &gGen.pathCut, 0.0f,
                                                           1.0f, "%.2f");
                            genDirty |= ImGui::Checkbox("Terrace Road",
                                                        &gGen.spiralRoad);
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip(
                                    "One road wrapping the hill: it follows "
                                    "each terrace for part of a turn, then "
                                    "steps down onto the next.\nOff: trails "
                                    "are routed between the clearings "
                                    "instead, taking the short way.");
                            if (gGen.spiralRoad) {
                                genDirty |= ImGui::SliderFloat(
                                    "Turns per Terrace", &gGen.spiralTurn,
                                    0.1f, 4.0f, "%.2f");
                                if (ImGui::IsItemHovered())
                                    ImGui::SetTooltip(
                                        "Laps of the hill the road takes to "
                                        "climb one terrace.\nRaise it if "
                                        "the road merges into the slope: "
                                        "that is consecutive laps landing "
                                        "on top of each other.");
                                genDirty |= ImGui::SliderFloat(
                                    "Road Inset", &gGen.spiralInset,
                                    0.0f, 5.0f, "%.2f");
                                genDirty |= ImGui::SliderFloat(
                                    "Road Hugs Terraces", &gGen.spiralHug,
                                    0.0f, 1.0f, "%.2f");
                                genDirty |= ImGui::SliderFloat(
                                    "Follow Ground", &gGen.pathFollow,
                                    0.0f, 1.0f, "%.2f");
                                genDirty |= ImGui::Checkbox(
                                    "Build Hillside for Road",
                                    &gGen.roadSupport);
                                if (ImGui::IsItemHovered())
                                    ImGui::SetTooltip(
                                        "Grows the slope outward to carry "
                                        "the road instead of only cutting "
                                        "into it.\nOn a face too steep to "
                                        "hold a tread this is what gives "
                                        "the road somewhere to sit, and "
                                        "what lets laps of the spiral meet "
                                        "up rather than cut through each "
                                        "other.");
                                if (gGen.roadSupport)
                                    genDirty |= ImGui::SliderFloat(
                                        "Buttress Reach", &gGen.roadSupportW,
                                        1.0f, 20.0f, "%.1f");
                                if (ImGui::IsItemHovered())
                                    ImGui::SetTooltip(
                                        "1: follows the terrace treads and "
                                        "the rock faces exactly.\n0: reads "
                                        "the hill as smooth slopes and cuts "
                                        "the road across the risers.\n"
                                        "Separate from Cling to Terrain, "
                                        "which steers the routed trails.");
                            }
                            genDirty |= ImGui::SliderFloat("Cling to Terrain",
                                                           &gGen.pathCling,
                                                           0.0f, 1.0f, "%.2f");
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip(
                                    "1: every terrace riser is a wall, so "
                                    "trails wind around the hill following "
                                    "the treads -- switchback roads.\n"
                                    "0: risers are just slopes, so trails "
                                    "cut across and climb straight up.");
                            genDirty |= ImGui::SliderFloat("Bank Slope",
                                                           &gGen.pathBank,
                                                           0.2f, 3.0f, "%.2f");
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip(
                                    "Angle of the cut and fill either side "
                                    "of the tread.\nLower spreads the "
                                    "earthworks wider and gentler.");
                            genDirty |= ImGui::SliderFloat("Max Grade",
                                                           &gGen.pathGrade,
                                                           0.05f, 1.2f, "%.2f");
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip(
                                    "Steepest climb a trail will accept.\n"
                                    "Lower it and routes stop charging over "
                                    "hills -- they hug the contours and "
                                    "switchback up instead.");
                            genDirty |= ImGui::Checkbox("Paint Dirt",
                                                        &gGen.pathPaint);
                            if (gGen.pathPaint) {
                                const char* surf[] = { "Path Dirt (brown)",
                                                       "Soft Dirt (sand)" };
                                genDirty |= ImGui::Combo("Surface",
                                                         &gGen.pathLayer,
                                                         surf, 2);
                            }
                            genDirty |= ImGui::Checkbox("Trail to Beach",
                                                        &gGen.shorePath);
                            genDirty |= ImGui::Checkbox("Trail to Summit",
                                                        &gGen.summitPath);
                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip(
                                    "Runs a trail to the island's high "
                                    "point. With Cling up it has to spiral "
                                    "the hill to get there.");
                        }
                        ImGui::SeparatorText("Landmarks");
                        genDirty |= ImGui::Checkbox("Places of Interest",
                                                    &gGen.landmarks);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip(
                                "A cave cut into a steep flank and roofed "
                                "with cliff props, a cairn on the summit, "
                                "and groves on the clearings.\nThe cave "
                                "roof is props because a heightmap holds "
                                "one height per spot and cannot overhang.");
                        if (gGen.landmarks)
                            genDirty |= ImGui::SliderFloat(
                                "Density", &gGen.landmarkDens, 0.2f, 3.0f,
                                "%.2f");
                        ImGui::SeparatorText("Apply");
                        genDirty |= ImGui::Checkbox("Layer Over Sculpt",
                                                    &gGen.add);
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("On: adds the generated land on "
                                              "top of what you already had.\n"
                                              "Off: replaces the terrain "
                                              "entirely.");
                        if (ImGui::Button("Reset to Defaults")) {
                            int keepSeed = gGen.seed;
                            bool keepOn = gGen.on;
                            gGen = GenParams();
                            gGen.seed = keepSeed;
                            gGen.on = keepOn;
                            genDirty = true;
                        }
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Every slider back to the "
                                              "built-in defaults, keeping "
                                              "the seed.");
                        if (ImGui::Button("Bake into Terrain")) {
                            gGenBase.clear();   // keep the result, drop revert
                            gGenMask.clear(); gGenMask2.clear();
                            gGenKill.clear(); gGenProps.clear();
                            gGen.on = false;
                            gShoreBase.clear();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Revert")) {
                            gGen.on = false;
                            genDirty = true;
                        }
                        ImGui::TextDisabled("Bake, then use Shape Paint and\n"
                                            "Taper to Sea in the Sculpt tab\n"
                                            "to work the result by hand.");
                    }
                    if (genDirty)
                        apply_generator(gWaterline);
                    ImGui::EndTabItem();
                }
                if (gShowGround && ImGui::BeginTabItem("Paint")) {
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
                if (gShowGround && ImGui::BeginTabItem("Details")) {
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
                    ImGui::SliderFloat("Shadow Dark", &grassShadowDark,
                                       0.2f, 1.0f, "%.2f");
                    ImGui::SliderFloat("Ground AO", &groundAO,
                                       0.0f, 0.8f, "%.2f");
                    ImGui::SliderFloat("AO Radius", &aoRadius,
                                       0.0f, 6.0f, "%.1f");
                    brushGallery();
                    ImGui::SliderFloat("Strength", &brushStrength,
                                       0.1f, 3.0f, "%.1f");
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Props")) {
                    activeTab = 3;
                    if (ImGui::Button("Import GLB..."))
                        SDL_ShowOpenFileDialog(map_dialog_cb, (void*)6, win,
                                               kGlbFilters, 1, nullptr, false);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Brings your own model in as a prop, under an "
                            "Imported category.\nColours come from each "
                            "material's base colour; textures are not "
                            "unpacked yet, so a textured model arrives "
                            "flat-shaded.");
                    ImGui::SameLine();
                    if (ImGui::Button("Import .blend..."))
                        SDL_ShowOpenFileDialog(map_dialog_cb, (void*)7, win,
                                               kBlendFilters, 1, nullptr,
                                               false);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Runs Blender in the background to convert the "
                            "file, then imports it.\nNeeds Blender "
                            "installed; set the BLENDER environment "
                            "variable if it is somewhere unusual.");
                    ImGui::Separator();
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
                    // per-model style presets by material role: swap the
                    // Trunk / Leaves texture and gradient colors, saved to
                    // props/styles.txt (survives pack re-exports)
                    if (propSel >= 0 &&
                        ImGui::CollapsingHeader("Style Presets") &&
                        load_prop(propSel)) {
                        PropMesh& spm = gPropMeshes[propSel];
                        if (!spm.ptMat.empty()) {
                            ImGui::SeparatorText("Collision");
                            ImGui::TextWrapped(
                                "Which parts of this model are ground. The "
                                "footprint takes the highest surface over "
                                "each spot, so leaving a canopy on means "
                                "standing on the leaves.");
                            for (size_t mi = 0; mi < spm.mats.size(); mi++) {
                                PropMaterial& mm = spm.mats[mi];
                                ImGui::PushID((int)(1000 + mi));
                                ImGui::Checkbox(mm.name.empty()
                                                    ? "(unnamed)"
                                                    : mm.name.c_str(),
                                                &mm.collide);
                                ImGui::PopID();
                            }
                        }
                        for (int role = 0; role < 3; role++) {
                            PropMaterial* rep = nullptr;
                            for (PropMaterial& mm : spm.mats)
                                if (mat_role(mm.name) == role) {
                                    rep = &mm;
                                    break;
                                }
                            if (!rep)
                                continue;
                            ImGui::PushID(role);
                            ImGui::SeparatorText(kRoleNames[role]);
                            std::string cur = rep->texName;
                            size_t slash = cur.find_last_of("/\\");
                            if (slash != std::string::npos)
                                cur = cur.substr(slash + 1);
                            bool changed = false;
                            if (ImGui::BeginCombo("Texture",
                                    cur.empty() ? "(none)" : cur.c_str())) {
                                for (const std::string& tf : gPropTexFiles)
                                    if (ImGui::Selectable(tf.c_str(),
                                                          tf == cur)) {
                                        cur = tf;
                                        changed = true;
                                    }
                                ImGui::EndCombo();
                            }
                            changed |= ImGui::ColorEdit3("Top", rep->kd,
                                ImGuiColorEditFlags_NoInputs);
                            changed |= ImGui::ColorEdit3("Bottom", rep->ka,
                                ImGuiColorEditFlags_NoInputs);
                            if (changed) {
                                StyleOverride s;
                                s.tex = cur;
                                memcpy(s.kd, rep->kd, sizeof s.kd);
                                memcpy(s.ka, rep->ka, sizeof s.ka);
                                gStyles[mesh_id(spm) + "|" +
                                        kRoleNames[role]] = s;
                                apply_styles(spm);
                                save_styles();
                            }
                            ImGui::PopID();
                        }
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
                if (ImGui::BeginTabItem("World")) {
                    activeTab = 5;
                    ImGui::TextWrapped("Sea chart: assign exported .wmap "
                                       "islands to quadrants. The chart "
                                       "loops at its edges, Wind Waker "
                                       "style.");
                    ImGui::SliderInt("Chart Size", &gWorldSize, 2, WORLD_MAX);
                    ImGui::SliderFloat("Waterline", &gWaterline,
                                       -20.0f, 10.0f, "%.1f");
                    ImGui::SameLine();
                    ImGui::Checkbox("Show", &gShowWater);
                    ImGui::Separator();
                    float cellPx = SDL_min(38.0f * uiScale,
                        (ImGui::GetContentRegionAvail().x - 8.0f) /
                        gWorldSize - 4.0f);
                    for (int cy = 0; cy < gWorldSize; cy++) {
                        for (int cx = 0; cx < gWorldSize; cx++) {
                            if (cx)
                                ImGui::SameLine();
                            char lbl[8];
                            SDL_snprintf(lbl, sizeof lbl, "%c%d",
                                         'A' + cx, cy + 1);
                            bool sel = (gWorldSel[0] == cx &&
                                        gWorldSel[1] == cy);
                            bool has = !gWorldCells[cy][cx].empty();
                            ImVec4 c = has
                                ? ImVec4(0.22f, 0.55f, 0.28f, 1.0f)
                                : ImVec4(0.16f, 0.28f, 0.48f, 1.0f);
                            if (sel)
                                c = ImVec4(0.85f, 0.65f, 0.2f, 1.0f);
                            ImGui::PushStyleColor(ImGuiCol_Button, c);
                            ImGui::PushID(cy * WORLD_MAX + cx);
                            if (ImGui::Button(lbl, ImVec2(cellPx, cellPx))) {
                                // switching quadrants saves what is open,
                                // then opens that cell's island -- or a
                                // clean slate when the cell is empty
                                if ((gWorldSel[0] != cx || gWorldSel[1] != cy) &&
                                    map_has_content()) {
                                    if (gWorldCells[gWorldSel[1]][gWorldSel[0]].empty())
                                        gWorldCells[gWorldSel[1]][gWorldSel[0]] =
                                            slot_path(gWorldSel[0], gWorldSel[1]);
                                    syncSettingsOut();
                                    save_map(gWorldCells[gWorldSel[1]]
                                                        [gWorldSel[0]].c_str());
                                }
                                gWorldSel[0] = cx;
                                gWorldSel[1] = cy;
                                // open the island living here, or a clean
                                // slate -- an empty cell stays empty until
                                // there is something to keep
                                const std::string nxt = gWorldCells[cy][cx];
                                if (!nxt.empty()) {
                                    if (load_map(nxt.c_str()))
                                        applySettingsIn();
                                } else {
                                    new_map();
                                }
                                save_world(editor_chart_path().c_str());
                                rebuild_terrain_mesh();
                                rebuild_grass_instances();
                                glBindBuffer(GL_ARRAY_BUFFER, instVbo);
                                glBufferData(GL_ARRAY_BUFFER,
                                             inst.size() * sizeof(float),
                                             inst.data(), GL_STATIC_DRAW);
                                glBindBuffer(GL_ARRAY_BUFFER, 0);
                                gMapResized = false;
                            }
                            if (has && ImGui::IsItemHovered()) {
                                std::string n = gWorldCells[cy][cx];
                                size_t s = n.find_last_of("/\\");
                                ImGui::SetTooltip("%s",
                                    s == std::string::npos
                                        ? n.c_str() : n.c_str() + s + 1);
                            }
                            ImGui::PopID();
                            ImGui::PopStyleColor();
                        }
                    }
                    ImGui::Separator();
                    {
                        char sl[8];
                        SDL_snprintf(sl, sizeof sl, "%c%d",
                                     'A' + gWorldSel[0], gWorldSel[1] + 1);
                        std::string cur =
                            gWorldCells[gWorldSel[1]][gWorldSel[0]];
                        size_t s = cur.find_last_of("/\\");
                        ImGui::Text("%s: %s", sl,
                                    cur.empty() ? "(open sea)"
                                    : (s == std::string::npos
                                           ? cur.c_str()
                                           : cur.c_str() + s + 1));
                    }
                    if (ImGui::Button("Assign Island..."))
                        SDL_ShowOpenFileDialog(map_dialog_cb, (void*)3, win,
                                               kMapFilters, 1, nullptr,
                                               false);
                    ImGui::SameLine();
                    if (ImGui::Button("Clear Cell")) {
                        // the island on screen, the file behind it and the
                        // chart's reference to it -- dropping the reference
                        // alone left the map open and the .wmap on disk
                        std::string& slot =
                            gWorldCells[gWorldSel[1]][gWorldSel[0]];
                        if (!slot.empty()) {
                            std::error_code ec2;
                            std::filesystem::remove(slot, ec2);
                            slot.clear();
                        }
                        new_map();
                        // Sink the ground rather than flatten it. A zeroed
                        // heightmap is still a floor: it saves into the
                        // map, the client loads it, and you stand on an
                        // invisible plane over open water with your shadow
                        // on it. Down here nothing draws and nothing
                        // carries you, in either program, and the map
                        // format needs no say in it.
                        std::fill(gHeights.begin(), gHeights.end(), -100.0f);
                        gShowGround = false;   // nothing at all, not a floor
                        gGen.on = false;
                        gGenBase.clear();
                        gGenMask.clear(); gGenMask2.clear(); gGenKill.clear();
                        gGenProps.clear();
                        save_world(editor_chart_path().c_str());
                        rebuild_terrain_mesh();
                        rebuild_grass_instances();
                        glBindBuffer(GL_ARRAY_BUFFER, instVbo);
                        glBufferData(GL_ARRAY_BUFFER,
                                     inst.size() * sizeof(float),
                                     inst.data(), GL_STATIC_DRAW);
                        glBindBuffer(GL_ARRAY_BUFFER, 0);
                        SDL_Log("cleared %c%d", 'A' + gWorldSel[0],
                                gWorldSel[1] + 1);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button(gShowGround ? "Hide Plane"
                                                  : "Add Plane")) {
                        gShowGround = !gShowGround;
                        if (gShowGround) {
                            for (float& h : gHeights)
                                if (h < -50.0f)
                                    h = 0.0f;
                            gShoreBase.clear();
                            mark_all_dirty();
                        }
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("A quadrant built out of props "
                                          "wants no ground under it.\nPut a "
                                          "plane back when you want to "
                                          "sculpt again.");
                    if (ImGui::Button("Reset Editor")) {
                        // Everything the editor remembers between runs:
                        // the chart, every island slot it owns, and the
                        // working map. Without this a deleted quadrant
                        // keeps coming back, because the chart is reloaded
                        // at startup and still points at what it knew.
                        std::error_code ec3;
                        for (int y = 0; y < WORLD_MAX; y++)
                            for (int x = 0; x < WORLD_MAX; x++) {
                                if (!gWorldCells[y][x].empty())
                                    std::filesystem::remove(gWorldCells[y][x],
                                                            ec3);
                                gWorldCells[y][x].clear();
                            }
                        std::filesystem::remove_all(
                            std::string(SDL_GetBasePath()) + "islands", ec3);
                        std::filesystem::remove(editor_chart_path(), ec3);
                        std::filesystem::remove(
                            std::string(SDL_GetBasePath()) + "../map.bin", ec3);
                        gWorldSel[0] = gWorldSel[1] = 0;
                        gSpawnCell[0] = gSpawnCell[1] = -1;
                        new_map();
                        std::fill(gHeights.begin(), gHeights.end(), -100.0f);
                        gShowGround = false;
                        gGen.on = false;
                        gGenBase.clear();
                        gGenMask.clear(); gGenMask2.clear(); gGenKill.clear();
                        gGenProps.clear();
                        rebuild_terrain_mesh();
                        rebuild_grass_instances();
                        glBindBuffer(GL_ARRAY_BUFFER, instVbo);
                        glBufferData(GL_ARRAY_BUFFER,
                                     inst.size() * sizeof(float),
                                     inst.data(), GL_STATIC_DRAW);
                        glBindBuffer(GL_ARRAY_BUFFER, 0);
                        SDL_Log("editor reset: chart, slots and map.bin gone");
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Forgets everything between runs: "
                                          "the chart, every island slot and "
                                          "the working map.\nDeletes those "
                                          "files. Islands saved elsewhere "
                                          "are left alone.");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Empties the quadrant: the island, "
                                          "its file and its place on the "
                                          "chart.\nThis deletes the .wmap on "
                                          "disk.");
                    ImGui::SeparatorText("Send to Game");
                    if (ImGui::Button("Install Chart to Game")) {
                        namespace fs = std::filesystem;
                        std::error_code ec;
                        // walk up from the editor to the repo, then into
                        // the client's exe folder -- that is where it
                        // looks for a chart
                        fs::path d = SDL_GetBasePath();
                        std::string dest;
                        for (int up = 0; up < 8; up++) {
                            fs::path cand = d / "SDL3" / "build" / "Release";
                            if (fs::exists(cand / "zelda.exe", ec)) {
                                dest = cand.string();
                                break;
                            }
                            if (d.parent_path() == d) break;
                            d = d.parent_path();
                        }
                        if (dest.empty()) {
                            SDL_Log("install: could not find zelda.exe");
                        } else {
                            // Export the island itself alongside the chart
                            // and point the selected quadrant at it, so the
                            // game derives its shoreline, foam and collision
                            // from exactly what is on screen here -- the
                            // current waterline, skirt and sculpt.
                            syncSettingsOut();
                            char isle[700];
                            SDL_snprintf(isle, sizeof isle,
                                         "%s/island_%c%d.wmap", dest.c_str(),
                                         'A' + gWorldSel[0], gWorldSel[1] + 1);
                            save_map(isle);
                            gWorldCells[gWorldSel[1]][gWorldSel[0]] = isle;
                            // one chart only, or the client picks whichever
                            // it happens to see first
                            for (const auto& e :
                                 fs::directory_iterator(dest, ec))
                                if (e.path().extension() == ".wworld")
                                    fs::remove(e.path(), ec);
                            save_world((dest + "/world.wworld").c_str());
                            SDL_Log("installed chart to %s", dest.c_str());
                        }
                    }
                    ImGui::TextDisabled("writes the chart next to zelda.exe");
                    if (ImGui::Button("Save Island to Cell")) {
                        // explicit version of what switching quadrants
                        // does, for when you are staying put
                        if (gWorldCells[gWorldSel[1]][gWorldSel[0]].empty())
                            gWorldCells[gWorldSel[1]][gWorldSel[0]] =
                                slot_path(gWorldSel[0], gWorldSel[1]);
                        syncSettingsOut();
                        save_map(gWorldCells[gWorldSel[1]][gWorldSel[0]].c_str());
                        save_world(editor_chart_path().c_str());
                    }
                    {
                        bool isSpawn = gSpawnCell[0] == gWorldSel[0] &&
                                       gSpawnCell[1] == gWorldSel[1];
                        char lbl[64];
                        SDL_snprintf(lbl, sizeof lbl,
                                     isSpawn ? "Start Cell: %c%d (set)"
                                             : "Start Here (%c%d)",
                                     'A' + gWorldSel[0], gWorldSel[1] + 1);
                        if (ImGui::Button(lbl)) {
                            gSpawnCell[0] = gWorldSel[0];
                            gSpawnCell[1] = gWorldSel[1];
                            save_world(editor_chart_path().c_str());
                        }
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip(
                                "The game drops you into this quadrant "
                                "instead of its usual start.\nSaved into "
                                "the chart, so Install Chart to Game "
                                "carries it across.");
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("Writes the open island into the "
                                          "selected quadrant's slot.\nThe "
                                          "chart only stores paths, so an "
                                          "island has to be saved before a "
                                          "world knows about it.");
                    ImGui::SeparatorText("World File");
                    if (ImGui::Button("Save World..."))
                        SDL_ShowSaveFileDialog(map_dialog_cb, (void*)4, win,
                                               kWorldFilters, 1, nullptr);
                    ImGui::SameLine();
                    if (ImGui::Button("Load World..."))
                        SDL_ShowOpenFileDialog(map_dialog_cb, (void*)5, win,
                                               kWorldFilters, 1, nullptr,
                                               false);
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
                        if (gMapResized) {
                            gMapResized = false;
                            rebuild_terrain_mesh();
                            rebuild_grass_instances();
                            glBindBuffer(GL_ARRAY_BUFFER, instVbo);
                            glBufferData(GL_ARRAY_BUFFER,
                                         inst.size() * sizeof(float),
                                         inst.data(), GL_STATIC_DRAW);
                            glBindBuffer(GL_ARRAY_BUFFER, 0);
                        }
                    }
                    ImGui::Checkbox("Shadows", &shadowsOn);
                    ImGui::SliderFloat("Sun Shadow Intensity",
                                       &shadowStrength, 0.0f, 1.0f, "%.2f");
                    ImGui::SeparatorText("Detail");
                    {
                        float dm = gDetailMult;
                        ImGui::SliderFloat("Terrain Detail", &dm, 0.5f, 4.0f,
                                           "x%.2f");
                        if (ImGui::IsItemDeactivatedAfterEdit() &&
                            fabsf(dm - gDetailMult) > 0.001f) {
                            gDetailMult = dm;
                            apply_detail();
                            refresh_if_resized();
                        }
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip(
                                "More cells for the same world, so cuts and "
                                "edges come out sharper.\nThe grid is "
                                "uniform, so this cannot be spent only where "
                                "it is needed -- it raises detail "
                                "everywhere, and costs memory and frame "
                                "time to match.");
                        ImGui::TextDisabled("grid %d, heights %d, masks %d",
                                            GRID_N, HN, MASK_N);
                    }
                    ImGui::SeparatorText("Map Size");
                    {
                        static const char* kSizes[] = { "48 (small)",
                                                        "96 (medium)",
                                                        "144 (large)" };
                        int cur = TER_HALF <= 25.0f ? 0
                                : TER_HALF <= 49.0f ? 1 : 2;
                        int pick = cur;
                        ImGui::SetNextItemWidth(150 * uiScale);
                        if (ImGui::Combo("World Size", &pick, kSizes, 3) &&
                            pick != cur) {
                            const float halves[3] = { 24.0f, 48.0f, 72.0f };
                            apply_map_resize(halves[pick]);
                        }
                        ImGui::TextDisabled("%.0f units across - %d quads, "
                                            "%d heights", TER_HALF * 2.0f,
                                            GRID_N, HN);
                        ImGui::TextDisabled("resampled to keep detail per "
                                            "unit (clears undo)");
                    }
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
