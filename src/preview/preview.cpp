// Preview window: an auxiliary native window (WindowService) with our own WebGPU renderer
// (GfxService present target). Rendering code runs on the render worker thread; everything it
// needs arrives in a small payload or via the mutex-protected snapshot in core.cpp.
#include "preview/preview.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "preview/math.hpp"
#include "shared/common.hpp"
#include "shared/core.hpp"
#include "mods/svc/gfx.h"
#include "mods/svc/window.h"
#include <webgpu/webgpu.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace mi {
namespace {

// ---------------------------------------------------------------------------------------------
// Shader
// ---------------------------------------------------------------------------------------------
const char* kShader = R"WGSL(
struct U {
  view_proj : mat4x4<f32>,
  p0 : vec4<f32>,     // x: textured; y: opaque or blended texture pass
  uvxf : vec4<f32>,   // uv -> ndc: xy scale, zw offset
  color : vec4<f32>,  // wire colour
};
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var tex : texture_2d<f32>;
@group(0) @binding(2) var smp : sampler;

struct VIn {
  @location(0) pos : vec3<f32>,
  @location(1) uv : vec2<f32>,
  @location(2) id : f32,
};

fn hash_color(id : f32) -> vec3<f32> {
  let a = fract(sin(id * 12.9898 + 78.233) * 43758.5453);
  let b = fract(sin(id * 39.346 + 11.135) * 24634.6345);
  let c = fract(sin(id * 73.156 + 52.235) * 12345.6789);
  return vec3<f32>(0.35 + 0.55 * a, 0.35 + 0.55 * b, 0.35 + 0.55 * c);
}

// ---- 3D solid ----
struct MeshOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) world : vec3<f32>,
  @location(2) id : f32,
};

@vertex fn vs_mesh(v : VIn) -> MeshOut {
  var o : MeshOut;
  o.clip = u.view_proj * vec4<f32>(v.pos, 1.0);
  o.uv = v.uv;
  o.world = v.pos;
  o.id = v.id;
  return o;
}

@fragment fn fs_mesh(i : MeshOut) -> @location(0) vec4<f32> {
  let t = textureSample(tex, smp, i.uv);
  if (u.p0.x > 0.5) {
    if (u.p0.y > 1.5) {
      if (t.a <= 0.0 || t.a >= 1.0) {
        discard;
      }
      return t;
    }
    if (u.p0.y > 0.5) {
      if (t.a < 1.0) {
        discard;
      }
      return vec4<f32>(t.rgb, 1.0);
    }
    return vec4<f32>(t.rgb, 1.0);
  }
  let dx = dpdx(i.world);
  let dy = dpdy(i.world);
  let n = normalize(cross(dx, dy));
  let l = normalize(vec3<f32>(0.35, 0.8, 0.5));
  let shade = 0.35 + 0.65 * abs(dot(n, l));
  return vec4<f32>(hash_color(i.id) * shade, 1.0);
}

// ---- 3D wire overlay ----
@vertex fn vs_wire(v : VIn) -> @builtin(position) vec4<f32> {
  return u.view_proj * vec4<f32>(v.pos, 1.0);
}

@fragment fn fs_wire() -> @location(0) vec4<f32> {
  return u.color;
}

// ---- UV map ----
struct UvOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) id : f32,
};

@vertex fn vs_uvwire(v : VIn) -> UvOut {
  var o : UvOut;
  o.clip = vec4<f32>(v.uv * u.uvxf.xy + u.uvxf.zw, 0.0, 1.0);
  o.id = v.id;
  return o;
}

@fragment fn fs_uvwire(i : UvOut) -> @location(0) vec4<f32> {
  return vec4<f32>(hash_color(i.id) * 0.6 + vec3<f32>(0.4), 1.0);
}

struct QuadOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex fn vs_quad(v : VIn) -> QuadOut {
  var o : QuadOut;
  o.clip = vec4<f32>(v.pos.xy * u.uvxf.xy + u.uvxf.zw, 0.0, 1.0);
  o.uv = v.uv;
  return o;
}

@fragment fn fs_quad(i : QuadOut) -> @location(0) vec4<f32> {
  let t = textureSample(tex, smp, i.uv);
  return vec4<f32>(t.rgb, 1.0);
}

struct OverlayIn {
  @location(0) pos : vec2<f32>,
  @location(1) color : vec4<f32>,
};

struct OverlayOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) color : vec4<f32>,
};

@vertex fn vs_overlay(v : OverlayIn) -> OverlayOut {
  var o : OverlayOut;
  o.clip = vec4<f32>(v.pos, 0.0, 1.0);
  o.color = v.color;
  return o;
}

@fragment fn fs_overlay(i : OverlayOut) -> @location(0) vec4<f32> {
  return i.color;
}
)WGSL";

constexpr uint32_t kUniformSize = 128;
constexpr uint32_t kVertexStride = 24;
struct Uniforms {
    float viewProj[16];
    float p0[4];
    float uvxf[4];
    float color[4];
};
static_assert(sizeof(Uniforms) <= kUniformSize);

struct PresentPayload {
    uint64_t version;
    float yaw;    // degrees
    float pitch;  // degrees
    float zoom;   // factor, 1 = fit
    uint32_t mode;
    uint32_t flags;  // bit0: nearest filtering
    int32_t solo;
    int32_t uvTex;
    uint32_t upAxis;
    uint32_t layout;
    int32_t hoverControl;
    float flyPosition[3];
    float uvPanX;
    float uvPanY;
    int32_t uvTileOffsetX;
    int32_t uvTileOffsetY;
    float uvZoom;
    int32_t texturePage;
    int32_t shapePage;
    uint32_t flyingCamera;
    uint32_t textureMenuOpen;
    uint32_t shapeMenuOpen;
    uint32_t liveUpdate;
};
static_assert(sizeof(PresentPayload) <= GFX_INLINE_DRAW_PAYLOAD_SIZE);

struct OverlayVertex {
    float pos[2];
    float color[4];
};
static_assert(sizeof(OverlayVertex) == 24);

// ---------------------------------------------------------------------------------------------
// Window state (game thread)
// ---------------------------------------------------------------------------------------------
WindowHandle g_window = 0;
GfxPresentTargetHandle g_target = 0;
bool g_recreateTarget = false;
bool g_previousF = false;
bool g_previousLeftMouse = false;
int g_dragMode = 0; // 0 none, 1 orbit, 2 zoom
int g_previousMouseX = 0;
int g_previousMouseY = 0;
int32_t g_hoverControl = -1;
std::chrono::steady_clock::time_point g_lastInputTime{};
std::chrono::steady_clock::time_point g_nextLiveSnapshot{};
PreviewExportHandler g_exportHandler = nullptr;

// ---------------------------------------------------------------------------------------------
// GPU state (render worker thread; released from the game thread at shutdown)
// ---------------------------------------------------------------------------------------------
struct Gpu {
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUTextureFormat colorFormat = WGPUTextureFormat_Undefined;

    WGPUShaderModule module = nullptr;
    WGPURenderPipeline pTri = nullptr, pTriAlpha = nullptr, pWire = nullptr, pUvBg = nullptr,
        pUvWire = nullptr, pOverlay = nullptr;
    WGPUBindGroupLayout lTri = nullptr, lTriAlpha = nullptr, lWire = nullptr, lUvBg = nullptr,
        lUvWire = nullptr;

    WGPUBuffer ubuf = nullptr, alphaUbuf = nullptr, uvUbuf = nullptr, uvWireUbuf = nullptr;
    WGPUBuffer overlayBuffer = nullptr;
    uint64_t overlayBufferSize = 0;
    WGPUSampler sampLinear = nullptr, sampNearest = nullptr;
    WGPUTexture texGray = nullptr, texDark = nullptr;
    WGPUTextureView viewGray = nullptr, viewDark = nullptr;

    WGPUTexture depth = nullptr;
    WGPUTextureView depthView = nullptr;
    uint32_t depthW = 0, depthH = 0;

    uint64_t meshVersion = 0;
    std::shared_ptr<const MeshData> mesh;
    WGPUBuffer vbuf = nullptr, uvBgVbuf = nullptr, ibuf = nullptr;
    uint64_t vbufSize = 0, ibufSize = 0;
    uint64_t uvBgVbufSize = 0;
    std::vector<WGPUTexture> texs;
    std::vector<WGPUTextureView> views;
};
Gpu g;

void rel(WGPUTextureView& v) { if (v) { wgpuTextureViewRelease(v); v = nullptr; } }
void rel(WGPUTexture& t) { if (t) { wgpuTextureRelease(t); t = nullptr; } }
void rel(WGPUBuffer& b) { if (b) { wgpuBufferRelease(b); b = nullptr; } }
void rel(WGPUSampler& s) { if (s) { wgpuSamplerRelease(s); s = nullptr; } }
void rel(WGPURenderPipeline& p) { if (p) { wgpuRenderPipelineRelease(p); p = nullptr; } }
void rel(WGPUBindGroupLayout& l) { if (l) { wgpuBindGroupLayoutRelease(l); l = nullptr; } }
void rel(WGPUShaderModule& m) { if (m) { wgpuShaderModuleRelease(m); m = nullptr; } }

void release_mesh_gpu() {
    rel(g.vbuf);
    rel(g.uvBgVbuf);
    rel(g.ibuf);
    for (auto& v : g.views) rel(v);
    for (auto& t : g.texs) rel(t);
    g.views.clear();
    g.texs.clear();
    g.vbufSize = g.uvBgVbufSize = g.ibufSize = 0;
    g.mesh.reset();
    g.meshVersion = 0;
}

void release_pipelines() {
    rel(g.pTri); rel(g.pTriAlpha); rel(g.pWire); rel(g.pUvBg); rel(g.pUvWire); rel(g.pOverlay);
    rel(g.lTri); rel(g.lTriAlpha); rel(g.lWire); rel(g.lUvBg); rel(g.lUvWire);
    rel(g.module);
    g.colorFormat = WGPUTextureFormat_Undefined;
}

void release_all_gpu() {
    release_mesh_gpu();
    release_pipelines();
    rel(g.depthView);
    rel(g.depth);
    g.depthW = g.depthH = 0;
    rel(g.viewGray); rel(g.viewDark);
    rel(g.texGray); rel(g.texDark);
    rel(g.sampLinear); rel(g.sampNearest);
    rel(g.overlayBuffer);
    g.overlayBufferSize = 0;
    rel(g.ubuf);
    rel(g.alphaUbuf);
    rel(g.uvUbuf);
    rel(g.uvWireUbuf);
    g.device = nullptr;
    g.queue = nullptr;
}

bool create_rgba(uint32_t w, uint32_t h, const uint8_t* px, WGPUTexture& tex, WGPUTextureView& view) {
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    td.label = {"inspector texture", WGPU_STRLEN};
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    td.dimension = WGPUTextureDimension_2D;
    td.size = {w, h, 1};
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    tex = wgpuDeviceCreateTexture(g.device, &td);
    if (tex == nullptr) return false;
    WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dst.texture = tex;
    dst.mipLevel = 0;
    dst.origin = {0, 0, 0};
    dst.aspect = WGPUTextureAspect_All;
    WGPUTexelCopyBufferLayout lay = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    lay.offset = 0;
    lay.bytesPerRow = w * 4;
    lay.rowsPerImage = h;
    const WGPUExtent3D ext = {w, h, 1};
    wgpuQueueWriteTexture(g.queue, &dst, px, static_cast<size_t>(w) * h * 4, &lay, &ext);
    view = wgpuTextureCreateView(tex, nullptr);
    return view != nullptr;
}

WGPUSampler make_sampler(bool nearest) {
    WGPUSamplerDescriptor sd = WGPU_SAMPLER_DESCRIPTOR_INIT;
    sd.addressModeU = WGPUAddressMode_Repeat;
    sd.addressModeV = WGPUAddressMode_Repeat;
    sd.addressModeW = WGPUAddressMode_Repeat;
    sd.magFilter = nearest ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
    sd.minFilter = nearest ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
    sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    sd.maxAnisotropy = 1;
    return wgpuDeviceCreateSampler(g.device, &sd);
}

WGPURenderPipeline make_pipeline(const char* label, const char* vs, const char* fs,
    WGPUPrimitiveTopology topo, bool vertexBuffer, bool depth, bool depthWrite,
    bool alphaBlend = false) {
    WGPUVertexAttribute attrs[3] = {
        WGPU_VERTEX_ATTRIBUTE_INIT, WGPU_VERTEX_ATTRIBUTE_INIT, WGPU_VERTEX_ATTRIBUTE_INIT};
    attrs[0].format = WGPUVertexFormat_Float32x3;
    attrs[0].offset = 0;
    attrs[0].shaderLocation = 0;
    attrs[1].format = WGPUVertexFormat_Float32x2;
    attrs[1].offset = 12;
    attrs[1].shaderLocation = 1;
    attrs[2].format = WGPUVertexFormat_Float32;
    attrs[2].offset = 20;
    attrs[2].shaderLocation = 2;
    WGPUVertexBufferLayout vbl = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
    vbl.stepMode = WGPUVertexStepMode_Vertex;
    vbl.arrayStride = kVertexStride;
    vbl.attributeCount = 3;
    vbl.attributes = attrs;

    WGPUColorTargetState ct = WGPU_COLOR_TARGET_STATE_INIT;
    ct.format = g.colorFormat;
    WGPUBlendState blend = WGPU_BLEND_STATE_INIT;
    if (alphaBlend) {
        blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
        blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blend.color.operation = WGPUBlendOperation_Add;
        blend.alpha.srcFactor = WGPUBlendFactor_One;
        blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blend.alpha.operation = WGPUBlendOperation_Add;
        ct.blend = &blend;
    }
    WGPUFragmentState frag = WGPU_FRAGMENT_STATE_INIT;
    frag.module = g.module;
    frag.entryPoint = {fs, WGPU_STRLEN};
    frag.targetCount = 1;
    frag.targets = &ct;

    WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
    ds.format = WGPUTextureFormat_Depth32Float;
    ds.depthWriteEnabled = depthWrite ? WGPUOptionalBool_True : WGPUOptionalBool_False;
    ds.depthCompare = WGPUCompareFunction_LessEqual;

    WGPURenderPipelineDescriptor pd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pd.label = {label, WGPU_STRLEN};
    pd.vertex.module = g.module;
    pd.vertex.entryPoint = {vs, WGPU_STRLEN};
    if (vertexBuffer) {
        pd.vertex.bufferCount = 1;
        pd.vertex.buffers = &vbl;
    }
    pd.primitive.topology = topo;
    pd.primitive.cullMode = WGPUCullMode_None;
    pd.multisample.count = 1;
    pd.fragment = &frag;
    if (depth) pd.depthStencil = &ds;
    return wgpuDeviceCreateRenderPipeline(g.device, &pd);
}

WGPURenderPipeline make_overlay_pipeline() {
    WGPUVertexAttribute attrs[2] = {WGPU_VERTEX_ATTRIBUTE_INIT, WGPU_VERTEX_ATTRIBUTE_INIT};
    attrs[0].format = WGPUVertexFormat_Float32x2;
    attrs[0].offset = 0;
    attrs[0].shaderLocation = 0;
    attrs[1].format = WGPUVertexFormat_Float32x4;
    attrs[1].offset = 8;
    attrs[1].shaderLocation = 1;
    WGPUVertexBufferLayout vbl = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
    vbl.stepMode = WGPUVertexStepMode_Vertex;
    vbl.arrayStride = sizeof(OverlayVertex);
    vbl.attributeCount = 2;
    vbl.attributes = attrs;

    WGPUColorTargetState ct = WGPU_COLOR_TARGET_STATE_INIT;
    ct.format = g.colorFormat;
    WGPUFragmentState frag = WGPU_FRAGMENT_STATE_INIT;
    frag.module = g.module;
    frag.entryPoint = {"fs_overlay", WGPU_STRLEN};
    frag.targetCount = 1;
    frag.targets = &ct;

    WGPURenderPipelineDescriptor pd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pd.label = {"inspector controls", WGPU_STRLEN};
    pd.vertex.module = g.module;
    pd.vertex.entryPoint = {"vs_overlay", WGPU_STRLEN};
    pd.vertex.bufferCount = 1;
    pd.vertex.buffers = &vbl;
    pd.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pd.primitive.cullMode = WGPUCullMode_None;
    pd.multisample.count = 1;
    pd.fragment = &frag;
    return wgpuDeviceCreateRenderPipeline(g.device, &pd);
}

bool ensure_gpu(const GfxPresentContext& c) {
    if (g.device != c.device) {
        // first use (or device change): rebuild everything
        release_all_gpu();
        g.device = c.device;
        g.queue = c.queue;
    }
    if (g.ubuf == nullptr || g.alphaUbuf == nullptr || g.uvUbuf == nullptr ||
        g.uvWireUbuf == nullptr ||
        g.sampLinear == nullptr || g.sampNearest == nullptr ||
        g.viewGray == nullptr || g.viewDark == nullptr) {
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.label = {"inspector uniforms", WGPU_STRLEN};
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        bd.size = kUniformSize;
        if (g.ubuf == nullptr) g.ubuf = wgpuDeviceCreateBuffer(g.device, &bd);
        bd.label = {"inspector alpha uniforms", WGPU_STRLEN};
        if (g.alphaUbuf == nullptr) g.alphaUbuf = wgpuDeviceCreateBuffer(g.device, &bd);
        bd.label = {"inspector UV uniforms", WGPU_STRLEN};
        if (g.uvUbuf == nullptr) g.uvUbuf = wgpuDeviceCreateBuffer(g.device, &bd);
        bd.label = {"inspector UV wire uniforms", WGPU_STRLEN};
        if (g.uvWireUbuf == nullptr) g.uvWireUbuf = wgpuDeviceCreateBuffer(g.device, &bd);
        if (g.sampLinear == nullptr) g.sampLinear = make_sampler(false);
        if (g.sampNearest == nullptr) g.sampNearest = make_sampler(true);
        const uint8_t gray[4] = {200, 200, 200, 255};
        const uint8_t dark[4] = {46, 46, 54, 255};
        if (g.texGray == nullptr || g.viewGray == nullptr) {
            rel(g.viewGray);
            rel(g.texGray);
            create_rgba(1, 1, gray, g.texGray, g.viewGray);
        }
        if (g.texDark == nullptr || g.viewDark == nullptr) {
            rel(g.viewDark);
            rel(g.texDark);
            create_rgba(1, 1, dark, g.texDark, g.viewDark);
        }
        if (g.ubuf == nullptr || g.alphaUbuf == nullptr || g.uvUbuf == nullptr ||
            g.uvWireUbuf == nullptr ||
            g.sampLinear == nullptr || g.sampNearest == nullptr ||
            g.viewGray == nullptr || g.viewDark == nullptr) return false;
    }
    if (g.pTri != nullptr && g.pTriAlpha != nullptr && g.pOverlay != nullptr &&
        g.colorFormat == c.target_format) return true;

    release_pipelines();
    g.colorFormat = c.target_format;
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = {kShader, WGPU_STRLEN};
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &wgsl.chain;
    md.label = {"inspector shader", WGPU_STRLEN};
    g.module = wgpuDeviceCreateShaderModule(g.device, &md);
    if (g.module == nullptr) return false;

    g.pTri = make_pipeline("inspector solid", "vs_mesh", "fs_mesh", WGPUPrimitiveTopology_TriangleList,
        true, true, true);
    g.pTriAlpha = make_pipeline("inspector textured alpha", "vs_mesh", "fs_mesh",
        WGPUPrimitiveTopology_TriangleList, true, true, false, true);
    g.pWire = make_pipeline("inspector wire", "vs_wire", "fs_wire", WGPUPrimitiveTopology_LineList,
        true, true, false, true);
    g.pUvBg = make_pipeline("inspector uv bg", "vs_quad", "fs_quad",
        WGPUPrimitiveTopology_TriangleList, true, false, false);
    g.pUvWire = make_pipeline("inspector uv wire", "vs_uvwire", "fs_uvwire",
        WGPUPrimitiveTopology_LineList, true, false, false);
    g.pOverlay = make_overlay_pipeline();
    if (!g.pTri || !g.pTriAlpha || !g.pWire || !g.pUvBg || !g.pUvWire || !g.pOverlay) {
        logf(2, "FAILED to create WebGPU pipelines!");
        release_pipelines();
        return false;
    }
    g.lTri = wgpuRenderPipelineGetBindGroupLayout(g.pTri, 0);
    g.lTriAlpha = wgpuRenderPipelineGetBindGroupLayout(g.pTriAlpha, 0);
    g.lWire = wgpuRenderPipelineGetBindGroupLayout(g.pWire, 0);
    g.lUvBg = wgpuRenderPipelineGetBindGroupLayout(g.pUvBg, 0);
    g.lUvWire = wgpuRenderPipelineGetBindGroupLayout(g.pUvWire, 0);
    return true;
}

void ensure_depth(uint32_t w, uint32_t h) {
    if (g.depth != nullptr && g.depthW == w && g.depthH == h) return;
    rel(g.depthView);
    rel(g.depth);
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    td.label = {"inspector depth", WGPU_STRLEN};
    td.usage = WGPUTextureUsage_RenderAttachment;
    td.dimension = WGPUTextureDimension_2D;
    td.size = {w, h, 1};
    td.format = WGPUTextureFormat_Depth32Float;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    g.depth = wgpuDeviceCreateTexture(g.device, &td);
    if (g.depth != nullptr) g.depthView = wgpuTextureCreateView(g.depth, nullptr);
    g.depthW = w;
    g.depthH = h;
}

void sync_mesh() {
    if (mesh_version() == g.meshVersion) return;
    std::shared_ptr<const MeshData> mesh = current_mesh();
    if (mesh != nullptr && g.mesh != nullptr && g.vbuf != nullptr &&
        g.mesh->key == mesh->key && g.mesh->dataKey == mesh->dataKey &&
        g.mesh->textures == mesh->textures && g.mesh->vertices.size() == mesh->vertices.size() &&
        g.mesh->vertexCount == mesh->vertexCount && g.mesh->subs.size() == mesh->subs.size() &&
        g.vbufSize == static_cast<uint64_t>(mesh->vertices.size()) * sizeof(float)) {
        bool sameLayout = true;
        for (size_t i = 0; i < mesh->subs.size(); ++i) {
            const SubMesh& oldSub = g.mesh->subs[i];
            const SubMesh& newSub = mesh->subs[i];
            if (oldSub.firstVertex != newSub.firstVertex || oldSub.vertexCount != newSub.vertexCount ||
                oldSub.texture != newSub.texture || oldSub.shapeIndex != newSub.shapeIndex) {
                sameLayout = false;
                break;
            }
        }
        if (sameLayout) {
            const uint64_t bytes = static_cast<uint64_t>(mesh->vertices.size()) * sizeof(float);
            wgpuQueueWriteBuffer(g.queue, g.vbuf, 0, mesh->vertices.data(), bytes);
            g.mesh = std::move(mesh);
            g.meshVersion = g.mesh->version;
            return;
        }
    }
    release_mesh_gpu();
    if (!mesh) return;
    g.mesh = mesh;
    g.meshVersion = mesh->version;
    const uint64_t vbytes = static_cast<uint64_t>(mesh->vertices.size()) * sizeof(float);
    if (vbytes == 0) return;

    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.label = {"inspector vertices", WGPU_STRLEN};
    bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    bd.size = vbytes;
    g.vbuf = wgpuDeviceCreateBuffer(g.device, &bd);
    g.vbufSize = vbytes;
    if (g.vbuf != nullptr) {
        wgpuQueueWriteBuffer(g.queue, g.vbuf, 0, mesh->vertices.data(), vbytes);
    }

    // wire indices: 3 edges per triangle
    const uint32_t triCount = mesh->vertexCount / 3;
    std::vector<uint32_t> idx;
    idx.reserve(static_cast<size_t>(triCount) * 6);
    for (uint32_t t = 0; t < triCount; ++t) {
        const uint32_t a = t * 3;
        idx.insert(idx.end(), {a, a + 1, a + 1, a + 2, a + 2, a});
    }
    const uint64_t ibytes = static_cast<uint64_t>(idx.size()) * sizeof(uint32_t);
    if (ibytes > 0) {
        WGPUBufferDescriptor ib = WGPU_BUFFER_DESCRIPTOR_INIT;
        ib.label = {"inspector wire indices", WGPU_STRLEN};
        ib.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
        ib.size = ibytes;
        g.ibuf = wgpuDeviceCreateBuffer(g.device, &ib);
        g.ibufSize = ibytes;
        if (g.ibuf != nullptr) wgpuQueueWriteBuffer(g.queue, g.ibuf, 0, idx.data(), ibytes);
    }

    g.texs.assign(mesh->textures->size(), nullptr);
    g.views.assign(mesh->textures->size(), nullptr);
    for (size_t i = 0; i < mesh->textures->size(); ++i) {
        const TexImage& t = (*mesh->textures)[i];
        if (t.w == 0 || t.h == 0 || t.rgba.empty()) continue;
        create_rgba(t.w, t.h, t.rgba.data(), g.texs[i], g.views[i]);
    }
}

WGPUBindGroup make_bg(WGPUBindGroupLayout layout, WGPUTextureView view, WGPUSampler samp,
    WGPUBuffer uniformBuffer = nullptr) {
    WGPUBindGroupEntry e[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
        WGPU_BIND_GROUP_ENTRY_INIT};
    e[0].binding = 0;
    e[0].buffer = uniformBuffer != nullptr ? uniformBuffer : g.ubuf;
    e[0].offset = 0;
    e[0].size = kUniformSize;
    uint32_t n = 1;
    if (view != nullptr) {
        e[1].binding = 1;
        e[1].textureView = view;
        e[2].binding = 2;
        e[2].sampler = samp;
        n = 3;
    }
    WGPUBindGroupDescriptor d = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    d.layout = layout;
    d.entryCount = n;
    d.entries = e;
    return wgpuDeviceCreateBindGroup(g.device, &d);
}

WGPUTextureView view_for(int32_t tex, WGPUTextureView fallback) {
    if (tex >= 0 && static_cast<size_t>(tex) < g.views.size() && g.views[tex] != nullptr) {
        return g.views[tex];
    }
    return fallback;
}

bool sub_visible(const SubMesh& s, int32_t solo) { return solo < 0 || static_cast<int32_t>(s.shapeIndex) == solo; }

int32_t default_uv_texture(const MeshData& mesh, int32_t solo) {
    if (solo >= 0) {
        for (const SubMesh& s : mesh.subs) {
            if (static_cast<int32_t>(s.shapeIndex) == solo && view_for(s.texture, nullptr) != nullptr) {
                return s.texture;
            }
        }
    }
    for (const SubMesh& s : mesh.subs) {
        if (s.texture >= 0 && view_for(s.texture, nullptr) != nullptr) return s.texture;
    }
    for (size_t i = 0; i < g.views.size(); ++i) {
        if (g.views[i] != nullptr) return static_cast<int32_t>(i);
    }
    return -1;
}

void draw_wire(WGPURenderPassEncoder pass, WGPUBindGroup bg, int32_t solo,
    WGPUBuffer vertexBuffer, uint64_t vertexBufferSize) {
    if (g.mesh == nullptr || vertexBuffer == nullptr || g.ibuf == nullptr) return;
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertexBuffer, 0, vertexBufferSize);
    wgpuRenderPassEncoderSetIndexBuffer(pass, g.ibuf, WGPUIndexFormat_Uint32, 0, g.ibufSize);
    for (const SubMesh& s : g.mesh->subs) {
        if (!sub_visible(s, solo)) continue;
        wgpuRenderPassEncoderDrawIndexed(pass, s.vertexCount * 2, 1, s.firstVertex * 2, 0, 0);
    }
}

constexpr const char* kToolbarModes[] = {"SOLID", "TEX", "SOLID+WIRE", "TEX+WIRE", "WIRE"};
constexpr float kToolbarModeWidths[] = {72.0f, 58.0f, 126.0f, 116.0f, 58.0f};
constexpr float kToolbarAxisWidth = 74.0f;
constexpr float kToolbarFlyWidth = 52.0f;
constexpr float kToolbarTextureWidth = 142.0f;
constexpr float kToolbarLiveWidth = 36.0f;
constexpr float kToolbarExportWidth = 88.0f;
constexpr float kToolbarShapeWidth = 100.0f;
constexpr float kToolbarGap = 5.0f;
constexpr int32_t kLayout2DControl = 20;
constexpr int32_t kLayout3DControl = 21;
constexpr int32_t kLayoutBothControl = 22;
constexpr int32_t kFlyControl = 6;
constexpr int32_t kAxisControl = 7;
constexpr int32_t kTextureControl = 8;
constexpr int32_t kTexturePrevControl = 9;
constexpr int32_t kTextureNextControl = 10;
constexpr int32_t kLiveControl = 11;
constexpr int32_t kUvTileUMinusControl = 12;
constexpr int32_t kUvTileUPlusControl = 13;
constexpr int32_t kUvTileVMinusControl = 14;
constexpr int32_t kUvTileVPlusControl = 15;
constexpr int32_t kShapeControl = 16;
constexpr int32_t kShapePrevControl = 17;
constexpr int32_t kShapeNextControl = 18;
constexpr int32_t kExportControl = 19;
constexpr int32_t kTextureRowFirstControl = 100;
constexpr int32_t kShapeRowFirstControl = 200;
constexpr int32_t kTextureRowsPerPage = 8;

struct ToolbarButton {
    float x, y, w, h;
    int32_t id;
};

std::vector<ToolbarButton> toolbar_buttons(float width, PreviewLayout layout, uint32_t mode) {
    std::vector<ToolbarButton> buttons;
    buttons.reserve(12);
    const float categoryY = 4.0f;
    const float categoryH = 24.0f;
    float x = 7.0f;
    buttons.push_back({x, categoryY, 48.0f, categoryH, kLayout2DControl});
    x += 53.0f;
    buttons.push_back({x, categoryY, 48.0f, categoryH, kLayout3DControl});
    x += 53.0f;
    buttons.push_back({x, categoryY, 68.0f, categoryH, kLayoutBothControl});

    const float controlsY = 33.0f;
    const float controlsH = 25.0f;
    if (layout != PreviewLayout::TwoD) {
        const float regionWidth = layout == PreviewLayout::Both ? width * 0.5f : width;
        const float scale = std::clamp(regionWidth / 850.0f, 0.3f, 1.0f);
        float controlX = 7.0f * scale;
        buttons.push_back({controlX, controlsY, kToolbarExportWidth * scale, controlsH,
            kExportControl});
        controlX += (kToolbarExportWidth + kToolbarGap) * scale;
        buttons.push_back({controlX, controlsY, kToolbarShapeWidth * scale, controlsH,
            kShapeControl});
        controlX += (kToolbarShapeWidth + kToolbarGap) * scale;
        for (int32_t i = 0; i < 5; ++i) {
            const float buttonWidth = kToolbarModeWidths[i] * scale;
            buttons.push_back({controlX, controlsY, buttonWidth, controlsH, i});
            controlX += buttonWidth + kToolbarGap * scale;
        }
        controlX += kToolbarGap * scale;
        buttons.push_back({controlX, controlsY, kToolbarFlyWidth * scale, controlsH, kFlyControl});
        controlX += kToolbarFlyWidth * scale + kToolbarGap * scale;
        buttons.push_back({controlX, controlsY, kToolbarAxisWidth * scale, controlsH, kAxisControl});
        controlX += kToolbarAxisWidth * scale + kToolbarGap * scale;
        buttons.push_back({controlX, controlsY, kToolbarLiveWidth * scale, controlsH, kLiveControl});
    }
    if (layout != PreviewLayout::ThreeD) {
        const float regionLeft = layout == PreviewLayout::Both ? width * 0.5f : 0.0f;
        const float regionWidth = layout == PreviewLayout::Both ? width * 0.5f : width;
        const float scale = std::clamp(regionWidth / 660.0f, 0.4f, 1.0f);
        float uvX = regionLeft + 7.0f * scale;
        buttons.push_back({uvX, controlsY, kToolbarTextureWidth * scale,
            controlsH, kTextureControl});
        uvX += (kToolbarTextureWidth + kToolbarGap) * scale;
        constexpr float tileButtonWidth = 32.0f;
        const int32_t tileControls[] = {
            kUvTileUMinusControl, kUvTileUPlusControl,
            kUvTileVMinusControl, kUvTileVPlusControl};
        for (int32_t control : tileControls) {
            buttons.push_back({uvX, controlsY, tileButtonWidth * scale, controlsH, control});
            uvX += (tileButtonWidth + kToolbarGap) * scale;
        }
    }
    return buttons;
}

float toolbar_height(float, PreviewLayout, uint32_t) {
    return 62.0f;
}

struct PickerMenuLayout {
    float x = 0.0f;
    float y = 0.0f;
    float width = 164.0f;
    float rowHeight = 20.0f;
    float pagerY = 0.0f;
    size_t first = 0;
    size_t rowCount = 0;
    int pageCount = 1;
    bool hasPager = false;
};

PickerMenuLayout picker_menu_layout(float width, float height, size_t itemCount, int page,
    PreviewLayout previewLayout, int32_t controlId, float menuWidth) {
    PickerMenuLayout layout{};
    layout.width = menuWidth;
    layout.pageCount = std::max(1, static_cast<int>(
        (itemCount + static_cast<size_t>(kTextureRowsPerPage) - 1) / kTextureRowsPerPage));
    const int clampedPage = std::clamp(page, 0, layout.pageCount - 1);
    layout.first = static_cast<size_t>(clampedPage) * kTextureRowsPerPage;
    layout.rowCount = itemCount > layout.first
        ? std::min(static_cast<size_t>(kTextureRowsPerPage), itemCount - layout.first)
        : 0;
    layout.hasPager = layout.pageCount > 1;
    const auto buttons = toolbar_buttons(width, previewLayout, 0);
    for (const ToolbarButton& button : buttons) {
        if (button.id == controlId) {
            layout.x = button.x;
            break;
        }
    }
    float regionLeft = 0.0f;
    float regionWidth = width;
    for (const ToolbarButton& button : buttons) {
        if (button.id == controlId) {
            if (previewLayout == PreviewLayout::Both && button.x >= width * 0.5f) {
                regionLeft = width * 0.5f;
                regionWidth = width * 0.5f;
            }
            break;
        }
    }
    if (layout.width > regionWidth - 8.0f) layout.width = std::max(80.0f, regionWidth - 8.0f);
    if (layout.x + layout.width > regionLeft + regionWidth - 4.0f) {
        layout.x = std::max(regionLeft + 4.0f, regionLeft + regionWidth - layout.width - 4.0f);
    }
    layout.y = toolbar_height(width, previewLayout, 0) + 2.0f;
    layout.pagerY = layout.y + 4.0f + static_cast<float>(layout.rowCount) * layout.rowHeight;
    const float menuHeight = 4.0f + static_cast<float>(layout.rowCount) * layout.rowHeight +
        (layout.hasPager ? 24.0f : 0.0f);
    if (layout.y + menuHeight > height - 2.0f) {
        layout.y = std::max(0.0f, height - menuHeight - 2.0f);
        layout.pagerY = layout.y + 4.0f + static_cast<float>(layout.rowCount) * layout.rowHeight;
    }
    return layout;
}

const char* axis_label(uint32_t axis) {
    switch (axis) {
    case 1: return "UP:Z";
    case 2: return "UP:X";
    default: return "UP:Y";
    }
}

uint8_t glyph_row(char ch, int row) {
    static constexpr uint8_t s[7] = {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
    static constexpr uint8_t o[7] = {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    static constexpr uint8_t l[7] = {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
    static constexpr uint8_t i[7] = {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E};
    static constexpr uint8_t d[7] = {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E};
    static constexpr uint8_t t[7] = {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    static constexpr uint8_t e[7] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    static constexpr uint8_t x[7] = {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11};
    static constexpr uint8_t plus[7] = {0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00};
    static constexpr uint8_t w[7] = {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A};
    static constexpr uint8_t r[7] = {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    static constexpr uint8_t u[7] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    static constexpr uint8_t v[7] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04};
    static constexpr uint8_t p[7] = {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
    static constexpr uint8_t colon[7] = {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00};
    static constexpr uint8_t y[7] = {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04};
    static constexpr uint8_t n[7] = {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    static constexpr uint8_t zero[7] = {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
    static constexpr uint8_t one[7] = {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
    static constexpr uint8_t two[7] = {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
    static constexpr uint8_t three[7] = {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E};
    static constexpr uint8_t four[7] = {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
    static constexpr uint8_t five[7] = {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E};
    static constexpr uint8_t six[7] = {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E};
    static constexpr uint8_t seven[7] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    static constexpr uint8_t eight[7] = {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
    static constexpr uint8_t nine[7] = {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E};
    static constexpr uint8_t dash[7] = {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00};
    static constexpr uint8_t a[7] = {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    static constexpr uint8_t b[7] = {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    static constexpr uint8_t c[7] = {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
    static constexpr uint8_t f[7] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    static constexpr uint8_t g[7] = {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E};
    static constexpr uint8_t h[7] = {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    static constexpr uint8_t m[7] = {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11};
    static constexpr uint8_t j[7] = {0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C};
    static constexpr uint8_t k[7] = {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    static constexpr uint8_t q[7] = {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
    static constexpr uint8_t z[7] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};
    static constexpr uint8_t underscore[7] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F};
    if (row < 0 || row >= 7) return 0;
    const uint8_t* rows = nullptr;
    switch (ch) {
    case 'S': rows = s; break;
    case 'O': rows = o; break;
    case 'L': rows = l; break;
    case 'I': rows = i; break;
    case 'D': rows = d; break;
    case 'T': rows = t; break;
    case 'E': rows = e; break;
    case 'X': rows = x; break;
    case '+': rows = plus; break;
    case 'W': rows = w; break;
    case 'R': rows = r; break;
    case 'U': rows = u; break;
    case 'V': rows = v; break;
    case 'P': rows = p; break;
    case ':': rows = colon; break;
    case 'Y': rows = y; break;
    case 'N': rows = n; break;
    case 'A': rows = a; break;
    case 'B': rows = b; break;
    case 'C': rows = c; break;
    case 'F': rows = f; break;
    case 'G': rows = g; break;
    case 'H': rows = h; break;
    case 'M': rows = m; break;
    case 'J': rows = j; break;
    case 'K': rows = k; break;
    case 'Q': rows = q; break;
    case 'Z': rows = z; break;
    case '_': rows = underscore; break;
    case '0': rows = zero; break;
    case '1': rows = one; break;
    case '2': rows = two; break;
    case '3': rows = three; break;
    case '4': rows = four; break;
    case '5': rows = five; break;
    case '6': rows = six; break;
    case '7': rows = seven; break;
    case '8': rows = eight; break;
    case '9': rows = nine; break;
    case '-': rows = dash; break;
    default: return 0;
    }
    return rows[row];
}

void append_rect(std::vector<OverlayVertex>& vertices, float x0, float y0, float x1, float y1,
    float width, float height, const float color[4]) {
    if (x1 <= x0 || y1 <= y0 || width <= 0.0f || height <= 0.0f) return;
    const float left = 2.0f * x0 / width - 1.0f;
    const float right = 2.0f * x1 / width - 1.0f;
    const float top = 1.0f - 2.0f * y0 / height;
    const float bottom = 1.0f - 2.0f * y1 / height;
    const auto vertex = [&](float x, float y) {
        OverlayVertex out{};
        out.pos[0] = x;
        out.pos[1] = y;
        std::memcpy(out.color, color, sizeof(out.color));
        vertices.push_back(out);
    };
    vertex(left, top); vertex(right, top); vertex(left, bottom);
    vertex(left, bottom); vertex(right, top); vertex(right, bottom);
}

void append_text(std::vector<OverlayVertex>& vertices, const char* text, float x, float y,
    float scale, float width, float height, const float color[4]) {
    for (size_t charIndex = 0; text[charIndex] != '\0'; ++charIndex) {
        for (int row = 0; row < 7; ++row) {
            const uint8_t bits = glyph_row(text[charIndex], row);
            for (int col = 0; col < 5; ++col) {
                if ((bits & (1u << (4 - col))) == 0) continue;
                append_rect(vertices, x + (static_cast<float>(charIndex) * 6 + col) * scale,
                    y + row * scale, x + (static_cast<float>(charIndex) * 6 + col + 1) * scale,
                    y + (row + 1) * scale, width, height, color);
            }
        }
    }
}

void append_triangle(std::vector<OverlayVertex>& vertices, float x0, float y0, float x1, float y1,
    float width, float height, const float color[4]) {
    const auto vertex = [&](float x, float y) {
        OverlayVertex out{};
        out.pos[0] = 2.0f * x / width - 1.0f;
        out.pos[1] = 1.0f - 2.0f * y / height;
        std::memcpy(out.color, color, sizeof(out.color));
        vertices.push_back(out);
    };
    vertex(x0, y0);
    vertex(x1, (y0 + y1) * 0.5f);
    vertex(x0, y1);
}

void draw_toolbar(WGPURenderPassEncoder pass, uint32_t targetWidth, uint32_t targetHeight,
    PreviewLayout layout, uint32_t mode, uint32_t upAxis, int32_t hoverControl, uint32_t flyingCamera,
    int32_t solo, int32_t uvTex, int32_t uvTileOffsetX, int32_t uvTileOffsetY,
    uint32_t textureMenuOpen, int32_t texturePage, uint32_t shapeMenuOpen, int32_t shapePage,
    uint32_t liveUpdate) {
    if (g.pOverlay == nullptr || targetWidth == 0 || targetHeight == 0) return;
    static thread_local std::vector<OverlayVertex> vertices;
    vertices.clear();
    vertices.reserve(6000);
    const float width = static_cast<float>(targetWidth);
    const float height = static_cast<float>(targetHeight);
    const auto buttons = toolbar_buttons(width, layout, mode);
    const float toolbarBg[4] = {0.045f, 0.055f, 0.075f, 1.0f};
    const float buttonBg[4] = {0.11f, 0.13f, 0.17f, 1.0f};
    const float activeBg[4] = {0.10f, 0.32f, 0.53f, 1.0f};
    const float liveBg[4] = {0.12f, 0.40f, 0.22f, 1.0f};
    const float hoverBg[4] = {0.23f, 0.27f, 0.33f, 1.0f};
    const float textColor[4] = {0.95f, 0.96f, 0.98f, 1.0f};
    char textureLabel[24] = {};
    if (uvTex >= 0) std::snprintf(textureLabel, sizeof(textureLabel), "TEX %02d", uvTex);
    else std::snprintf(textureLabel, sizeof(textureLabel), "TEX --");
    char shapeLabel[24] = {};
    if (solo >= 0) std::snprintf(shapeLabel, sizeof(shapeLabel), "SH %02d", solo);
    else std::snprintf(shapeLabel, sizeof(shapeLabel), "SHAPE");
    append_rect(vertices, 0, 0, width, toolbar_height(width, layout, mode), width, height, toolbarBg);
    if (layout == PreviewLayout::Both) {
        const float divider[4] = {0.22f, 0.25f, 0.30f, 1.0f};
        const float center = width * 0.5f;
        append_rect(vertices, center - 0.5f, toolbar_height(width, layout, mode),
            center + 0.5f, height, width, height, divider);
    }

    for (const ToolbarButton& button : buttons) {
        const bool active = (button.id < 5 && button.id == static_cast<int32_t>(mode)) ||
            (button.id == kLayout2DControl && layout == PreviewLayout::TwoD) ||
            (button.id == kLayout3DControl && layout == PreviewLayout::ThreeD) ||
            (button.id == kLayoutBothControl && layout == PreviewLayout::Both) ||
            (button.id == kFlyControl && flyingCamera != 0) ||
            (button.id == kTextureControl && textureMenuOpen != 0) ||
            (button.id == kShapeControl && shapeMenuOpen != 0);
        const float* fill = button.id == hoverControl ? hoverBg :
            button.id == kLiveControl && liveUpdate != 0 ? liveBg : active ? activeBg : buttonBg;
        append_rect(vertices, button.x, button.y, button.x + button.w, button.y + button.h,
            width, height, fill);
        if (button.id == kLiveControl) {
            const float iconColor[4] = {0.96f, 0.97f, 0.99f, 1.0f};
            if (liveUpdate != 0) {
                const float barW = button.w * 0.16f;
                const float gap = button.w * 0.12f;
                const float barH = button.h * 0.42f;
                const float startX = button.x + (button.w - (barW * 2.0f + gap)) * 0.5f;
                const float startY = button.y + (button.h - barH) * 0.5f;
                append_rect(vertices, startX, startY, startX + barW, startY + barH,
                    width, height, iconColor);
                append_rect(vertices, startX + barW + gap, startY,
                    startX + barW * 2.0f + gap, startY + barH, width, height, iconColor);
            } else {
                const float iconW = button.w * 0.30f;
                const float iconH = button.h * 0.42f;
                append_triangle(vertices, button.x + (button.w - iconW) * 0.5f,
                    button.y + (button.h - iconH) * 0.5f,
                    button.x + (button.w + iconW) * 0.5f,
                    button.y + (button.h + iconH) * 0.5f, width, height, iconColor);
            }
            continue;
        }
        const char* label = button.id < 5 ? kToolbarModes[button.id] :
            button.id == kLayout2DControl ? "2D" :
            button.id == kLayout3DControl ? "3D" :
            button.id == kLayoutBothControl ? "BOTH" :
            button.id == kExportControl ? "EXPORT" :
            button.id == kFlyControl ? "FLY" :
            button.id == kAxisControl ? axis_label(upAxis) :
            button.id == kShapeControl ? shapeLabel :
            button.id == kUvTileUMinusControl ? "U-" :
            button.id == kUvTileUPlusControl ? "U+" :
            button.id == kUvTileVMinusControl ? "V-" :
            button.id == kUvTileVPlusControl ? "V+" : textureLabel;
        const float glyphWidth = static_cast<float>(std::strlen(label)) * 6.0f - 1.0f;
        const float fontScale = std::min({2.0f, button.h / 14.0f,
            std::max(0.1f, (button.w - 6.0f) / glyphWidth)});
        const float textWidth = glyphWidth * fontScale;
        const float textHeight = 7.0f * fontScale;
        const float textX = button.x + (button.w - textWidth) * 0.5f;
        const float textY = button.y + (button.h - textHeight) * 0.5f;
        append_text(vertices, label, textX, textY, fontScale, width, height, textColor);
    }

    if (layout != PreviewLayout::ThreeD) {
        float statusX = layout == PreviewLayout::Both ? width * 0.5f + 8.0f : 8.0f;
        for (const ToolbarButton& button : buttons) {
            if (button.id >= kUvTileUMinusControl && button.id <= kUvTileVPlusControl) {
                statusX = std::max(statusX, button.x + button.w + 6.0f);
            }
        }
        char tileLabel[48];
        std::snprintf(tileLabel, sizeof(tileLabel), "UV U%d V%d", uvTileOffsetX, uvTileOffsetY);
        const float scale = 1.0f;
        append_text(vertices, tileLabel, statusX, 41.0f, scale, width, height, textColor);
    }

    if (layout != PreviewLayout::ThreeD && textureMenuOpen != 0 &&
        g.mesh != nullptr && !g.mesh->textures->empty()) {
        const PickerMenuLayout menuLayout = picker_menu_layout(
            width, height, g.mesh->textures->size(), texturePage, layout, kTextureControl, 164.0f);
        const float menuBg[4] = {0.055f, 0.065f, 0.085f, 1.0f};
        const float rowBg[4] = {0.12f, 0.14f, 0.18f, 1.0f};
        const float selectedBg[4] = {0.10f, 0.32f, 0.53f, 1.0f};
        append_rect(vertices, menuLayout.x, menuLayout.y, menuLayout.x + menuLayout.width,
            menuLayout.pagerY + (menuLayout.hasPager ? 24.0f : 4.0f), width, height, menuBg);
        for (size_t row = 0; row < menuLayout.rowCount; ++row) {
            const size_t index = menuLayout.first + row;
            const float y = menuLayout.y + 2.0f + static_cast<float>(row) * menuLayout.rowHeight;
            const int32_t control = kTextureRowFirstControl + static_cast<int32_t>(row);
            char label[64];
            const TexImage& texture = (*g.mesh->textures)[index];
            if (!texture.desc.empty()) {
                std::snprintf(label, sizeof(label), "%02zu %s", index, texture.desc.c_str());
            } else {
                std::snprintf(label, sizeof(label), "%02zu %ux%u", index, texture.w, texture.h);
            }
            const bool selected = uvTex == static_cast<int32_t>(index);
            const float* fill = control == hoverControl ? hoverBg : selected ? selectedBg : rowBg;
            append_rect(vertices, menuLayout.x + 2.0f, y, menuLayout.x + menuLayout.width - 2.0f,
                y + menuLayout.rowHeight - 1.0f, width, height, fill);
            append_text(vertices, label, menuLayout.x + 8.0f, y + 6.0f, 1.0f,
                width, height, textColor);
        }
        if (menuLayout.hasPager) {
            const float pagerWidth = (menuLayout.width - 6.0f) * 0.5f;
            const float* prevFill = hoverControl == kTexturePrevControl ? hoverBg : rowBg;
            const float* nextFill = hoverControl == kTextureNextControl ? hoverBg : rowBg;
            append_rect(vertices, menuLayout.x + 2.0f, menuLayout.pagerY,
                menuLayout.x + 2.0f + pagerWidth, menuLayout.pagerY + 20.0f, width, height, prevFill);
            append_rect(vertices, menuLayout.x + 4.0f + pagerWidth, menuLayout.pagerY,
                menuLayout.x + menuLayout.width - 2.0f, menuLayout.pagerY + 20.0f, width, height, nextFill);
            append_text(vertices, "PREV", menuLayout.x + 8.0f, menuLayout.pagerY + 6.0f,
                1.0f, width, height, textColor);
            append_text(vertices, "NEXT", menuLayout.x + 12.0f + pagerWidth, menuLayout.pagerY + 6.0f,
                1.0f, width, height, textColor);
        }
    }
    if (layout != PreviewLayout::TwoD && shapeMenuOpen != 0 && g.mesh != nullptr) {
        const size_t itemCount = g.mesh->subs.size() + 1;
        const PickerMenuLayout menuLayout = picker_menu_layout(
            width, height, itemCount, shapePage, layout, kShapeControl, 220.0f);
        const float menuBg[4] = {0.055f, 0.065f, 0.085f, 1.0f};
        const float rowBg[4] = {0.12f, 0.14f, 0.18f, 1.0f};
        const float selectedBg[4] = {0.10f, 0.32f, 0.53f, 1.0f};
        append_rect(vertices, menuLayout.x, menuLayout.y, menuLayout.x + menuLayout.width,
            menuLayout.pagerY + (menuLayout.hasPager ? 24.0f : 4.0f), width, height, menuBg);
        for (size_t row = 0; row < menuLayout.rowCount; ++row) {
            const size_t index = menuLayout.first + row;
            const float y = menuLayout.y + 2.0f + static_cast<float>(row) * menuLayout.rowHeight;
            const int32_t control = kShapeRowFirstControl + static_cast<int32_t>(row);
            char label[64];
            bool selected = solo < 0 && index == 0;
            if (index == 0) {
                std::snprintf(label, sizeof(label), "ALL SHAPES");
            } else {
                const SubMesh& sub = g.mesh->subs[index - 1];
                char shapeName[25] = {};
                const std::string& sourceName = sub.label;
                const size_t nameLength = std::min(sourceName.size(), sizeof(shapeName) - 1);
                for (size_t i = 0; i < nameLength; ++i) {
                    char ch = sourceName[i];
                    if (ch >= 'a' && ch <= 'z') ch = static_cast<char>(ch - 'a' + 'A');
                    shapeName[i] = ch;
                }
                std::snprintf(label, sizeof(label), "SH %02u %s", sub.shapeIndex,
                    shapeName[0] == '\0' ? "-" : shapeName);
                selected = solo == static_cast<int32_t>(sub.shapeIndex);
            }
            const float* fill = control == hoverControl ? hoverBg : selected ? selectedBg : rowBg;
            append_rect(vertices, menuLayout.x + 2.0f, y, menuLayout.x + menuLayout.width - 2.0f,
                y + menuLayout.rowHeight - 1.0f, width, height, fill);
            append_text(vertices, label, menuLayout.x + 8.0f, y + 6.0f, 1.0f,
                width, height, textColor);
        }
        if (menuLayout.hasPager) {
            const float pagerWidth = (menuLayout.width - 6.0f) * 0.5f;
            const float* prevFill = hoverControl == kShapePrevControl ? hoverBg : rowBg;
            const float* nextFill = hoverControl == kShapeNextControl ? hoverBg : rowBg;
            append_rect(vertices, menuLayout.x + 2.0f, menuLayout.pagerY,
                menuLayout.x + 2.0f + pagerWidth, menuLayout.pagerY + 20.0f, width, height, prevFill);
            append_rect(vertices, menuLayout.x + 4.0f + pagerWidth, menuLayout.pagerY,
                menuLayout.x + menuLayout.width - 2.0f, menuLayout.pagerY + 20.0f, width, height, nextFill);
            append_text(vertices, "PREV", menuLayout.x + 8.0f, menuLayout.pagerY + 6.0f,
                1.0f, width, height, textColor);
            append_text(vertices, "NEXT", menuLayout.x + 12.0f + pagerWidth, menuLayout.pagerY + 6.0f,
                1.0f, width, height, textColor);
        }
    }
    if (vertices.empty()) return;

    const uint64_t bytes = vertices.size() * sizeof(OverlayVertex);
    if (g.overlayBuffer == nullptr || g.overlayBufferSize < bytes) {
        rel(g.overlayBuffer);
        uint64_t capacity = 1;
        while (capacity < bytes) capacity <<= 1;
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.label = {"inspector control vertices", WGPU_STRLEN};
        bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
        bd.size = capacity;
        g.overlayBuffer = wgpuDeviceCreateBuffer(g.device, &bd);
        g.overlayBufferSize = g.overlayBuffer != nullptr ? capacity : 0;
    }
    if (g.overlayBuffer == nullptr) return;

    wgpuQueueWriteBuffer(g.queue, g.overlayBuffer, 0, vertices.data(), bytes);
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, width, height, 0.0f, 1.0f);
    wgpuRenderPassEncoderSetPipeline(pass, g.pOverlay);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g.overlayBuffer, 0, g.overlayBufferSize);
    wgpuRenderPassEncoderDraw(pass, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
}

void axis_basis(uint32_t upAxis, float right[3], float up[3], float forward[3]) {
    right[0] = 1.0f; right[1] = 0.0f; right[2] = 0.0f;
    up[0] = 0.0f; up[1] = 1.0f; up[2] = 0.0f;
    forward[0] = 0.0f; forward[1] = 0.0f; forward[2] = 1.0f;
    if (upAxis == 1) {
        up[1] = 0.0f; up[2] = 1.0f;
        forward[1] = -1.0f; forward[2] = 0.0f;
    } else if (upAxis == 2) {
        right[0] = 0.0f; right[1] = -1.0f;
        up[0] = 1.0f; up[1] = 0.0f;
    }
}

void mesh_sphere(const MeshData* mesh, float center[3], float& radius) {
    if (mesh == nullptr) {
        center[0] = center[1] = center[2] = 0.0f;
        radius = 1.0f;
        return;
    }
    float extent[3];
    for (int i = 0; i < 3; ++i) {
        center[i] = (mesh->bmin[i] + mesh->bmax[i]) * 0.5f;
        extent[i] = mesh->bmax[i] - mesh->bmin[i];
    }
    radius = 0.5f * std::sqrt(extent[0] * extent[0] + extent[1] * extent[1] +
        extent[2] * extent[2]);
    if (!std::isfinite(radius) || radius <= 0.0f) radius = 1.0f;
}

void orbit_eye(const float center[3], float radius, float yawDegrees, float pitchDegrees,
    float zoom, uint32_t upAxis, float eye[3]) {
    float right[3], up[3], forward[3];
    axis_basis(upAxis, right, up, forward);
    const float yaw = yawDegrees * 3.14159265f / 180.0f;
    const float pitch = pitchDegrees * 3.14159265f / 180.0f;
    const float distance = radius / std::sin(20.0f * 3.14159265f / 180.0f) * 1.05f /
        std::max(zoom, 0.05f);
    const float horizontal = std::cos(pitch);
    for (int i = 0; i < 3; ++i) {
        eye[i] = center[i] + distance * (horizontal * std::sin(yaw) * right[i] +
            std::sin(pitch) * up[i] + horizontal * std::cos(yaw) * forward[i]);
    }
}

void camera_forward(float yawDegrees, float pitchDegrees, uint32_t upAxis, float direction[3]) {
    float right[3], up[3], forward[3];
    axis_basis(upAxis, right, up, forward);
    const float yaw = yawDegrees * 3.14159265f / 180.0f;
    const float pitch = pitchDegrees * 3.14159265f / 180.0f;
    const float horizontal = std::cos(pitch);
    for (int i = 0; i < 3; ++i) {
        direction[i] = -(horizontal * std::sin(yaw) * right[i] +
            std::sin(pitch) * up[i] + horizontal * std::cos(yaw) * forward[i]);
    }
}

void initialize_fly_position(ViewState& v) {
    float center[3];
    float radius = 1.0f;
    const std::shared_ptr<const MeshData> mesh = current_mesh();
    mesh_sphere(mesh.get(), center, radius);
    orbit_eye(center, radius, v.yaw, static_cast<float>(v.pitch),
        static_cast<float>(v.zoomPct) / 100.0f, static_cast<uint32_t>(std::clamp(v.upAxis, 0, 2)),
        v.flyPosition);
    v.flyPositionInitialized = true;
}

void on_present(ModContext*, const GfxPresentContext* c, const void* payload, size_t size, void*) {
    PresentPayload pl{};
    pl.zoom = 1.0f;
    pl.solo = -1;
    pl.uvTex = -1;
    pl.layout = static_cast<uint32_t>(PreviewLayout::ThreeD);
    if (size == sizeof(pl)) std::memcpy(&pl, payload, sizeof(pl));
    const PreviewLayout layout = pl.layout <= static_cast<uint32_t>(PreviewLayout::Both)
        ? static_cast<PreviewLayout>(pl.layout) : PreviewLayout::ThreeD;

    const bool ready = ensure_gpu(*c);
    if (ready) {
        ensure_depth(c->target_width, c->target_height);
        sync_mesh();
    }

    WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    ca.view = c->target_view;
    ca.loadOp = WGPULoadOp_Clear;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = WGPUColor{0.09, 0.09, 0.11, 1.0};
    WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    da.view = g.depthView;
    da.depthLoadOp = WGPULoadOp_Clear;
    da.depthStoreOp = WGPUStoreOp_Store;
    da.depthClearValue = 1.0f;

    const bool show3D = layout != PreviewLayout::TwoD;
    const bool showUV = layout != PreviewLayout::ThreeD;
    const bool haveMesh = ready && g.mesh != nullptr && g.vbuf != nullptr &&
        (!show3D || g.depthView != nullptr);
    WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    pd.label = {"inspector preview", WGPU_STRLEN};
    pd.colorAttachmentCount = 1;
    pd.colorAttachments = &ca;
    if (ready && show3D && g.depthView != nullptr) pd.depthStencilAttachment = &da;
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(c->encoder, &pd);

    const float targetWidth = static_cast<float>(std::max<uint32_t>(c->target_width, 1));
    const float targetHeight = static_cast<float>(std::max<uint32_t>(c->target_height, 1));
    const float toolbar = toolbar_height(targetWidth, layout, pl.mode);
    const float contentHeight = std::max(targetHeight - toolbar, 1.0f);
    const float leftWidth = layout == PreviewLayout::Both ? std::floor(targetWidth * 0.5f) : targetWidth;
    const float uvX = layout == PreviewLayout::Both ? leftWidth : 0.0f;
    const float uvWidth = layout == PreviewLayout::Both ? targetWidth - leftWidth : targetWidth;

    if (haveMesh && show3D) {
        wgpuRenderPassEncoderSetViewport(pass, 0, toolbar, leftWidth, contentHeight, 0.0f, 1.0f);
        const MeshData& mesh = *g.mesh;
        float center[3];
        float radius = 1.0f;
        mesh_sphere(&mesh, center, radius);
        const float fov = 40.0f * 3.14159265f / 180.0f;
        float eye[3], target[3], up[3], right[3], axisForward[3];
        axis_basis(pl.upAxis, right, up, axisForward);
        if (pl.flyingCamera != 0) {
            std::memcpy(eye, pl.flyPosition, sizeof(eye));
            float direction[3];
            camera_forward(pl.yaw, pl.pitch, pl.upAxis, direction);
            for (int i = 0; i < 3; ++i) target[i] = eye[i] + direction[i];
        } else {
            orbit_eye(center, radius, pl.yaw, pl.pitch, pl.zoom, pl.upAxis, eye);
            std::memcpy(target, center, sizeof(target));
        }
        const float dx = eye[0] - center[0];
        const float dy = eye[1] - center[1];
        const float dz = eye[2] - center[2];
        const float cameraDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float nearPlane = std::max(radius * 0.0001f, 0.01f);
        const float farPlane = std::max({radius * 4.0f, cameraDistance + radius * 2.0f,
            nearPlane + 1.0f});
        const Mat4 viewMatrix = mat_look_at(eye, target, up);
        const Mat4 proj = mat_perspective(fov, leftWidth / contentHeight, nearPlane, farPlane);
        const Mat4 vp = mat_mul(proj, viewMatrix);
        Uniforms un{};
        std::memcpy(un.viewProj, vp.m, sizeof(un.viewProj));
        const bool textured = pl.mode == 1 || pl.mode == 3;
        un.p0[0] = textured ? 1.0f : 0.0f;
        un.p0[1] = textured ? 1.0f : 0.0f;
        un.color[0] = 0.0f;
        un.color[1] = 0.0f;
        un.color[2] = 0.0f;
        un.color[3] = 0.5f;
        wgpuQueueWriteBuffer(g.queue, g.ubuf, 0, &un, sizeof(un));
        if (textured) {
            Uniforms alphaUn = un;
            alphaUn.p0[1] = 2.0f;
            wgpuQueueWriteBuffer(g.queue, g.alphaUbuf, 0, &alphaUn, sizeof(alphaUn));
        }
        const bool solid = pl.mode <= 3;
        const bool wire = pl.mode >= 2;
        WGPUSampler sampler = (pl.flags & 1u) ? g.sampNearest : g.sampLinear;
        if (solid) {
            wgpuRenderPassEncoderSetPipeline(pass, g.pTri);
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g.vbuf, 0, g.vbufSize);
            for (const SubMesh& sub : mesh.subs) {
                if (!sub_visible(sub, pl.solo)) continue;
                WGPUBindGroup bg = make_bg(g.lTri, view_for(sub.texture, g.viewGray), sampler);
                if (bg == nullptr) continue;
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                wgpuRenderPassEncoderDraw(pass, sub.vertexCount, 1, sub.firstVertex, 0);
                wgpuBindGroupRelease(bg);
            }
            if (textured) {
                wgpuRenderPassEncoderSetPipeline(pass, g.pTriAlpha);
                for (const SubMesh& sub : mesh.subs) {
                    if (!sub_visible(sub, pl.solo)) continue;
                    WGPUBindGroup bg = make_bg(g.lTriAlpha,
                        view_for(sub.texture, g.viewGray), sampler, g.alphaUbuf);
                    if (bg == nullptr) continue;
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(pass, sub.vertexCount, 1, sub.firstVertex, 0);
                    wgpuBindGroupRelease(bg);
                }
            }
        }
        if (wire) {
            WGPUBindGroup bg = make_bg(g.lWire, nullptr, nullptr);
            if (bg != nullptr) {
                wgpuRenderPassEncoderSetPipeline(pass, g.pWire);
                draw_wire(pass, bg, pl.solo, g.vbuf, g.vbufSize);
                wgpuBindGroupRelease(bg);
            }
        }
    }

    if (haveMesh && showUV) {
        wgpuRenderPassEncoderSetViewport(pass, uvX, toolbar, uvWidth, contentHeight, 0.0f, 1.0f);
        const float tileScale = std::min(uvWidth, contentHeight) *
            0.9f * std::max(pl.uvZoom, 0.05f);
        const float sx = 2.0f * tileScale / uvWidth;
        const float sy = 2.0f * tileScale / contentHeight;
        const float centerU = 0.5f;
        const float centerV = 0.5f;

        static thread_local std::vector<float> quadVertices;
        quadVertices.clear();
        quadVertices.reserve(36);
        const auto appendQuad = [&](int32_t tileX, int32_t tileY) {
            const float x0 = (static_cast<float>(tileX) - centerU) * sx + pl.uvPanX;
            const float x1 = x0 + sx;
            const float y0 = (centerV - static_cast<float>(tileY)) * sy + pl.uvPanY;
            const float y1 = y0 - sy;
            const float data[6][6] = {
                {x0, y0, 0.0f, 0.0f, 0.0f, 0.0f}, {x1, y0, 0.0f, 1.0f, 0.0f, 0.0f},
                {x0, y1, 0.0f, 0.0f, 1.0f, 0.0f}, {x0, y1, 0.0f, 0.0f, 1.0f, 0.0f},
                {x1, y0, 0.0f, 1.0f, 0.0f, 0.0f}, {x1, y1, 0.0f, 1.0f, 1.0f, 0.0f}};
            for (const auto& vertex : data) quadVertices.insert(quadVertices.end(), vertex, vertex + 6);
        };
        appendQuad(0, 0);
        const uint64_t quadBytes = static_cast<uint64_t>(quadVertices.size()) * sizeof(float);
        if (g.uvBgVbuf == nullptr || g.uvBgVbufSize < quadBytes) {
            rel(g.uvBgVbuf);
            uint64_t capacity = 1;
            while (capacity < quadBytes) capacity <<= 1;
            WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
            bd.label = {"inspector UV backgrounds", WGPU_STRLEN};
            bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
            bd.size = capacity;
            g.uvBgVbuf = wgpuDeviceCreateBuffer(g.device, &bd);
            g.uvBgVbufSize = g.uvBgVbuf != nullptr ? capacity : 0;
        }
        if (g.uvBgVbuf != nullptr && quadBytes > 0) {
            wgpuQueueWriteBuffer(g.queue, g.uvBgVbuf, 0, quadVertices.data(), quadBytes);
            Uniforms backgroundUniforms{};
            backgroundUniforms.uvxf[0] = 1.0f;
            backgroundUniforms.uvxf[1] = 1.0f;
            wgpuQueueWriteBuffer(g.queue, g.uvUbuf, 0, &backgroundUniforms,
                sizeof(backgroundUniforms));
            Uniforms wireUniforms{};
            wireUniforms.uvxf[0] = sx;
            wireUniforms.uvxf[1] = -sy;
            wireUniforms.uvxf[2] = -centerU * sx +
                static_cast<float>(pl.uvTileOffsetX) * sx + pl.uvPanX;
            wireUniforms.uvxf[3] = centerV * sy -
                static_cast<float>(pl.uvTileOffsetY) * sy + pl.uvPanY;
            wgpuQueueWriteBuffer(g.queue, g.uvWireUbuf, 0, &wireUniforms,
                sizeof(wireUniforms));
            const WGPUSampler sampler = (pl.flags & 1u) ? g.sampNearest : g.sampLinear;
            wgpuRenderPassEncoderSetPipeline(pass, g.pUvBg);
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g.uvBgVbuf, 0, quadBytes);
            const int32_t uvTexture = pl.uvTex >= 0
                ? pl.uvTex : default_uv_texture(*g.mesh, pl.solo);
            WGPUBindGroup bg = make_bg(g.lUvBg, view_for(uvTexture, g.viewDark), sampler, g.uvUbuf);
            if (bg != nullptr) {
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
                wgpuBindGroupRelease(bg);
            }
            WGPUBindGroup bgw = make_bg(g.lUvWire, nullptr, nullptr, g.uvWireUbuf);
            if (bgw != nullptr) {
                wgpuRenderPassEncoderSetPipeline(pass, g.pUvWire);
                draw_wire(pass, bgw, pl.solo, g.vbuf, g.vbufSize);
                wgpuBindGroupRelease(bgw);
            }
        }
    }

    if (ready) {
        draw_toolbar(pass, c->target_width, c->target_height, layout, pl.mode, pl.upAxis,
            pl.hoverControl, pl.flyingCamera, pl.solo, pl.uvTex,
            pl.uvTileOffsetX, pl.uvTileOffsetY, pl.textureMenuOpen, pl.texturePage,
            pl.shapeMenuOpen, pl.shapePage, pl.liveUpdate);
    }
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
}

ModResult register_target() {
    if (g_window == 0 || g_target != 0) return MOD_CONFLICT;
    GfxPresentTargetDesc d = GFX_PRESENT_TARGET_DESC_INIT;
    d.label = "Model preview surface";
    d.render = on_present;
    return svc_gfx->register_window_present_target(mod_ctx, g_window, &d, &g_target);
}

void on_window_event(ModContext*, WindowHandle, const WindowEvent* event, void*) {
    if (event->type == WINDOW_EVENT_CLOSE_REQUESTED) {
        preview_close();
    } else if (event->type == WINDOW_EVENT_FOCUS_LOST) {
        g_dragMode = 0;
        g_previousF = false;
        g_previousLeftMouse = false;
        g_hoverControl = -1;
        g_lastInputTime = {};
    }
}

void update_preview_input(ViewState& v) {
    WindowInfo info = WINDOW_INFO_INIT;
    if (g_window == 0 || svc_window->get_info(mod_ctx, g_window, &info) != MOD_OK) {
        g_dragMode = 0;
        g_previousF = false;
        g_previousLeftMouse = false;
        g_hoverControl = -1;
        g_lastInputTime = {};
        return;
    }
    bool focused = info.focused;
    bool fDown = false;
    bool alt = false;
    bool escapeDown = false;
    bool pointerInPreview = false;
    int mouseX = 0;
    int mouseY = 0;
    bool leftDown = false;
    bool rightDown = false;
    int clientWidth = 0;
    int clientHeight = 0;
    bool moveForward = false;
    bool moveBackward = false;
    bool moveLeft = false;
    bool moveRight = false;
    bool moveUp = false;
    bool moveDown = false;
    bool speedBoost = false;
#if defined(_WIN32)
    HWND previewWindow = FindWindowW(nullptr, L"Model Preview");
    focused = focused && previewWindow != nullptr && GetForegroundWindow() == previewWindow;
    fDown = focused && (GetAsyncKeyState('F') & 0x8000) != 0;
    alt = focused && (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;
    escapeDown = focused && (GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0;
    leftDown = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    rightDown = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    moveForward = focused && (GetAsyncKeyState('W') & 0x8000) != 0;
    moveBackward = focused && (GetAsyncKeyState('S') & 0x8000) != 0;
    moveLeft = focused && (GetAsyncKeyState('A') & 0x8000) != 0;
    moveRight = focused && (GetAsyncKeyState('D') & 0x8000) != 0;
    moveDown = focused && (GetAsyncKeyState('Q') & 0x8000) != 0;
    moveUp = focused && (GetAsyncKeyState('E') & 0x8000) != 0;
    speedBoost = focused && (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
    POINT cursor{};
    RECT client{};
    if (focused && GetCursorPos(&cursor) && ScreenToClient(previewWindow, &cursor) &&
        GetClientRect(previewWindow, &client)) {
        const int rectWidth = client.right - client.left;
        const int rectHeight = client.bottom - client.top;
        const float scaleX = rectWidth > 0 && info.pixel_width != 0
            ? static_cast<float>(info.pixel_width) / rectWidth
            : 1.0f;
        const float scaleY = rectHeight > 0 && info.pixel_height != 0
            ? static_cast<float>(info.pixel_height) / rectHeight
            : 1.0f;
        pointerInPreview = cursor.x >= client.left && cursor.x < client.right &&
            cursor.y >= client.top && cursor.y < client.bottom;
        mouseX = static_cast<int>(std::lround((cursor.x - client.left) * scaleX));
        mouseY = static_cast<int>(std::lround((cursor.y - client.top) * scaleY));
        clientWidth = info.pixel_width != 0 ? static_cast<int>(info.pixel_width) : rectWidth;
        clientHeight = info.pixel_height != 0 ? static_cast<int>(info.pixel_height) : rectHeight;
    }
#endif
    const auto now = std::chrono::steady_clock::now();
    float deltaSeconds = 1.0f / 60.0f;
    if (g_lastInputTime.time_since_epoch().count() != 0) {
        deltaSeconds = std::chrono::duration<float>(now - g_lastInputTime).count();
        deltaSeconds = std::clamp(deltaSeconds, 0.0f, 0.1f);
    }
    g_lastInputTime = focused ? now : std::chrono::steady_clock::time_point{};

    if (fDown && !g_previousF) {
        v.yaw = 30.0f;
        v.pitch = 15;
        v.zoomPct = 100;
        v.uvZoomPct = 100;
        v.uvPanX = 0.0f;
        v.uvPanY = 0.0f;
        v.uvTileOffsetX = 0;
        v.uvTileOffsetY = 0;
        v.autoRotate = false;
        if (v.flyingCamera) v.flyPositionInitialized = false;
        g_dragMode = 0;
    }
    g_previousF = fDown;

    if (!focused) {
        g_dragMode = 0;
        g_previousLeftMouse = false;
        g_hoverControl = -1;
        return;
    }

    const std::shared_ptr<const MeshData> mesh = current_mesh();
    const size_t textureCount = mesh ? mesh->textures->size() : 0;
    const size_t shapeItemCount = mesh ? mesh->subs.size() + 1 : 0;
    const int pageCount = std::max(1, static_cast<int>(
        (textureCount + static_cast<size_t>(kTextureRowsPerPage) - 1) / kTextureRowsPerPage));
    const int shapePageCount = std::max(1, static_cast<int>(
        (shapeItemCount + static_cast<size_t>(kTextureRowsPerPage) - 1) / kTextureRowsPerPage));
    v.texturePage = std::clamp(v.texturePage, 0, pageCount - 1);
    v.shapePage = std::clamp(v.shapePage, 0, shapePageCount - 1);
    if (v.flyingCamera && !v.flyPositionInitialized) initialize_fly_position(v);

    g_hoverControl = -1;
    if (pointerInPreview && v.layout != PreviewLayout::ThreeD &&
        v.textureMenuOpen && textureCount > 0) {
        const PickerMenuLayout layout = picker_menu_layout(
            static_cast<float>(clientWidth), static_cast<float>(clientHeight), textureCount,
            v.texturePage, v.layout, kTextureControl, 164.0f);
        const bool insideMenu = mouseX >= layout.x && mouseX < layout.x + layout.width &&
            mouseY >= layout.y && mouseY < layout.pagerY + (layout.hasPager ? 20.0f : 0.0f);
        if (insideMenu) {
            for (size_t row = 0; row < layout.rowCount; ++row) {
                const float y = layout.y + 2.0f + static_cast<float>(row) * layout.rowHeight;
                if (mouseY >= y && mouseY < y + layout.rowHeight - 1.0f) {
                    g_hoverControl = kTextureRowFirstControl + static_cast<int32_t>(row);
                    break;
                }
            }
            if (layout.hasPager && mouseY >= layout.pagerY && mouseY < layout.pagerY + 20.0f) {
                const float pagerWidth = (layout.width - 6.0f) * 0.5f;
                if (mouseX >= layout.x + 2.0f &&
                    mouseX < layout.x + 2.0f + pagerWidth) {
                    g_hoverControl = kTexturePrevControl;
                } else if (mouseX >= layout.x + 4.0f + pagerWidth &&
                    mouseX < layout.x + layout.width - 2.0f) {
                    g_hoverControl = kTextureNextControl;
                }
            }
        }
    }
    if (g_hoverControl < 0 && pointerInPreview && v.layout != PreviewLayout::TwoD &&
        v.shapeMenuOpen && shapeItemCount > 0) {
        const PickerMenuLayout layout = picker_menu_layout(
            static_cast<float>(clientWidth), static_cast<float>(clientHeight), shapeItemCount,
            v.shapePage, v.layout, kShapeControl, 220.0f);
        const bool insideMenu = mouseX >= layout.x && mouseX < layout.x + layout.width &&
            mouseY >= layout.y && mouseY < layout.pagerY + (layout.hasPager ? 20.0f : 0.0f);
        if (insideMenu) {
            for (size_t row = 0; row < layout.rowCount; ++row) {
                const float y = layout.y + 2.0f + static_cast<float>(row) * layout.rowHeight;
                if (mouseY >= y && mouseY < y + layout.rowHeight - 1.0f) {
                    g_hoverControl = kShapeRowFirstControl + static_cast<int32_t>(row);
                    break;
                }
            }
            if (layout.hasPager && mouseY >= layout.pagerY && mouseY < layout.pagerY + 20.0f) {
                const float pagerWidth = (layout.width - 6.0f) * 0.5f;
                if (mouseX >= layout.x + 2.0f &&
                    mouseX < layout.x + 2.0f + pagerWidth) {
                    g_hoverControl = kShapePrevControl;
                } else if (mouseX >= layout.x + 4.0f + pagerWidth &&
                    mouseX < layout.x + layout.width - 2.0f) {
                    g_hoverControl = kShapeNextControl;
                }
            }
        }
    }
    if (g_hoverControl < 0 && pointerInPreview &&
        mouseY < toolbar_height(static_cast<float>(clientWidth), v.layout,
            static_cast<uint32_t>(v.mode))) {
        const auto buttons = toolbar_buttons(static_cast<float>(clientWidth), v.layout,
            static_cast<uint32_t>(v.mode));
        for (const ToolbarButton& button : buttons) {
            if (mouseX >= button.x && mouseX < button.x + button.w &&
                mouseY >= button.y && mouseY < button.y + button.h) {
                g_hoverControl = button.id;
                break;
            }
        }
    }

    const bool leftPressed = leftDown && !g_previousLeftMouse;
    const bool toolbarClick = leftPressed && pointerInPreview &&
        g_hoverControl >= 0 && !alt;
    g_previousLeftMouse = leftDown;
    if (toolbarClick) {
        if (g_hoverControl < 5) {
            v.mode = g_hoverControl;
            v.textureMenuOpen = false;
            v.shapeMenuOpen = false;
        } else if (g_hoverControl == kExportControl) {
            v.textureMenuOpen = false;
            v.shapeMenuOpen = false;
            if (g_exportHandler != nullptr) {
                g_exportHandler(v.solo);
            } else {
                logf(2, "GLB export handler is not available");
            }
        } else if (g_hoverControl == kLayout2DControl) {
            set_preview_layout(PreviewLayout::TwoD);
        } else if (g_hoverControl == kLayout3DControl) {
            set_preview_layout(PreviewLayout::ThreeD);
        } else if (g_hoverControl == kLayoutBothControl) {
            set_preview_layout(PreviewLayout::Both);
        } else if (g_hoverControl == kFlyControl) {
            set_flying_camera(!v.flyingCamera);
            if (v.flyingCamera) initialize_fly_position(v);
        } else if (g_hoverControl == kAxisControl) {
            v.upAxis = (v.upAxis + 1) % 3;
        } else if (g_hoverControl == kUvTileUMinusControl) {
            v.uvTileOffsetX = std::max(v.uvTileOffsetX - 1, -9999);
        } else if (g_hoverControl == kUvTileUPlusControl) {
            v.uvTileOffsetX = std::min(v.uvTileOffsetX + 1, 9999);
        } else if (g_hoverControl == kUvTileVMinusControl) {
            v.uvTileOffsetY = std::max(v.uvTileOffsetY - 1, -9999);
        } else if (g_hoverControl == kUvTileVPlusControl) {
            v.uvTileOffsetY = std::min(v.uvTileOffsetY + 1, 9999);
        } else if (g_hoverControl == kTextureControl) {
            v.textureMenuOpen = !v.textureMenuOpen;
            v.shapeMenuOpen = false;
        } else if (g_hoverControl == kShapeControl) {
            v.shapeMenuOpen = !v.shapeMenuOpen;
            v.textureMenuOpen = false;
        } else if (g_hoverControl == kLiveControl) {
            set_live_update(!v.liveUpdate);
        } else if (g_hoverControl == kTexturePrevControl) {
            v.texturePage = std::max(v.texturePage - 1, 0);
        } else if (g_hoverControl == kTextureNextControl) {
            v.texturePage = std::min(v.texturePage + 1, pageCount - 1);
        } else if (g_hoverControl == kShapePrevControl) {
            v.shapePage = std::max(v.shapePage - 1, 0);
        } else if (g_hoverControl == kShapeNextControl) {
            v.shapePage = std::min(v.shapePage + 1, shapePageCount - 1);
        } else if (g_hoverControl >= kTextureRowFirstControl &&
            g_hoverControl < kShapeRowFirstControl) {
            const size_t textureIndex = static_cast<size_t>(v.texturePage) * kTextureRowsPerPage +
                static_cast<size_t>(g_hoverControl - kTextureRowFirstControl);
            if (textureIndex < textureCount) {
                v.uvTex = static_cast<int>(textureIndex);
            }
            v.textureMenuOpen = false;
        } else if (g_hoverControl >= kShapeRowFirstControl &&
            g_hoverControl < kShapeRowFirstControl + kTextureRowsPerPage) {
            const size_t shapeIndex = static_cast<size_t>(v.shapePage) * kTextureRowsPerPage +
                static_cast<size_t>(g_hoverControl - kShapeRowFirstControl);
            if (shapeIndex == 0) {
                v.solo = -1;
                v.uvTex = -1;
            } else if (shapeIndex < shapeItemCount && mesh != nullptr) {
                const SubMesh& sub = mesh->subs[shapeIndex - 1];
                v.solo = static_cast<int32_t>(sub.shapeIndex);
                v.uvTex = sub.texture;
            }
            v.shapeMenuOpen = false;
        }
        g_dragMode = 0;
        return;
    }
    if (leftPressed && v.textureMenuOpen &&
        g_hoverControl != kTextureControl) {
        v.textureMenuOpen = false;
    }
    if (leftPressed && v.shapeMenuOpen && g_hoverControl != kShapeControl) {
        v.shapeMenuOpen = false;
    }
    if (escapeDown) {
        v.textureMenuOpen = false;
        v.shapeMenuOpen = false;
    }

    if (v.flyingCamera && v.layout != PreviewLayout::TwoD) {
        float center[3];
        float radius = 1.0f;
        mesh_sphere(mesh.get(), center, radius);
        float right[3], up[3], axisForward[3];
        axis_basis(static_cast<uint32_t>(std::clamp(v.upAxis, 0, 2)), right, up, axisForward);
        const float yaw = v.yaw * 3.14159265f / 180.0f;
        float forward[3];
        for (int i = 0; i < 3; ++i) {
            forward[i] = -(std::sin(yaw) * right[i] + std::cos(yaw) * axisForward[i]);
        }
        float side[3];
        for (int i = 0; i < 3; ++i) {
            side[i] = std::cos(yaw) * right[i] - std::sin(yaw) * axisForward[i];
        }
        float movement[3] = {0.0f, 0.0f, 0.0f};
        const float forwardAmount = (moveForward ? 1.0f : 0.0f) - (moveBackward ? 1.0f : 0.0f);
        const float sideAmount = (moveRight ? 1.0f : 0.0f) - (moveLeft ? 1.0f : 0.0f);
        const float verticalAmount = (moveUp ? 1.0f : 0.0f) - (moveDown ? 1.0f : 0.0f);
        for (int i = 0; i < 3; ++i) {
            movement[i] = forward[i] * forwardAmount + side[i] * sideAmount + up[i] * verticalAmount;
        }
        const float movementLength = std::sqrt(movement[0] * movement[0] +
            movement[1] * movement[1] + movement[2] * movement[2]);
        if (movementLength > 0.0f) {
            const float speedScale = static_cast<float>(std::clamp(v.flySpeedPct, 25, 3000)) / 100.0f;
            const float speed = std::clamp(radius * 0.05f, 2.0f, 200.0f) * speedScale *
                (speedBoost ? 4.0f : 1.0f);
            for (int i = 0; i < 3; ++i) {
                v.flyPosition[i] += movement[i] / movementLength * speed * deltaSeconds;
            }
            v.autoRotate = false;
        }
    }

    const float uvPanelStart = v.layout == PreviewLayout::Both ? clientWidth * 0.5f : 0.0f;
    const bool pointerInUvPanel = pointerInPreview && v.layout != PreviewLayout::ThreeD &&
        static_cast<float>(mouseX) >= uvPanelStart;
    const bool pointerIn3DPanel = pointerInPreview && v.layout != PreviewLayout::TwoD &&
        (v.layout != PreviewLayout::Both || static_cast<float>(mouseX) < uvPanelStart);
    const int dragMode = pointerInUvPanel
        ? (leftDown ? 3 : rightDown ? 2 : 0)
        : pointerIn3DPanel
            ? (alt && leftDown ? 1 : rightDown ? (v.flyingCamera ? 1 : 2) : 0)
            : 0;

    if (dragMode == 0) {
        g_dragMode = 0;
        return;
    }
    if (dragMode != g_dragMode) {
        g_dragMode = dragMode;
        g_previousMouseX = mouseX;
        g_previousMouseY = mouseY;
        return;
    }

    const float dx = static_cast<float>(mouseX - g_previousMouseX);
    const float dy = static_cast<float>(mouseY - g_previousMouseY);
    g_previousMouseX = mouseX;
    g_previousMouseY = mouseY;
    if (dragMode == 1) {
        v.yaw = std::fmod(v.yaw - dx * 0.5f + 360.0f, 360.0f);
        v.pitch = std::clamp(static_cast<int>(std::lround(v.pitch + dy * 0.5f)), -89, 89);
        v.autoRotate = false;
    } else if (dy != 0.0f) {
        if (dragMode == 3 && clientWidth > 0 && clientHeight > 0) {
            const float uvWidth = v.layout == PreviewLayout::Both
                ? static_cast<float>(clientWidth) * 0.5f : static_cast<float>(clientWidth);
            const float uvHeight = static_cast<float>(clientHeight) -
                toolbar_height(static_cast<float>(clientWidth), v.layout,
                    static_cast<uint32_t>(v.mode));
            v.uvPanX += 2.0f * dx / std::max(uvWidth, 1.0f);
            v.uvPanY -= 2.0f * dy / std::max(uvHeight, 1.0f);
        } else {
            int& zoomPct = pointerInUvPanel ? v.uvZoomPct : v.zoomPct;
            const float zoom = static_cast<float>(zoomPct) * std::exp(-dy * 0.01f);
            zoomPct = std::clamp(static_cast<int>(std::lround(zoom)), 10, 800);
        }
        v.autoRotate = false;
    } else if (dragMode == 3 && clientWidth > 0 && clientHeight > 0 && dx != 0.0f) {
        const float uvWidth = v.layout == PreviewLayout::Both
            ? static_cast<float>(clientWidth) * 0.5f : static_cast<float>(clientWidth);
        v.uvPanX += 2.0f * dx / std::max(uvWidth, 1.0f);
    }
}

}  // namespace

bool preview_is_open() { return g_window != 0; }

void preview_set_export_handler(PreviewExportHandler handler) {
    g_exportHandler = handler;
}

void set_flying_camera(bool enabled) {
    ViewState& v = view();
    if (enabled && v.layout == PreviewLayout::TwoD) return;
    if (v.flyingCamera == enabled) return;
    v.flyingCamera = enabled;
    v.flyPositionInitialized = false;
    v.textureMenuOpen = false;
    v.shapeMenuOpen = false;
    if (enabled) v.autoRotate = false;
}

void set_live_update(bool enabled) {
    ViewState& v = view();
    if (enabled && v.layout == PreviewLayout::TwoD) return;
    v.liveUpdate = enabled;
    g_nextLiveSnapshot = enabled ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
}

void set_preview_layout(PreviewLayout layout) {
    ViewState& v = view();
    v.layout = layout;
    v.textureMenuOpen = false;
    v.shapeMenuOpen = false;
    if (layout == PreviewLayout::TwoD) {
        set_flying_camera(false);
        set_live_update(false);
    }
}

ModResult preview_open() {
    if (g_window != 0) return MOD_OK;
    WindowDesc wd = WINDOW_DESC_INIT;
    wd.title = "Model Preview";
    wd.width = 900;
    wd.height = 700;
    wd.flags |= WINDOW_FLAG_ALWAYS_ON_TOP;
    wd.on_event = on_window_event;
    ModResult r = svc_window->create_window(mod_ctx, &wd, &g_window);
    if (r != MOD_OK) {
        g_window = 0;
        logf(2, "could not create preview window");
        return r;
    }
    r = register_target();
    if (r != MOD_OK) {
        logf(2, "could not attach WebGPU surface to preview window");
        preview_close();
        return r;
    }
    return svc_window->show_window(mod_ctx, g_window);
}

void preview_close() {
    g_recreateTarget = false;
    set_live_update(false);
    g_dragMode = 0;
    g_previousF = false;
    g_previousLeftMouse = false;
    g_hoverControl = -1;
    g_lastInputTime = {};
    if (g_target != 0) {
        if (svc_gfx->unregister_present_target(mod_ctx, g_target) == MOD_OK) g_target = 0;
    }
    if (g_window != 0 && g_target == 0) {
        if (svc_window->destroy_window(mod_ctx, g_window) == MOD_OK) g_window = 0;
    }
}

void preview_frame(ModContext*, const GfxStageContext*, void*) {
    if (g_target == 0) return;
    ViewState& v = view();
    update_preview_input(v);
    if (v.flyingCamera && !v.flyPositionInitialized) initialize_fly_position(v);
    if (v.liveUpdate) {
        const auto now = std::chrono::steady_clock::now();
        if (g_nextLiveSnapshot.time_since_epoch().count() == 0 || now >= g_nextLiveSnapshot) {
            request_refresh();
            g_nextLiveSnapshot = now +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(1.0 / 30.0));
        }
    }
    if (v.autoRotate && v.layout != PreviewLayout::TwoD && !v.flyingCamera) {
        v.yaw = std::fmod(v.yaw + 0.7f, 360.0f);
    }
    PresentPayload pl{};
    pl.version = mesh_version();
    pl.yaw = v.yaw;
    pl.pitch = static_cast<float>(v.pitch);
    pl.zoom = static_cast<float>(v.zoomPct) / 100.0f;
    pl.mode = static_cast<uint32_t>(v.mode);
    pl.flags = v.nearest ? 1u : 0u;
    pl.solo = v.solo;
    pl.uvTex = v.uvTex;
    pl.upAxis = static_cast<uint32_t>(std::clamp(v.upAxis, 0, 2));
    pl.layout = static_cast<uint32_t>(v.layout);
    pl.hoverControl = g_hoverControl;
    std::memcpy(pl.flyPosition, v.flyPosition, sizeof(pl.flyPosition));
    pl.texturePage = v.texturePage;
    pl.shapePage = v.shapePage;
    pl.flyingCamera = v.flyingCamera ? 1u : 0u;
    pl.textureMenuOpen = v.textureMenuOpen ? 1u : 0u;
    pl.shapeMenuOpen = v.shapeMenuOpen ? 1u : 0u;
    pl.liveUpdate = v.liveUpdate ? 1u : 0u;
    pl.uvPanX = v.uvPanX;
    pl.uvPanY = v.uvPanY;
    pl.uvTileOffsetX = v.uvTileOffsetX;
    pl.uvTileOffsetY = v.uvTileOffsetY;
    pl.uvZoom = static_cast<float>(v.uvZoomPct) / 100.0f;
    if (svc_gfx->push_present(mod_ctx, g_target, &pl, sizeof(pl)) == MOD_ERROR) {
        g_recreateTarget = true;
    }
}

ModResult preview_update(ModError* error) {
    if (!g_recreateTarget || g_window == 0) return MOD_OK;
    g_recreateTarget = false;
    if (g_target != 0) {
        const ModResult r = svc_gfx->unregister_present_target(mod_ctx, g_target);
        if (r != MOD_OK) return mods::set_error(error, r, "failed to unregister lost present target");
        g_target = 0;
    }
    const ModResult r = register_target();
    if (r != MOD_OK) return mods::set_error(error, r, "failed to recreate present target");
    return MOD_OK;
}

void preview_shutdown() {
    preview_close();
    release_all_gpu();
}

}  // namespace mi
