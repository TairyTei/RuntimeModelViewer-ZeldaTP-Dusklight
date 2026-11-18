// Preview window: an auxiliary native window (WindowService) with our own WebGPU renderer
// (GfxService present target). Rendering code runs on the render worker thread; everything it
// needs arrives in a small payload or via the mutex-protected snapshot in core.cpp.
#include "preview.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "common.hpp"
#include "core.hpp"
#include "math.hpp"
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
  p0 : vec4<f32>,     // x: 1 = use texture colour
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
  let dx = dpdx(i.world);
  let dy = dpdy(i.world);
  let n = normalize(cross(dx, dy));
  let l = normalize(vec3<f32>(0.35, 0.8, 0.5));
  let shade = 0.35 + 0.65 * abs(dot(n, l));
  let t = textureSample(tex, smp, i.uv);
  var base = hash_color(i.id);
  if (u.p0.x > 0.5) {
    base = t.rgb;
  }
  if (u.p0.x > 0.5 && t.a < 0.1) {
    discard;
  }
  return vec4<f32>(base * shade, 1.0);
}

// ---- 3D wire overlay ----
@vertex fn vs_wire(v : VIn) -> @builtin(position) vec4<f32> {
  var p = u.view_proj * vec4<f32>(v.pos, 1.0);
  p.z = p.z - 0.0004 * p.w;
  return p;
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

@vertex fn vs_quad(@builtin(vertex_index) vi : u32) -> QuadOut {
  var c = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0));
  let uv = c[vi];
  var o : QuadOut;
  o.clip = vec4<f32>(uv * u.uvxf.xy + u.uvxf.zw, 0.0, 1.0);
  o.uv = uv;
  return o;
}

@fragment fn fs_quad(i : QuadOut) -> @location(0) vec4<f32> {
  let t = textureSample(tex, smp, i.uv);
  return vec4<f32>(t.rgb, 1.0);
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
};
static_assert(sizeof(PresentPayload) <= GFX_INLINE_DRAW_PAYLOAD_SIZE);

// ---------------------------------------------------------------------------------------------
// Window state (game thread)
// ---------------------------------------------------------------------------------------------
WindowHandle g_window = 0;
GfxPresentTargetHandle g_target = 0;
bool g_recreateTarget = false;
bool g_previousF = false;
int g_dragMode = 0; // 0 none, 1 orbit, 2 zoom
int g_previousMouseX = 0;
int g_previousMouseY = 0;

// ---------------------------------------------------------------------------------------------
// GPU state (render worker thread; released from the game thread at shutdown)
// ---------------------------------------------------------------------------------------------
struct Gpu {
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUTextureFormat colorFormat = WGPUTextureFormat_Undefined;

    WGPUShaderModule module = nullptr;
    WGPURenderPipeline pTri = nullptr, pWire = nullptr, pUvBg = nullptr, pUvWire = nullptr;
    WGPUBindGroupLayout lTri = nullptr, lWire = nullptr, lUvBg = nullptr, lUvWire = nullptr;

    WGPUBuffer ubuf = nullptr;
    WGPUSampler sampLinear = nullptr, sampNearest = nullptr;
    WGPUTexture texGray = nullptr, texDark = nullptr;
    WGPUTextureView viewGray = nullptr, viewDark = nullptr;

    WGPUTexture depth = nullptr;
    WGPUTextureView depthView = nullptr;
    uint32_t depthW = 0, depthH = 0;

    uint64_t meshVersion = 0;
    std::shared_ptr<const MeshData> mesh;
    WGPUBuffer vbuf = nullptr, ibuf = nullptr;
    uint64_t vbufSize = 0, ibufSize = 0;
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
    rel(g.ibuf);
    for (auto& v : g.views) rel(v);
    for (auto& t : g.texs) rel(t);
    g.views.clear();
    g.texs.clear();
    g.mesh.reset();
    g.meshVersion = 0;
}

void release_pipelines() {
    rel(g.pTri); rel(g.pWire); rel(g.pUvBg); rel(g.pUvWire);
    rel(g.lTri); rel(g.lWire); rel(g.lUvBg); rel(g.lUvWire);
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
    rel(g.ubuf);
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
    WGPUPrimitiveTopology topo, bool vertexBuffer, bool depth, bool depthWrite) {
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

bool ensure_gpu(const GfxPresentContext& c) {
    if (g.device != c.device) {
        // first use (or device change): rebuild everything
        release_all_gpu();
        g.device = c.device;
        g.queue = c.queue;
    }
    if (g.ubuf == nullptr) {
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.label = {"inspector uniforms", WGPU_STRLEN};
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        bd.size = kUniformSize;
        g.ubuf = wgpuDeviceCreateBuffer(g.device, &bd);
        g.sampLinear = make_sampler(false);
        g.sampNearest = make_sampler(true);
        const uint8_t gray[4] = {200, 200, 200, 255};
        const uint8_t dark[4] = {46, 46, 54, 255};
        create_rgba(1, 1, gray, g.texGray, g.viewGray);
        create_rgba(1, 1, dark, g.texDark, g.viewDark);
        if (g.ubuf == nullptr || g.sampLinear == nullptr || g.viewGray == nullptr) return false;
    }
    if (g.pTri != nullptr && g.colorFormat == c.target_format) return true;

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
    g.pWire = make_pipeline("inspector wire", "vs_wire", "fs_wire", WGPUPrimitiveTopology_LineList,
        true, true, false);
    g.pUvBg = make_pipeline("inspector uv bg", "vs_quad", "fs_quad",
        WGPUPrimitiveTopology_TriangleList, false, false, false);
    g.pUvWire = make_pipeline("inspector uv wire", "vs_uvwire", "fs_uvwire",
        WGPUPrimitiveTopology_LineList, true, false, false);
    if (!g.pTri || !g.pWire || !g.pUvBg || !g.pUvWire) {
        logf(2, "FAILED to create WebGPU pipelines!");
        release_pipelines();
        return false;
    }
    g.lTri = wgpuRenderPipelineGetBindGroupLayout(g.pTri, 0);
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

    g.texs.assign(mesh->textures.size(), nullptr);
    g.views.assign(mesh->textures.size(), nullptr);
    for (size_t i = 0; i < mesh->textures.size(); ++i) {
        const TexImage& t = mesh->textures[i];
        if (t.w == 0 || t.h == 0 || t.rgba.empty()) continue;
        create_rgba(t.w, t.h, t.rgba.data(), g.texs[i], g.views[i]);
    }
}

WGPUBindGroup make_bg(WGPUBindGroupLayout layout, WGPUTextureView view, WGPUSampler samp) {
    WGPUBindGroupEntry e[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
        WGPU_BIND_GROUP_ENTRY_INIT};
    e[0].binding = 0;
    e[0].buffer = g.ubuf;
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

void draw_wire(WGPURenderPassEncoder pass, WGPUBindGroup bg, int32_t solo) {
    if (g.mesh == nullptr || g.vbuf == nullptr || g.ibuf == nullptr) return;
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g.vbuf, 0, g.vbufSize);
    wgpuRenderPassEncoderSetIndexBuffer(pass, g.ibuf, WGPUIndexFormat_Uint32, 0, g.ibufSize);
    for (const SubMesh& s : g.mesh->subs) {
        if (!sub_visible(s, solo)) continue;
        wgpuRenderPassEncoderDrawIndexed(pass, s.vertexCount * 2, 1, s.firstVertex * 2, 0, 0);
    }
}

void on_present(ModContext*, const GfxPresentContext* c, const void* payload, size_t size, void*) {
    PresentPayload pl{};
    pl.zoom = 1.0f;
    pl.solo = -1;
    pl.uvTex = -1;
    if (size == sizeof(pl)) std::memcpy(&pl, payload, sizeof(pl));

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

    const bool uvMode = pl.mode == 5;
    const bool haveMesh = ready && g.mesh != nullptr && g.vbuf != nullptr &&
                          (uvMode || g.depthView != nullptr);
    const bool useDepth = ready && !uvMode && g.depthView != nullptr;

    WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    pd.label = {"inspector preview", WGPU_STRLEN};
    pd.colorAttachmentCount = 1;
    pd.colorAttachments = &ca;
    if (useDepth) pd.depthStencilAttachment = &da;
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(c->encoder, &pd);

    if (haveMesh) {
        const float w = static_cast<float>(std::max<uint32_t>(c->target_width, 1));
        const float h = static_cast<float>(std::max<uint32_t>(c->target_height, 1));
        Uniforms un{};
        wgpuRenderPassEncoderSetViewport(pass, 0, 0, w, h, 0.0f, 1.0f);

        if (!uvMode) {
            const MeshData& m = *g.mesh;
            const float ctr[3] = {(m.bmin[0] + m.bmax[0]) * 0.5f, (m.bmin[1] + m.bmax[1]) * 0.5f,
                (m.bmin[2] + m.bmax[2]) * 0.5f};
            const float ex = m.bmax[0] - m.bmin[0], ey = m.bmax[1] - m.bmin[1],
                        ez = m.bmax[2] - m.bmin[2];
                        float radius = std::max(0.5f * std::sqrt(ex * ex + ey * ey + ez * ez), 0.001f);
                        if (!std::isfinite(radius) || radius <= 0.0f) radius = 1.0f; // Защита от NaN
            const float fov = 40.0f * 3.14159265f / 180.0f;
            const float dist = radius / std::sin(fov * 0.5f) * 1.05f / std::max(pl.zoom, 0.05f);
            const float yaw = pl.yaw * 3.14159265f / 180.0f;
            const float pitch = pl.pitch * 3.14159265f / 180.0f;
            const float eye[3] = {ctr[0] + dist * std::cos(pitch) * std::sin(yaw),
                ctr[1] + dist * std::sin(pitch), ctr[2] + dist * std::cos(pitch) * std::cos(yaw)};
            const float up[3] = {0, 1, 0};
            const Mat4 view = mat_look_at(eye, ctr, up);
            const Mat4 proj = mat_perspective(fov, w / h, std::max(radius * 0.01f, 0.01f), dist + radius * 4.0f);
            const Mat4 vp = mat_mul(proj, view);
            std::memcpy(un.viewProj, vp.m, sizeof(un.viewProj));
            const bool textured = pl.mode == 1 || pl.mode == 3;
            un.p0[0] = textured ? 1.0f : 0.0f;
            if (pl.mode == 4) {
                un.color[0] = 0.85f; un.color[1] = 0.9f; un.color[2] = 0.95f; un.color[3] = 1.0f;
            } else {
                un.color[0] = 0.02f; un.color[1] = 0.02f; un.color[2] = 0.03f; un.color[3] = 1.0f;
            }
            wgpuQueueWriteBuffer(g.queue, g.ubuf, 0, &un, sizeof(un));

            const bool solid = pl.mode <= 3;
            const bool wire = pl.mode >= 2;
            WGPUSampler samp = (pl.flags & 1u) ? g.sampNearest : g.sampLinear;
            if (solid) {
                wgpuRenderPassEncoderSetPipeline(pass, g.pTri);
                wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g.vbuf, 0, g.vbufSize);
                for (const SubMesh& s : m.subs) {
                    if (!sub_visible(s, pl.solo)) continue;
                    WGPUBindGroup bg = make_bg(g.lTri, view_for(s.texture, g.viewGray), samp);
                    if (bg == nullptr) continue;
                    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                    wgpuRenderPassEncoderDraw(pass, s.vertexCount, 1, s.firstVertex, 0);
                    wgpuBindGroupRelease(bg);
                }
            }
            if (wire) {
                WGPUBindGroup bg = make_bg(g.lWire, nullptr, nullptr);
                if (bg != nullptr) {
                    wgpuRenderPassEncoderSetPipeline(pass, g.pWire);
                    draw_wire(pass, bg, pl.solo);
                    wgpuBindGroupRelease(bg);
                }
            }
        } else {
            const float side = std::min(w, h) * 0.9f * std::max(pl.zoom, 0.05f);
            const float sx = 2.0f * side / w;
            const float sy = 2.0f * side / h;
            un.uvxf[0] = sx;
            un.uvxf[1] = -sy;
            un.uvxf[2] = -0.5f * sx;
            un.uvxf[3] = 0.5f * sy;
            wgpuQueueWriteBuffer(g.queue, g.ubuf, 0, &un, sizeof(un));

            WGPUSampler samp = (pl.flags & 1u) ? g.sampNearest : g.sampLinear;
            const int32_t uvTexture = pl.uvTex >= 0 ? pl.uvTex : default_uv_texture(*g.mesh, pl.solo);
            WGPUBindGroup bgq = make_bg(g.lUvBg, view_for(uvTexture, g.viewDark), samp);
            if (bgq != nullptr) {
                wgpuRenderPassEncoderSetPipeline(pass, g.pUvBg);
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bgq, 0, nullptr);
                wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
                wgpuBindGroupRelease(bgq);
            }
            WGPUBindGroup bgw = make_bg(g.lUvWire, nullptr, nullptr);
            if (bgw != nullptr) {
                wgpuRenderPassEncoderSetPipeline(pass, g.pUvWire);
                draw_wire(pass, bgw, pl.solo);
                wgpuBindGroupRelease(bgw);
            }
        }
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
    }
}

void update_preview_input(ViewState& v) {
    WindowInfo info = WINDOW_INFO_INIT;
    if (g_window == 0 || svc_window->get_info(mod_ctx, g_window, &info) != MOD_OK) {
        g_dragMode = 0;
        g_previousF = false;
        return;
    }
    bool focused = info.focused;
    bool fDown = false;
    bool alt = false;
    bool pointerInPreview = false;
    int mouseX = 0;
    int mouseY = 0;
    bool leftDown = false;
    bool rightDown = false;
#if defined(_WIN32)
    HWND previewWindow = FindWindowW(nullptr, L"Model Preview");
    focused = focused && previewWindow != nullptr && GetForegroundWindow() == previewWindow;
    fDown = focused && (GetAsyncKeyState('F') & 0x8000) != 0;
    alt = focused && (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;
    leftDown = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    rightDown = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    POINT cursor{};
    RECT client{};
    if (focused && GetCursorPos(&cursor) && ScreenToClient(previewWindow, &cursor) &&
        GetClientRect(previewWindow, &client)) {
        pointerInPreview = cursor.x >= client.left && cursor.x < client.right &&
            cursor.y >= client.top && cursor.y < client.bottom;
        mouseX = cursor.x;
        mouseY = cursor.y;
    }
#endif
    if (fDown && !g_previousF) {
        v.yaw = 30.0f;
        v.pitch = 15;
        v.zoomPct = 100;
        v.autoRotate = false;
        g_dragMode = 0;
    }
    g_previousF = fDown;

    if (!focused) {
        g_dragMode = 0;
        return;
    }

    const int dragMode = pointerInPreview && alt && leftDown ? 1 :
        pointerInPreview && rightDown ? 2 : 0;

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
        v.yaw = std::fmod(v.yaw + dx * 0.5f + 360.0f, 360.0f);
        v.pitch = std::clamp(static_cast<int>(std::lround(v.pitch + dy * 0.5f)), -89, 89);
        v.autoRotate = false;
    } else if (dy != 0.0f) {
        const float zoom = static_cast<float>(v.zoomPct) * std::exp(-dy * 0.01f);
        v.zoomPct = std::clamp(static_cast<int>(std::lround(zoom)), 10, 800);
        v.autoRotate = false;
    }
}

}  // namespace

bool preview_is_open() { return g_window != 0; }

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
    g_dragMode = 0;
    g_previousF = false;
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
    if (v.autoRotate && v.mode != 5) {
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

void inspect_model(uint64_t key) {
    select_model(key);
    preview_open();
}

}  // namespace mi
