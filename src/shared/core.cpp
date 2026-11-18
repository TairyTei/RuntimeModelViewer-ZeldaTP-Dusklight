#include "shared/core.hpp"

#include <algorithm>
#include <cmath>
#include <map>

#include "shared/common.hpp"

namespace mi {
namespace {

std::mutex g_mutex;
std::map<uint64_t, ModelInfo> g_models;
uint64_t g_frame = 0;
uint64_t g_selected = 0;
uint64_t g_pending = 0;
uint64_t g_pendingSince = 0;
std::shared_ptr<const MeshData> g_mesh;
uint64_t g_meshVersion = 0;
ViewState g_view;
CameraSnap g_cam;
std::vector<Selector> g_selectors;

constexpr uint64_t kStaleFrames = 3;
constexpr uint64_t kPendingTimeout = 180;

}  // namespace

void begin_frame() {
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_frame;
    for (auto it = g_models.begin(); it != g_models.end();) {
        if (it->second.lastFrame + kStaleFrames < g_frame) it = g_models.erase(it);
        else ++it;
    }
    if (g_pending != 0 && g_pendingSince + kPendingTimeout < g_frame) {
        logf(1, "model %llx was not drawn again; snapshot request dropped",
            static_cast<unsigned long long>(g_pending));
        g_pending = 0;
    }
}

uint64_t frame_no() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_frame;
}

void registry_note(const ModelInfo& info) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ModelInfo& e = g_models[info.key];
    e = info;
    e.lastFrame = g_frame;
}

std::vector<ModelInfo> registry_snapshot() {
    std::vector<ModelInfo> out;
    CameraSnap cam;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        cam = g_cam;
        for (const auto& kv : g_models) out.push_back(kv.second);
    }
    auto dist2 = [&](const ModelInfo& m) {
        if (!cam.valid) return 0.0f;
        const float dx = m.pos[0] - cam.eye[0], dy = m.pos[1] - cam.eye[1],
                    dz = m.pos[2] - cam.eye[2];
        return dx * dx + dy * dy + dz * dz;
    };
    std::sort(out.begin(), out.end(), [&](const ModelInfo& a, const ModelInfo& b) {
        return dist2(a) < dist2(b);
    });
    return out;
}

void select_model(uint64_t key) {
    std::lock_guard<std::mutex> lock(g_mutex);
    logf(0, "select_model called with key=%llx", (unsigned long long)key);
    g_selected = key;
    g_pending = key;
    g_pendingSince = g_frame;
    g_view.solo = -1;
    g_view.uvTex = -1;
    g_view.texturePage = 0;
    g_view.textureMenuOpen = false;
    g_view.uvPanX = 0.0f;
    g_view.uvPanY = 0.0f;
    g_view.uvTileOffsetX = 0;
    g_view.uvTileOffsetY = 0;
    g_view.uvZoomPct = 100;
}

uint64_t selected_key() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_selected;
}

void request_refresh() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_selected != 0) {
        g_pending = g_selected;
        g_pendingSince = g_frame;
    }
}

bool take_pending(uint64_t key) {
    std::lock_guard<std::mutex> lock(g_mutex);
    // logf(0, "take_pending: key=%llx pending=%llx",
    //     (unsigned long long)key, (unsigned long long)g_pending);
    if (g_pending != 0 && g_pending == key) {
        g_pending = 0;
        return true;
    }
    return false;
}

void publish_mesh(std::shared_ptr<MeshData> mesh) {
    std::lock_guard<std::mutex> lock(g_mutex);
    mesh->version = ++g_meshVersion;
    g_mesh = std::move(mesh);
}

std::shared_ptr<const MeshData> current_mesh() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_mesh;
}

uint64_t mesh_version() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_meshVersion;
}

ViewState& view() { return g_view; }

void set_camera(const CameraSnap& cam) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_cam = cam;
}

CameraSnap camera() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_cam;
}

void register_selector(const Selector& s) { g_selectors.push_back(s); }
const std::vector<Selector>& selectors() { return g_selectors; }

bool pick_screen_center(const PickContext& ctx, uint64_t* outKey) {
    if (!ctx.cam.valid || ctx.models == nullptr) return false;
    float best = 1e30f;
    bool found = false;
    for (const ModelInfo& m : *ctx.models) {
        const float r = std::max(m.radius, 20.0f);
        const float v[3] = {m.pos[0] - ctx.cam.eye[0], m.pos[1] + r * 0.5f - ctx.cam.eye[1],
            m.pos[2] - ctx.cam.eye[2]};
        const float t = v[0] * ctx.cam.fwd[0] + v[1] * ctx.cam.fwd[1] + v[2] * ctx.cam.fwd[2];
        if (t <= 0.0f) continue;
        const float d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - t * t;
        const float d = std::sqrt(std::max(d2, 0.0f));
        if (d > r) continue;
        const float score = d / r;
        if (score < best) {
            best = score;
            *outKey = m.key;
            found = true;
        }
    }
    return found;
}

}  // namespace mi
