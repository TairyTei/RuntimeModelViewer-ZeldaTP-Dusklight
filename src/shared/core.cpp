#include "shared/core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include "shared/common.hpp"

namespace mi {
namespace {

std::mutex g_mutex;
std::map<uint64_t, ModelInfo> g_models;
struct SceneModel {
    ModelInfo info;
    std::shared_ptr<const MeshData> mesh;
};
std::map<uint64_t, SceneModel> g_sceneModels;
std::map<uint64_t, uint64_t> g_sceneCaptureFrames;
uint64_t g_frame = 0;
uint64_t g_selected = 0;
uint64_t g_pending = 0;
uint64_t g_pendingSince = 0;
std::shared_ptr<const MeshData> g_mesh;
uint64_t g_meshVersion = 0;
bool g_sceneViewEnabled = false;
bool g_sceneMeshDirty = true;
bool g_sceneCompositionDirty = true;
std::shared_ptr<const MeshData> g_sceneMesh;
uint64_t g_sceneMeshVersion = 0;
uint64_t g_sceneCaptureGeneration = 0;
uint64_t g_sceneMeshBuiltGeneration = UINT64_MAX;
std::chrono::steady_clock::time_point g_nextSceneCapture{};
uint64_t g_sceneCaptureFrame = 0;
bool g_sceneCapturePending = false;
std::map<uint64_t, std::vector<int32_t>> g_sceneTextureRemaps;
ViewState g_view;
CameraSnap g_cam;
std::vector<Selector> g_selectors;

constexpr uint64_t kStaleFrames = 3;
constexpr uint64_t kPendingTimeout = 180;

std::shared_ptr<const MeshData> build_scene_mesh_locked() {
    if (!g_sceneMeshDirty || g_sceneMeshBuiltGeneration == g_sceneCaptureGeneration) {
        return g_sceneMesh;
    }
    g_sceneMeshDirty = false;
    g_sceneMeshBuiltGeneration = g_sceneCaptureGeneration;

    auto scene = std::make_shared<MeshData>();
    scene->title = "All visible scene models";
    scene->key = 0;
    if (g_sceneCompositionDirty || g_sceneMesh == nullptr) {
        g_sceneTextureRemaps.clear();
        auto sceneTextures = std::make_shared<std::vector<TexImage>>();
        std::map<std::pair<uint64_t, size_t>, int32_t> textureIndices;
        for (const auto& entry : g_sceneModels) {
            const std::shared_ptr<const MeshData>& mesh = entry.second.mesh;
            if (mesh == nullptr || mesh->textures == nullptr ||
                entry.second.info.dataKey != mesh->dataKey) continue;
            std::vector<int32_t>& remap = g_sceneTextureRemaps[entry.first];
            remap.resize(mesh->textures->size(), -1);
            for (size_t texture = 0; texture < mesh->textures->size(); ++texture) {
                const auto textureKey = std::make_pair(mesh->dataKey, texture);
                const auto found = textureIndices.find(textureKey);
                if (found != textureIndices.end()) {
                    remap[texture] = found->second;
                } else if (sceneTextures->size() < static_cast<size_t>(INT32_MAX)) {
                    remap[texture] = static_cast<int32_t>(sceneTextures->size());
                    textureIndices.emplace(textureKey, remap[texture]);
                    sceneTextures->push_back((*mesh->textures)[texture]);
                }
            }
        }
        scene->textures = std::move(sceneTextures);
        g_sceneCompositionDirty = false;
    } else {
        scene->textures = g_sceneMesh->textures;
    }

    float lo[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    float hi[3] = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()};
    uint32_t nextShapeIndex = 0;

    for (const auto& entry : g_sceneModels) {
        const std::shared_ptr<const MeshData>& mesh = entry.second.mesh;
        const ModelInfo& modelInfo = entry.second.info;
        if (mesh == nullptr || modelInfo.dataKey != mesh->dataKey || mesh->textures == nullptr) {
            continue;
        }

        const uint64_t firstVertex = scene->vertices.size() / 6u;
        if (firstVertex + mesh->vertices.size() / 6u > UINT32_MAX) {
            logf(1, "scene preview skipped model %llx because combined geometry exceeds limits",
                static_cast<unsigned long long>(entry.first));
            continue;
        }
        bool validTransform = true;
        for (float value : modelInfo.transform) {
            validTransform = validTransform && std::isfinite(value);
        }
        if (!validTransform) {
            logf(1, "scene preview skipped model %llx with a non-finite world transform",
                static_cast<unsigned long long>(entry.first));
            continue;
        }

        scene->vertices.reserve(scene->vertices.size() + mesh->vertices.size());
        for (size_t vertex = 0; vertex + 5 < mesh->vertices.size(); vertex += 6) {
            const float x = mesh->vertices[vertex];
            const float y = mesh->vertices[vertex + 1];
            const float z = mesh->vertices[vertex + 2];
            const float wx = modelInfo.transform[0] * x +
                modelInfo.transform[1] * y + modelInfo.transform[2] * z +
                modelInfo.transform[3];
            const float wy = modelInfo.transform[4] * x +
                modelInfo.transform[5] * y + modelInfo.transform[6] * z +
                modelInfo.transform[7];
            const float wz = modelInfo.transform[8] * x +
                modelInfo.transform[9] * y + modelInfo.transform[10] * z +
                modelInfo.transform[11];
            if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz)) {
                validTransform = false;
                break;
            }
            scene->vertices.insert(scene->vertices.end(),
                {wx, wy, wz, mesh->vertices[vertex + 3], mesh->vertices[vertex + 4],
                    mesh->vertices[vertex + 5] + static_cast<float>(nextShapeIndex)});
        }
        if (!validTransform) {
            scene->vertices.resize(static_cast<size_t>(firstVertex) * 6u);
            logf(1, "scene preview skipped model %llx with invalid transformed vertices",
                static_cast<unsigned long long>(entry.first));
            continue;
        }

        for (uint32_t corner = 0; corner < 8; ++corner) {
            const float x = mesh->bmin[0] + ((corner & 1u) ? mesh->bmax[0] - mesh->bmin[0] : 0.0f);
            const float y = mesh->bmin[1] + ((corner & 2u) ? mesh->bmax[1] - mesh->bmin[1] : 0.0f);
            const float z = mesh->bmin[2] + ((corner & 4u) ? mesh->bmax[2] - mesh->bmin[2] : 0.0f);
            const float world[3] = {
                modelInfo.transform[0] * x + modelInfo.transform[1] * y +
                    modelInfo.transform[2] * z + modelInfo.transform[3],
                modelInfo.transform[4] * x + modelInfo.transform[5] * y +
                    modelInfo.transform[6] * z + modelInfo.transform[7],
                modelInfo.transform[8] * x + modelInfo.transform[9] * y +
                    modelInfo.transform[10] * z + modelInfo.transform[11],
            };
            for (int axis = 0; axis < 3; ++axis) {
                lo[axis] = std::min(lo[axis], world[axis]);
                hi[axis] = std::max(hi[axis], world[axis]);
            }
        }

        const auto remapIt = g_sceneTextureRemaps.find(entry.first);

        for (const SubMesh& source : mesh->subs) {
            const uint64_t endVertex = static_cast<uint64_t>(source.firstVertex) + source.vertexCount;
            if (endVertex > mesh->vertices.size() / 6u ||
                firstVertex + source.firstVertex > UINT32_MAX) {
                logf(1, "scene preview skipped invalid submesh in model %llx",
                    static_cast<unsigned long long>(entry.first));
                continue;
            }
            SubMesh sub = source;
            sub.firstVertex = static_cast<uint32_t>(firstVertex + source.firstVertex);
            sub.shapeIndex = source.shapeIndex + nextShapeIndex;
            sub.instanceKey = entry.first;
            if (source.texture >= 0 &&
                remapIt != g_sceneTextureRemaps.end() &&
                static_cast<size_t>(source.texture) < remapIt->second.size()) {
                sub.texture = remapIt->second[static_cast<size_t>(source.texture)];
            } else {
                sub.texture = -1;
            }
            scene->subs.push_back(std::move(sub));
        }
        scene->joints += mesh->joints;
        scene->shapes += mesh->shapes;
        scene->materials += mesh->materials;
        nextShapeIndex += mesh->shapes;
    }

    scene->vertexCount = static_cast<uint32_t>(scene->vertices.size() / 6u);
    scene->triCount = scene->vertexCount / 3u;
    if (scene->vertexCount != 0) {
        for (int axis = 0; axis < 3; ++axis) {
            scene->bmin[axis] = lo[axis];
            scene->bmax[axis] = hi[axis];
        }
    }
    scene->version = ++g_sceneMeshVersion;
    g_sceneMesh = scene->vertexCount != 0 ? scene : nullptr;
    return g_sceneMesh;
}

}  // namespace

void begin_frame() {
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_frame;
    for (auto it = g_models.begin(); it != g_models.end();) {
        if (it->second.lastFrame + kStaleFrames < g_frame) {
            if (g_sceneViewEnabled && g_view.liveUpdate &&
                g_sceneModels.erase(it->first) != 0) {
                g_sceneTextureRemaps.erase(it->first);
                g_sceneMeshDirty = true;
                g_sceneCompositionDirty = true;
            }
            g_sceneCaptureFrames.erase(it->first);
            it = g_models.erase(it);
        }
        else ++it;
    }
    if (g_sceneViewEnabled && g_sceneMeshDirty) {
        build_scene_mesh_locked();
    }
    g_sceneCaptureFrame = 0;
    if (g_sceneViewEnabled) {
        const auto now = std::chrono::steady_clock::now();
        if (g_sceneCapturePending || (g_view.liveUpdate && now >= g_nextSceneCapture)) {
            ++g_sceneCaptureGeneration;
            g_sceneCaptureFrame = g_frame;
            g_sceneCapturePending = false;
            g_nextSceneCapture = now + std::chrono::milliseconds(33);
        }
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
    const auto sceneModel = g_sceneModels.find(info.key);
    if (sceneModel != g_sceneModels.end() && sceneModel->second.mesh != nullptr &&
        sceneModel->second.mesh->dataKey != info.dataKey) {
        g_sceneModels.erase(sceneModel);
        g_sceneTextureRemaps.erase(info.key);
        g_sceneMeshDirty = true;
        g_sceneCompositionDirty = true;
    }
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
    g_pending = g_sceneViewEnabled ? 0 : key;
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

void set_scene_view_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_sceneViewEnabled == enabled) return;
    g_sceneViewEnabled = enabled;
    g_sceneMeshDirty = true;
    g_sceneCompositionDirty = true;
    g_sceneMeshBuiltGeneration = UINT64_MAX;
    if (enabled) {
        g_sceneModels.clear();
        g_sceneCaptureFrames.clear();
        g_sceneTextureRemaps.clear();
        g_sceneCapturePending = true;
        g_nextSceneCapture = std::chrono::steady_clock::now();
    } else {
        g_sceneModels.clear();
        g_sceneCaptureFrames.clear();
        g_sceneTextureRemaps.clear();
        g_sceneCapturePending = false;
        g_sceneCaptureFrame = 0;
        g_sceneMesh.reset();
        if (g_selected != 0) {
            g_pending = g_selected;
            g_pendingSince = g_frame;
        }
    }
}

bool scene_view_enabled() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_sceneViewEnabled;
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

bool take_scene_capture_request(uint64_t key) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_sceneViewEnabled || g_sceneCaptureFrame != g_frame) return false;
    uint64_t& capturedGeneration = g_sceneCaptureFrames[key];
    if (capturedGeneration == g_frame) return false;
    capturedGeneration = g_frame;
    return true;
}

void publish_scene_mesh(const ModelInfo& info, std::shared_ptr<MeshData> mesh) {
    if (mesh == nullptr) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto existing = g_sceneModels.find(info.key);
    if (existing == g_sceneModels.end() || existing->second.mesh == nullptr ||
        existing->second.mesh->dataKey != mesh->dataKey ||
        existing->second.mesh->textures != mesh->textures) {
        g_sceneCompositionDirty = true;
    }
    g_sceneModels[info.key] = {info, std::move(mesh)};
    g_sceneMeshDirty = true;
}

bool update_scene_model_info(const ModelInfo& info) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_sceneModels.find(info.key);
    if (found == g_sceneModels.end() || found->second.mesh == nullptr ||
        found->second.mesh->dataKey != info.dataKey) {
        return false;
    }
    bool transformChanged = false;
    for (size_t i = 0; i < 12; ++i) {
        if (found->second.info.transform[i] != info.transform[i]) {
            transformChanged = true;
            break;
        }
    }
    found->second.info = info;
    if (transformChanged) g_sceneMeshDirty = true;
    return true;
}

std::shared_ptr<const MeshData> scene_model_mesh(uint64_t key) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto found = g_sceneModels.find(key);
    return found == g_sceneModels.end() ? nullptr : found->second.mesh;
}

void publish_mesh(std::shared_ptr<MeshData> mesh) {
    std::lock_guard<std::mutex> lock(g_mutex);
    mesh->version = ++g_meshVersion;
    g_mesh = std::move(mesh);
}

std::shared_ptr<const MeshData> selected_mesh() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_mesh;
}

std::shared_ptr<const MeshData> current_mesh() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_sceneViewEnabled ? g_sceneMesh : g_mesh;
}

uint64_t mesh_version() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_sceneViewEnabled) {
        return g_sceneMesh != nullptr ? g_sceneMesh->version : g_sceneMeshVersion;
    }
    return g_meshVersion;
}

bool scene_focus_sphere(uint64_t key, float center[3], float& radius) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_sceneViewEnabled || key == 0) return false;
    const auto found = g_sceneModels.find(key);
    if (found == g_sceneModels.end() || found->second.mesh == nullptr) return false;
    const MeshData& mesh = *found->second.mesh;
    float lo[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    float hi[3] = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max()};
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const float x = mesh.bmin[0] + ((corner & 1u) ? mesh.bmax[0] - mesh.bmin[0] : 0.0f);
        const float y = mesh.bmin[1] + ((corner & 2u) ? mesh.bmax[1] - mesh.bmin[1] : 0.0f);
        const float z = mesh.bmin[2] + ((corner & 4u) ? mesh.bmax[2] - mesh.bmin[2] : 0.0f);
        const float world[3] = {
            found->second.info.transform[0] * x + found->second.info.transform[1] * y +
                found->second.info.transform[2] * z + found->second.info.transform[3],
            found->second.info.transform[4] * x + found->second.info.transform[5] * y +
                found->second.info.transform[6] * z + found->second.info.transform[7],
            found->second.info.transform[8] * x + found->second.info.transform[9] * y +
                found->second.info.transform[10] * z + found->second.info.transform[11],
        };
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(world[axis])) return false;
            lo[axis] = std::min(lo[axis], world[axis]);
            hi[axis] = std::max(hi[axis], world[axis]);
        }
    }
    float extent[3];
    for (int axis = 0; axis < 3; ++axis) {
        center[axis] = (lo[axis] + hi[axis]) * 0.5f;
        extent[axis] = hi[axis] - lo[axis];
    }
    radius = 0.5f * std::sqrt(extent[0] * extent[0] + extent[1] * extent[1] +
        extent[2] * extent[2]);
    if (!std::isfinite(radius) || radius <= 0.0f) radius = 1.0f;
    return true;
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
