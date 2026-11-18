#pragma once
// Shared model registry, snapshots, and view state. No game, GUI, or GPU types in this layer.
//
// EXTENDING SELECTION: add a mi::Selector and register it from mod_initialize; the application
// service handles the resulting selection and opens the preview.
#include <cstdint>
#include <memory>
#include <vector>

#include "shared/mesh.hpp"

namespace mi {

struct CameraSnap {
    bool valid = false;
    float eye[3] = {0, 0, 0};
    float fwd[3] = {0, 0, -1};
};

struct PickContext {
    CameraSnap cam;
    const std::vector<ModelInfo>* models = nullptr;  // nearest to the camera first
};

struct Selector {
    const char* label;
    // Return true and set *outKey when something was picked.
    bool (*pick)(const PickContext& ctx, uint64_t* outKey);
};

enum class PreviewLayout : int {
    TwoD = 0,
    ThreeD = 1,
    Both = 2,
};

struct ViewState {
    int mode = 1;  // 0-4 are 3D display modes
    PreviewLayout layout = PreviewLayout::ThreeD;
    bool autoRotate = true;
    bool nearest = false;
    bool flyingCamera = false;
    bool flyPositionInitialized = false;
    bool liveUpdate = false;
    float yaw = 30.0f;     // degrees
    int pitch = 15;        // degrees
    int zoomPct = 100;
    int uvZoomPct = 100;
    int flySpeedPct = 500;
    float flyPosition[3] = {0.0f, 0.0f, 0.0f};
    float uvPanX = 0.0f;
    float uvPanY = 0.0f;
    int uvTileOffsetX = 0;
    int uvTileOffsetY = 0;
    int upAxis = 0;        // 0 = Y-up, 1 = Z-up, 2 = X-up
    int solo = -1;         // shape index shown alone, -1 = all
    int uvTex = -1;        // texture index shown behind the UV map, -1 = none
    int texturePage = 0;
    bool textureMenuOpen = false;
    int shapePage = 0;
    bool shapeMenuOpen = false;
};

// ---- frame / registry (game thread) ----
void begin_frame();  // call once per frame from mod_update
uint64_t frame_no();
void registry_note(const ModelInfo& info);  // called by the J3D hook for every model entry
std::vector<ModelInfo> registry_snapshot();  // sorted nearest-first, stale entries dropped

// ---- selection ----
void select_model(uint64_t key);   // select only
uint64_t selected_key();
void request_refresh();            // re-snapshot the selected model (current pose)
bool take_pending(uint64_t key);   // hook side: true once when `key` should be snapshotted

// ---- mesh snapshot ----
void publish_mesh(std::shared_ptr<MeshData> mesh);
std::shared_ptr<const MeshData> current_mesh();
uint64_t mesh_version();

// ---- view / camera ----
ViewState& view();
void set_camera(const CameraSnap& cam);
CameraSnap camera();

// ---- selectors ----
void register_selector(const Selector& s);
const std::vector<Selector>& selectors();
bool pick_screen_center(const PickContext& ctx, uint64_t* outKey);
}  // namespace mi
