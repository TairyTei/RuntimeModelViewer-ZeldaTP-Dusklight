// UI: a three-tab window built with UiService.
//   Scene  - models drawn this frame (nearest first) + "Pick: ..." buttons. Press a row to inspect.
//   Model  - tree of the selected model: model -> shapes -> textures. Press a shape to show it
//            alone (and show its texture behind the UV map); press a texture to view it in UV mode.
//   View   - display mode, rotation, zoom, filtering.
#include "ui.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "common.hpp"
#include "core.hpp"
#include "preview.hpp"
#include "mods/svc/ui.h"

namespace mi {
namespace {

UiWindowHandle g_win = 0;
UiMenuTabHandle g_menuTab = 0;
UiListHandle g_sceneList = 0, g_modelList = 0;
UiElementHandle g_sceneInfo = 0, g_modelInfo = 0;
uint64_t g_lastSceneRefresh = 0;
uint64_t g_lastModelVersion = ~0ull;

std::vector<std::string> g_sceneLabels, g_modelLabels, g_pickLabels;

constexpr uint64_t kShapeKey = 1000;
constexpr uint64_t kTexKey = 2000;

// ---------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------
void add_button(UiElementHandle pane, const char* label, UiPressedFn fn, void* ud = nullptr) {
    UiControlDesc c = UI_CONTROL_DESC_INIT;
    c.kind = UI_CONTROL_BUTTON;
    c.label = label;
    c.on_pressed = fn;
    c.user_data = ud;
    svc_ui->pane_add_control(mod_ctx, pane, &c, nullptr);
}

void get_bool(ModContext*, void* ud, UiControlValue* out) { out->bool_value = *static_cast<bool*>(ud); }
void set_bool(ModContext*, void* ud, const UiControlValue* v) { *static_cast<bool*>(ud) = v->bool_value; }
void get_int(ModContext*, void* ud, UiControlValue* out) { out->int_value = *static_cast<int*>(ud); }
void set_int(ModContext*, void* ud, const UiControlValue* v) { *static_cast<int*>(ud) = static_cast<int>(v->int_value); }

void add_toggle(UiElementHandle pane, const char* label, bool* flag) {
    UiControlDesc c = UI_CONTROL_DESC_INIT;
    c.kind = UI_CONTROL_TOGGLE;
    c.label = label;
    c.get = get_bool;
    c.set = set_bool;
    c.user_data = flag;
    svc_ui->pane_add_control(mod_ctx, pane, &c, nullptr);
}

void add_number(UiElementHandle pane, const char* label, UiControlGetFn get, UiControlSetFn set,
    void* ud, int64_t lo, int64_t hi, int64_t step, const char* suffix = nullptr) {
    UiControlDesc c = UI_CONTROL_DESC_INIT;
    c.kind = UI_CONTROL_NUMBER;
    c.label = label;
    c.get = get;
    c.set = set;
    c.user_data = ud;
    c.min = lo;
    c.max = hi;
    c.step = step;
    c.suffix = suffix;
    svc_ui->pane_add_control(mod_ctx, pane, &c, nullptr);
}

void get_yaw(ModContext*, void*, UiControlValue* out) { out->int_value = static_cast<int64_t>(view().yaw) % 360; }
void set_yaw(ModContext*, void*, const UiControlValue* v) { view().yaw = static_cast<float>(v->int_value); }

void toast(const char* text) {
    UiToastDesc t = UI_TOAST_DESC_INIT;
    t.body_rml = text;
    svc_ui->push_toast(mod_ctx, &t);
}

// ---------------------------------------------------------------------------------------------
// Scene tab
// ---------------------------------------------------------------------------------------------
void refresh_scene_list() {
    g_lastSceneRefresh = frame_no();
    if (g_sceneList == 0) return;
    const std::vector<ModelInfo> models = registry_snapshot();
    const CameraSnap cam = camera();
    g_sceneLabels.clear();
    g_sceneLabels.reserve(models.size());
    for (const ModelInfo& m : models) {
        float d = 0.0f;
        if (cam.valid) {
            const float dx = m.pos[0] - cam.eye[0], dy = m.pos[1] - cam.eye[1], dz = m.pos[2] - cam.eye[2];
            d = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        char buf[128];
        std::snprintf(buf, sizeof(buf), "d=%.0f  data@%06llx  joints %u  shapes %u  verts %u", d,
            static_cast<unsigned long long>(m.dataKey & 0xFFFFFF), m.joints, m.shapes, m.verts);
        g_sceneLabels.push_back(buf);
    }
    std::vector<UiListItem> items;
    items.reserve(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
        UiListItem it = UI_LIST_ITEM_INIT;
        it.key = models[i].key;
        it.label = g_sceneLabels[i].c_str();
        items.push_back(it);
    }
    svc_ui->list_set_items(mod_ctx, g_sceneList, items.data(), items.size());
    if (g_sceneInfo != 0) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%zu models drawn recently, nearest first.<br/>Press a row to open it in the preview window.", models.size());
        svc_ui->elem_set_rml(mod_ctx, g_sceneInfo, buf);
    }
}

void on_scene_row(ModContext*, UiListHandle, uint64_t key, void*) {
    logf(0, "UI click: selecting key=%llx", (unsigned long long)key);
    inspect_model(key);
}
bool scene_row_selected(ModContext*, UiListHandle, uint64_t key, void*) { return key == selected_key(); }

void on_pick(ModContext*, void* ud) {
    if (!run_selector(reinterpret_cast<uintptr_t>(ud))) {
        toast("Nothing found with this method.");
    }
}
void on_refresh_scene(ModContext*, void*) { refresh_scene_list(); }

ModResult build_scene_tab(ModContext*, UiWindowHandle, UiElementHandle left, UiElementHandle right, void*, ModError*) {
    g_sceneList = 0;
    g_sceneInfo = 0;
    svc_ui->pane_add_section(mod_ctx, left, "Models in view");
    add_button(left, "Refresh list", on_refresh_scene);
    const auto& sels = selectors();
    g_pickLabels.clear();
    g_pickLabels.reserve(sels.size());
    for (size_t i = 0; i < sels.size(); ++i) {
        g_pickLabels.push_back(std::string("Pick: ") + sels[i].label);
        add_button(left, g_pickLabels.back().c_str(), on_pick, reinterpret_cast<void*>(i));
    }
    UiListDesc ld = UI_LIST_DESC_INIT;
    ld.on_pressed = on_scene_row;
    ld.is_selected = scene_row_selected;
    svc_ui->pane_add_list(mod_ctx, left, &ld, &g_sceneList);
    svc_ui->pane_add_rml(mod_ctx, right, "", &g_sceneInfo);
    refresh_scene_list();
    return MOD_OK;
}

ModResult update_scene_tab(ModContext*, void*, ModError*) {
    // if (g_sceneList != 0 && frame_no() >= g_lastSceneRefresh + 20) refresh_scene_list();
    return MOD_OK;
}

// ---------------------------------------------------------------------------------------------
// Model tab
// ---------------------------------------------------------------------------------------------
void refresh_model_list() {
    const std::shared_ptr<const MeshData> mesh = current_mesh();
    g_lastModelVersion = mesh ? mesh->version : 0;
    if (g_modelList == 0) return;
    std::vector<uint64_t> keys;
    g_modelLabels.clear();
    if (mesh) {
        g_modelLabels.reserve(2 + mesh->subs.size() + mesh->textures.size());
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s  -  %u tris  (press: show all shapes)", mesh->title.c_str(), mesh->triCount);
        g_modelLabels.push_back(buf);
        keys.push_back(0);
        for (const SubMesh& s : mesh->subs) {
            std::snprintf(buf, sizeof(buf), "   Shape %u  %s  tris %u  tex %d", s.shapeIndex,
                s.label.empty() ? "-" : s.label.c_str(), s.vertexCount / 3, s.texture);
            g_modelLabels.push_back(buf);
            keys.push_back(kShapeKey + s.shapeIndex);
        }
        for (size_t i = 0; i < mesh->textures.size(); ++i) {
            std::snprintf(buf, sizeof(buf), "   Texture %zu  %s", i, mesh->textures[i].desc.c_str());
            g_modelLabels.push_back(buf);
            keys.push_back(kTexKey + i);
        }
    } else {
        g_modelLabels.push_back("(nothing selected - pick a model in the Scene tab)");
        keys.push_back(0);
    }
    std::vector<UiListItem> items;
    for (size_t i = 0; i < g_modelLabels.size(); ++i) {
        UiListItem it = UI_LIST_ITEM_INIT;
        it.key = keys[i];
        it.label = g_modelLabels[i].c_str();
        items.push_back(it);
    }
    svc_ui->list_set_items(mod_ctx, g_modelList, items.data(), items.size());
    if (g_modelInfo != 0) {
        std::string rml;
        if (mesh) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "%s<br/>joints %u, shapes %u, materials %u<br/>%u vertices, %u triangles<br/>%zu textures<br/><br/>"
                "Press a shape to show it alone. Press a texture to see it behind the UV map (View tab, mode UV Map).",
                mesh->title.c_str(), mesh->joints, mesh->shapes, mesh->materials, mesh->vertexCount, mesh->triCount,
                mesh->textures.size());
            rml = buf;
        }
        svc_ui->elem_set_rml(mod_ctx, g_modelInfo, rml.c_str());
    }
}

void on_model_row(ModContext*, UiListHandle, uint64_t key, void*) {
    ViewState& v = view();
    if (key == 0) {
        v.solo = -1;
    } else if (key >= kTexKey) {
        v.uvTex = static_cast<int>(key - kTexKey);
    } else if (key >= kShapeKey) {
        const int idx = static_cast<int>(key - kShapeKey);
        if (v.solo == idx) {
            v.solo = -1;
        } else {
            v.solo = idx;
            if (auto mesh = current_mesh()) {
                for (const SubMesh& s : mesh->subs) {
                    if (static_cast<int>(s.shapeIndex) == idx) {
                        v.uvTex = s.texture;
                        break;
                    }
                }
            }
        }
    }
}

bool model_row_selected(ModContext*, UiListHandle, uint64_t key, void*) {
    const ViewState& v = view();
    if (key == 0) return v.solo < 0;
    if (key >= kTexKey) return v.uvTex == static_cast<int>(key - kTexKey);
    return v.solo == static_cast<int>(key - kShapeKey);
}

void on_refresh_pose(ModContext*, void*) { request_refresh(); }

ModResult build_model_tab(ModContext*, UiWindowHandle, UiElementHandle left, UiElementHandle right, void*, ModError*) {
    g_modelList = 0;
    g_modelInfo = 0;
    svc_ui->pane_add_section(mod_ctx, left, "Selected model");
    add_button(left, "Re-snapshot (current pose)", on_refresh_pose);
    UiListDesc ld = UI_LIST_DESC_INIT;
    ld.on_pressed = on_model_row;
    ld.is_selected = model_row_selected;
    svc_ui->pane_add_list(mod_ctx, left, &ld, &g_modelList);
    svc_ui->pane_add_rml(mod_ctx, right, "", &g_modelInfo);
    refresh_model_list();
    return MOD_OK;
}

ModResult update_model_tab(ModContext*, void*, ModError*) {
    if (g_modelList != 0 && mesh_version() != g_lastModelVersion) refresh_model_list();
    return MOD_OK;
}

// ---------------------------------------------------------------------------------------------
// View tab
// ---------------------------------------------------------------------------------------------
const char* const kModes[] = {"Solid", "Textured", "Solid + Wire", "Textured + Wire", "Wire", "UV Map"};

void get_mode(ModContext*, void*, UiControlValue* out) { out->int_value = view().mode; }
void set_mode(ModContext*, void*, const UiControlValue* v) { view().mode = static_cast<int>(v->int_value); }
void on_toggle_preview(ModContext*, void*) {
    if (preview_is_open()) preview_close();
    else preview_open();
}

ModResult build_view_tab(ModContext*, UiWindowHandle, UiElementHandle left, UiElementHandle right, void*, ModError*) {
    ViewState& v = view();
    svc_ui->pane_add_section(mod_ctx, left, "Preview window");
    add_button(left, "Open / close preview window", on_toggle_preview);

    UiControlDesc m = UI_CONTROL_DESC_INIT;
    m.kind = UI_CONTROL_DROPDOWN;
    m.label = "Display mode";
    m.get = get_mode;
    m.set = set_mode;
    m.options = kModes;
    m.option_count = sizeof(kModes) / sizeof(kModes[0]);
    svc_ui->pane_add_control(mod_ctx, left, &m, nullptr);

    add_toggle(left, "Auto-rotate", &v.autoRotate);
    add_number(left, "Yaw", get_yaw, set_yaw, nullptr, 0, 359, 10, " deg");
    add_number(left, "Pitch", get_int, set_int, &v.pitch, -89, 89, 5, " deg");
    add_number(left, "Zoom", get_int, set_int, &v.zoomPct, 10, 800, 10, " %");
    add_toggle(left, "Nearest texture filtering", &v.nearest);

    svc_ui->pane_add_rml(mod_ctx, right,
        "Modes 0-4 show the 3D model, UV Map shows the unwrap of the selected shapes over a texture.<br/><br/>"
        "The preview window has no mouse control yet; use Yaw / Pitch / Zoom here.", nullptr);
    return MOD_OK;
}

// ---------------------------------------------------------------------------------------------
// window / panel entry points
// ---------------------------------------------------------------------------------------------
void on_window_closed(ModContext*, UiWindowHandle, void*) {
    g_win = 0;
    g_sceneList = g_modelList = 0;
    g_sceneInfo = g_modelInfo = 0;
}

void on_open(ModContext*, void*) { ui_open_inspector(); }

ModResult build_mods_panel(ModContext*, UiElementHandle panel, void*, ModError*) {
    add_button(panel, "Open Model Inspector", on_open);
    svc_ui->pane_add_text(mod_ctx, panel,
        "Lists the 3D models the game draws, and shows the selected one in a separate preview window.", nullptr);
    return MOD_OK;
}

}  // namespace

void ui_open_inspector() {
    if (g_win != 0) return;
    UiTabDesc tabs[3] = {UI_TAB_DESC_INIT, UI_TAB_DESC_INIT, UI_TAB_DESC_INIT};
    tabs[0].title = "Scene";
    tabs[0].build = build_scene_tab;
    tabs[0].update = update_scene_tab;
    tabs[1].title = "Model";
    tabs[1].build = build_model_tab;
    tabs[1].update = update_model_tab;
    tabs[2].title = "View";
    tabs[2].build = build_view_tab;
    UiWindowDesc wd = UI_WINDOW_DESC_INIT;
    wd.tabs = tabs;
    wd.tab_count = 3;
    wd.on_closed = on_window_closed;
    if (svc_ui->window_push(mod_ctx, &wd, &g_win) != MOD_OK) {
        g_win = 0;
        logf(2, "failed to open the inspector window");
    }
}

ModResult ui_register() {
    UiModsPanelDesc pd = UI_MODS_PANEL_DESC_INIT;
    pd.build = build_mods_panel;
    ModResult r = svc_ui->register_mods_panel(mod_ctx, &pd);
    if (r != MOD_OK) return r;
    UiMenuTabDesc td = UI_MENU_TAB_DESC_INIT;
    td.label = "Model Inspector";
    td.on_selected = on_open;
    svc_ui->register_menu_tab(mod_ctx, &td, &g_menuTab);  // optional convenience
    return MOD_OK;
}

void ui_shutdown() {
    if (g_menuTab != 0) {
        svc_ui->unregister_menu_tab(mod_ctx, g_menuTab);
        g_menuTab = 0;
    }
    g_win = 0;
    g_sceneList = g_modelList = 0;
    g_sceneInfo = g_modelInfo = 0;
}

}  // namespace mi
