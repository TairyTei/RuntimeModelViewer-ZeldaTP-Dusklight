// Composition root: imports Dusklight services and wires the application modules together.
#include "mods/service.hpp"
#include "mods/svc/camera.h"
#include "mods/svc/file.h"
#include "mods/svc/gfx.h"
#include "mods/svc/hook.h"
#include "mods/svc/log.h"
#include "mods/svc/ui.h"
#include "mods/svc/window.h"

#include "game/j3d_hook.hpp"
#include "app/export_service.hpp"
#include "gui/dusklight_ui.hpp"
#include "preview/preview.hpp"
#include "shared/common.hpp"
#include "shared/core.hpp"

DEFINE_MOD();
IMPORT_SERVICE(LogService, svc_log);
IMPORT_SERVICE(UiService, svc_ui);
IMPORT_SERVICE(WindowService, svc_window);
IMPORT_SERVICE(FileService, svc_file);
IMPORT_SERVICE(GfxService, svc_gfx);
IMPORT_SERVICE(CameraService, svc_camera);
IMPORT_SERVICE(HookService, svc_hook);

namespace {

GfxStageHookHandle g_cameraHook = 0;
GfxStageHookHandle g_frameHook = 0;

// World-camera stage: remember where the camera is and where it looks (for "Pick: Screen center").
void on_scene_begin(ModContext*, const GfxStageContext* stage, void*) {
    if (stage == nullptr || stage->struct_size < sizeof(GfxStageContext) || stage->game_view == nullptr) return;
    CameraInfo info = CAMERA_INFO_INIT;
    if (svc_camera->get_camera(mod_ctx, stage->game_view, &info) != MOD_OK) return;
    mi::CameraSnap snap;
    snap.valid = true;
    for (int i = 0; i < 3; ++i) snap.eye[i] = info.eye[i];
    // column 2 of world_from_view is the camera's +Z; the camera looks down -Z
    snap.fwd[0] = -info.world_from_view[8];
    snap.fwd[1] = -info.world_from_view[9];
    snap.fwd[2] = -info.world_from_view[10];
    mi::set_camera(snap);
}

}  // namespace

extern "C" {

MOD_EXPORT ModResult mod_initialize(ModError* error) {
    mi::register_selector({"Screen center", mi::pick_screen_center});
    mi::preview_set_export_handler(mi::request_glb_export);

    if (mi::j3d_install_hooks() != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "could not hook J3DModel::entry");
    }

    GfxStageHookDesc sd = GFX_STAGE_HOOK_DESC_INIT;
    sd.callback = on_scene_begin;
    if (svc_gfx->register_stage_hook(mod_ctx, GFX_STAGE_SCENE_BEGIN, &sd, &g_cameraHook) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register camera stage hook");
    }
    sd.callback = mi::preview_frame;
    if (svc_gfx->register_stage_hook(mod_ctx, GFX_STAGE_FRAME_AFTER_HUD, &sd, &g_frameHook) != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register frame stage hook");
    }

    if (mi::ui_register() != MOD_OK) {
        return mods::set_error(error, MOD_ERROR, "failed to register UI");
    }
    mi::logf(0, "Model Inspector ready: open it from the Mods window or the in-game menu bar");
    return MOD_OK;
}

MOD_EXPORT ModResult mod_update(ModError* error) {
    mi::begin_frame();
    return mi::preview_update(error);
}

MOD_EXPORT ModResult mod_shutdown(ModError*) {
    if (g_frameHook != 0) svc_gfx->unregister_stage_hook(mod_ctx, g_frameHook);
    if (g_cameraHook != 0) svc_gfx->unregister_stage_hook(mod_ctx, g_cameraHook);
    g_frameHook = g_cameraHook = 0;
    mi::preview_set_export_handler(nullptr);
    mi::ui_shutdown();
    mi::preview_shutdown();
    return MOD_OK;
}

}
