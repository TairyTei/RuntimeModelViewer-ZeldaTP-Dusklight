#pragma once
#include "mods/api.h"

struct GfxStageContext;

namespace mi {

enum class PreviewLayout : int;

ModResult preview_open();
void preview_close();
bool preview_is_open();
void set_flying_camera(bool enabled);
void set_live_update(bool enabled);
void set_preview_layout(PreviewLayout layout);

// Stage callback (register at GFX_STAGE_FRAME_AFTER_HUD): pushes one frame to the preview window.
void preview_frame(ModContext* ctx, const GfxStageContext* stage, void* user);
// mod_update: recreates a lost present target.
ModResult preview_update(ModError* error);
// mod_shutdown: closes the window and releases GPU objects.
void preview_shutdown();

}  // namespace mi
