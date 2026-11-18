#pragma once
#include "mods/api.h"
#include "mods/svc/gfx.h"

namespace mi {

ModResult preview_open();
void preview_close();
bool preview_is_open();

// Stage callback (register at GFX_STAGE_FRAME_AFTER_HUD): pushes one frame to the preview window.
void preview_frame(ModContext* ctx, const GfxStageContext* stage, void* user);
// mod_update: recreates a lost present target.
ModResult preview_update(ModError* error);
// mod_shutdown: closes the window and releases GPU objects.
void preview_shutdown();

}  // namespace mi
