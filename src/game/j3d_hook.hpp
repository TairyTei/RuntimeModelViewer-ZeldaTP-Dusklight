#pragma once
#include "mods/api.h"

namespace mi {
// Hooks J3DModel::entry so every model drawn in a frame is registered, and so the selected
// model can be snapshotted while it is guaranteed to be alive.
ModResult j3d_install_hooks();
}  // namespace mi
