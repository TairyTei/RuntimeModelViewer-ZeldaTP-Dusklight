#include <cstdio>
#include <memory>
#include <string>

#include "mods/svc/hook.h"
#include "mods/svc/hook.hpp"

#include "game/j3d_hook.hpp"
#include "parser/j3d_importer.hpp"
#include "shared/common.hpp"
#include "shared/core.hpp"

#include "global.h"
#include "JSystem/J3DGraphAnimator/J3DModel.h"

DEFINE_HOOK(&J3DModel::entry, J3DModelEntryHook);

namespace mi {
namespace {

HookAction on_model_entry(ModContext*, void* args, void*, void*) {
    J3DModel* model = mods::arg<J3DModel*>(args, 0);
    if (model == nullptr) return HOOK_CONTINUE;

    ModelInfo info = parser::describe_j3d_model(model);
    if (info.key == 0) return HOOK_CONTINUE;
    registry_note(info);

    const bool selectedRequest = take_pending(info.key);
    const bool sceneRequest = take_scene_capture_request(info.key);
    if (!selectedRequest && !sceneRequest) return HOOK_CONTINUE;
    info = parser::describe_j3d_model(model, true);
    registry_note(info);

    if (sceneRequest && info.geometryCacheable) {
        const std::shared_ptr<const MeshData> cached = scene_model_mesh(info.key);
        if (cached != nullptr && cached->dataKey == info.dataKey &&
            cached->geometryCacheable && cached->poseKey == info.poseKey &&
            update_scene_model_info(info)) {
            return HOOK_CONTINUE;
        }
    }

    static uint64_t lastLiveFailureKey = 0;
    auto mesh = std::make_shared<MeshData>();
    mesh->key = info.key;
    mesh->dataKey = info.dataKey;
    mesh->poseKey = info.poseKey;
    mesh->geometryCacheable = info.geometryCacheable;
    std::shared_ptr<const MeshData> previous = selected_mesh();
    if (previous == nullptr || previous->key != info.key || previous->dataKey != info.dataKey) {
        previous = scene_model_mesh(info.key);
    }
    const bool sameData = previous != nullptr && previous->key == info.key &&
        previous->dataKey == info.dataKey;
    const bool verbose = selectedRequest && (!view().liveUpdate || !sameData);

    char title[96];
    std::snprintf(title, sizeof(title), "model %06llx (data %06llx)",
        static_cast<unsigned long long>(info.key & 0xFFFFFF),
        static_cast<unsigned long long>(info.dataKey & 0xFFFFFF));
    mesh->title = title;
    std::string error;
    if (verbose) logf(0, "snapshot starting for %llx",
        static_cast<unsigned long long>(info.key));

    if (parser::snapshot_j3d_model(model, *mesh, error,
            sameData ? previous.get() : nullptr, verbose)) {
        if (verbose) {
            logf(0, "snapshot %s: %u tris, %zu shapes, %zu textures", title,
                mesh->triCount, mesh->subs.size(), mesh->textures->size());
        }
        if (lastLiveFailureKey == info.key) lastLiveFailureKey = 0;
        if (selectedRequest) publish_mesh(mesh);
        if (sceneRequest) publish_scene_mesh(info, std::move(mesh));
    } else {
        if (verbose || lastLiveFailureKey != info.key) {
            logf(2, "snapshot of %s failed: %s", title, error.c_str());
        }
        lastLiveFailureKey = info.key;
    }
    return HOOK_CONTINUE;
}

}  // namespace

ModResult j3d_install_hooks() {
    return mods::hook::add_pre<J3DModelEntryHook>(on_model_entry);
}

}  // namespace mi
