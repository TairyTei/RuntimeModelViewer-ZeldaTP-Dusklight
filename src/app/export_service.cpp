#include "app/export_service.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "export/glb_exporter.hpp"
#include "mods/svc/file.h"
#include "shared/common.hpp"
#include "shared/core.hpp"

namespace mi {
namespace {

struct PendingExport {
    std::vector<uint8_t> bytes;
    uint64_t modelKey = 0;
    int32_t soloShape = -1;
};

void on_export_folder_selected(ModContext* ctx, ModResult status,
    const char* const* locations, uint32_t locationCount, const char* error, void* userData) {
    std::unique_ptr<PendingExport> pending(static_cast<PendingExport*>(userData));
    if (status == MOD_UNAVAILABLE) return;
    if (status != MOD_OK) {
        logf(2, "GLB export folder selection failed: %s",
            error != nullptr ? error : "file service error");
        return;
    }
    if (pending == nullptr || locations == nullptr || locationCount == 0 || locations[0] == nullptr) {
        logf(2, "GLB export did not receive a destination folder");
        return;
    }

    const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    static std::atomic<uint64_t> exportSequence{0};
    const uint64_t sequence = exportSequence.fetch_add(1, std::memory_order_relaxed) + 1;
    char filename[128];
    if (pending->soloShape < 0) {
        std::snprintf(filename, sizeof(filename), "model_%06llx_all_%lld_%llu.glb",
            static_cast<unsigned long long>(pending->modelKey & 0xFFFFFF),
            static_cast<long long>(timestamp), static_cast<unsigned long long>(sequence));
    } else {
        std::snprintf(filename, sizeof(filename), "model_%06llx_shape_%d_%lld_%llu.glb",
            static_cast<unsigned long long>(pending->modelKey & 0xFFFFFF),
            pending->soloShape, static_cast<long long>(timestamp),
            static_cast<unsigned long long>(sequence));
    }

    const char* destination = nullptr;
    const ModResult createResult = svc_file->create_child(
        ctx, locations[0], filename, &destination);
    if (createResult != MOD_OK || destination == nullptr) {
        logf(2, "could not create GLB in the selected folder (error %d)", createResult);
        return;
    }
    const ModResult writeResult = svc_file->write_all(
        ctx, destination, pending->bytes.data(), pending->bytes.size());
    if (writeResult != MOD_OK) {
        logf(2, "could not write GLB file %s (error %d)", filename, writeResult);
        return;
    }
    logf(0, "exported %zu bytes as %s", pending->bytes.size(), filename);
}

}  // namespace

void request_glb_export(int32_t soloShape) {
    const std::shared_ptr<const MeshData> mesh = current_mesh();
    if (mesh == nullptr) {
        logf(1, "GLB export requested without a selected model");
        return;
    }

    PendingExport* pending = new PendingExport{};
    pending->modelKey = mesh->key;
    pending->soloShape = soloShape;
    std::string error;
    if (!gltf::build_glb(*mesh, soloShape, pending->bytes, error)) {
        logf(2, "GLB export failed: %s", error.c_str());
        delete pending;
        return;
    }

    FilePickOptions options = FILE_PICK_OPTIONS_INIT;
    const ModResult result = svc_file->pick_folder(
        mod_ctx, &options, on_export_folder_selected, pending);
    if (result != MOD_OK) {
        logf(2, "could not open GLB export folder picker (error %d)", result);
        delete pending;
    }
}

}  // namespace mi
