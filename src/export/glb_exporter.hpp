#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "shared/mesh.hpp"

namespace mi::gltf {

bool build_glb(const MeshData& mesh, int32_t soloShape, std::vector<uint8_t>& out,
    std::string& error);

}  // namespace mi::gltf
