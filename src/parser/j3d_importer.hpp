#pragma once

#include <string>

#include "shared/mesh.hpp"

class J3DModel;

namespace mi::parser {

ModelInfo describe_j3d_model(::J3DModel* model, bool includePoseSignature = false);
bool snapshot_j3d_model(::J3DModel* model, MeshData& out, std::string& err,
    const MeshData* previous, bool verbose);

}  // namespace mi::parser
