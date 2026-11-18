#pragma once
// Plain-C++ snapshot of one J3D model. No game types in here.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mi {

struct ModelInfo {
    uint64_t key = 0;
    uint64_t dataKey = 0;
    float pos[3] = {0, 0, 0};
    float radius = 0;
    uint32_t joints = 0, shapes = 0, materials = 0, verts = 0;
    uint64_t lastFrame = 0;
};

struct TexImage {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> rgba;  // w * h * 4, row 0 = top
    std::string desc;           // "64x64 CMPR"
};

// One shape (= one draw call of the original model). Triangles are stored un-indexed.
struct SubMesh {
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;  // multiple of 3
    int32_t texture = -1;      // index into MeshData::textures, -1 = none
    uint32_t shapeIndex = 0;
    std::string label;         // "mat_name"
};

struct MeshData {
    uint64_t key = 0;  // J3DModel* the snapshot came from
    std::string title;
    // 6 floats per vertex: pos.xyz, uv.xy, shape id
    std::vector<float> vertices;
    std::vector<SubMesh> subs;
    std::shared_ptr<const std::vector<TexImage>> textures =
        std::make_shared<const std::vector<TexImage>>();
    uint64_t dataKey = 0;
    float bmin[3] = {0, 0, 0};
    float bmax[3] = {0, 0, 0};
    uint32_t joints = 0, shapes = 0, materials = 0, vertexCount = 0, triCount = 0;
    uint64_t version = 0;  // assigned when published
};

}  // namespace mi
