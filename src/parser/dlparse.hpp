#pragma once
// Parser for J3D shape display lists (GX FIFO command stream, big-endian).
// Also understands Dusklight's Aurora DRAW_INDEXED command and its host-endian index buffer.
// Pure C++: the caller supplies the vertex descriptor/format information.
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mi::parser {

// GX numbers used here: attr  PNMTXIDX=0 TEX0MTXIDX..TEX7MTXIDX=1..8 POS=9 NRM=10 CLR0=11 CLR1=12
//                             TEX0..TEX7=13..20 NBT=25 NULL=0xFF
//                       type  NONE=0 DIRECT=1 INDEX8=2 INDEX16=3
struct AttrDesc {
    int attr;
    int type;
};

struct AttrFmt {
    bool found = false;
    int cnt = 0;   // component count enum (POS_XY=0/XYZ=1, NRM_XYZ=0/NBT=1/NBT3=2, TEX_S=0/ST=1, ...)
    int type = 0;  // component type enum (U8=0 S8=1 U16=2 S16=3 F32=4; colors: RGB565=0 ... RGBA8=5)
    int frac = 0;
};

struct VertexLayout {
    uint32_t stride = 0;   // bytes per vertex record in the display list
    int pnmtxOff = -1;     // direct u8 position-matrix index (value = slot * 3)
    int posOff = -1;
    int posBytes = 0;      // 1 or 2 (INDEX8 / INDEX16)
    std::array<int, 8> uvOffs = {-1, -1, -1, -1, -1, -1, -1, -1};
    std::array<int, 8> uvBytes = {};
};

struct VertexRef {
    int pnmtx = -1;  // -1 when the vertex record has no matrix index
    int posIdx = 0;
    std::array<int, 8> uvIdx = {-1, -1, -1, -1, -1, -1, -1, -1};
};

bool build_vertex_layout(const std::vector<AttrDesc>& descs,
    const std::function<AttrFmt(int)>& fmtOf, VertexLayout& out, std::string& err);

// Appends triangles (3 VertexRefs each) to `tris`. Quads, strips and fans are triangulated;
// lines and points are skipped. Returns false (with err) on an unknown opcode or truncated data;
// triangles parsed before the problem are kept.
bool parse_display_list(const uint8_t* dl, size_t size, const VertexLayout& layout,
    std::vector<VertexRef>& tris, std::string& err, int depth = 0);

}  // namespace mi::parser
