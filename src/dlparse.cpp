#include "dlparse.hpp"
#include "common.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace mi {
namespace {

inline uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}
inline uint32_t be16(const uint8_t* p) { return (uint32_t(p[0]) << 8) | p[1]; }
inline uint16_t host16(const uint8_t* p) {
    uint16_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

// NBT shares the normal slot in the vertex record.
int order_key(int attr) { return attr == 25 ? 10 : attr; }

int comp_size(int t) {
    switch (t) {
    case 0:
    case 1: return 1;
    case 2:
    case 3: return 2;
    case 4: return 4;
    default: return 0;
    }
}

VertexRef read_vertex_ref(const uint8_t* record, const VertexLayout& layout) {
    VertexRef ref;
    if (layout.pnmtxOff >= 0) ref.pnmtx = record[layout.pnmtxOff];
    ref.posIdx = layout.posBytes == 1 ? record[layout.posOff]
                                      : static_cast<int>(be16(record + layout.posOff));
    for (size_t i = 0; i < layout.uvOffs.size(); ++i) {
        if (layout.uvOffs[i] >= 0) {
            ref.uvIdx[i] = layout.uvBytes[i] == 1 ? record[layout.uvOffs[i]]
                                                  : static_cast<int>(be16(record + layout.uvOffs[i]));
        }
    }
    return ref;
}

int direct_bytes(int attr, const AttrFmt& f) {
    if (attr >= 0 && attr <= 8) return 1;
    if (attr == 9) return (f.cnt == 0 ? 2 : 3) * comp_size(f.type);
    if (attr == 10 || attr == 25) return (f.cnt == 0 ? 3 : 9) * comp_size(f.type);
    if (attr == 11 || attr == 12) {
        static const int s[6] = {2, 3, 4, 2, 3, 4};  // RGB565 RGB8 RGBX8 RGBA4 RGBA6 RGBA8
        return (f.type >= 0 && f.type < 6) ? s[f.type] : 0;
    }
    if (attr >= 13 && attr <= 20) return (f.cnt == 0 ? 1 : 2) * comp_size(f.type);
    return 0;
}

// Number of index values stored for one attribute of an indexed vertex.
int index_count(int attr, const AttrFmt& f) {
    if ((attr == 10 || attr == 25) && f.cnt == 2) return 3;  // NBT3
    return 1;
}

}  // namespace

bool build_vertex_layout(const std::vector<AttrDesc>& descs,
    const std::function<AttrFmt(int)>& fmtOf, VertexLayout& out, std::string& err) {
    out = VertexLayout{};
    std::vector<AttrDesc> sorted;
    for (const AttrDesc& d : descs) {
        if (d.type != 0) sorted.push_back(d);
    }
    std::stable_sort(sorted.begin(), sorted.end(), [](const AttrDesc& a, const AttrDesc& b) {
        return order_key(a.attr) < order_key(b.attr);
    });
    uint32_t off = 0;
    for (const AttrDesc& d : sorted) {
        const AttrFmt f = fmtOf(d.attr);
        int bytes = 0;
        if (d.type == 1) {
            bytes = direct_bytes(d.attr, f);
            if (bytes == 0) {
                err = "unsupported direct attribute " + std::to_string(d.attr);
                return false;
            }
        } else if (d.type == 2 || d.type == 3) {
            bytes = (d.type == 2 ? 1 : 2) * index_count(d.attr, f);
        } else {
            err = "bad attribute type";
            return false;
        }
        if (d.attr == 0 && d.type == 1) out.pnmtxOff = static_cast<int>(off);
        if (d.attr == 9) {
            if (d.type == 1) {
                err = "direct position data is not supported";
                return false;
            }
            out.posOff = static_cast<int>(off);
            out.posBytes = d.type == 2 ? 1 : 2;
        }
        if (d.attr >= 13 && d.attr <= 20 && d.type != 1) {
            const size_t texCoord = static_cast<size_t>(d.attr - 13);
            out.uvOffs[texCoord] = static_cast<int>(off);
            out.uvBytes[texCoord] = d.type == 2 ? 1 : 2;
        }
        off += static_cast<uint32_t>(bytes);
    }
    out.stride = off;
    // // В GX данные вершин выравниваются до 4 байт! (7 байт станут 8)
    // if (out.stride % 4 != 0) {
    //     out.stride = (out.stride + 3) & ~3;
    // }
    if (out.posOff < 0) {
        err = "shape has no position attribute";
        return false;
    }

    if (out.stride == 0) { // ДОБАВИТЬ ЭТО
        err = "zero vertex stride";
        return false;
    }
    return true;
}

bool parse_display_list(const uint8_t* dl, size_t size, const VertexLayout& layout,
    std::vector<VertexRef>& tris, std::string& err, int depth) {

    if (depth > 10) {
        err = "display list nesting limit exceeded";
        return false;
    }
    if (dl == nullptr || size == 0 || layout.stride == 0) {
        err = "invalid display list or vertex layout";
        return false;
    }

    size_t p = 0;
    std::vector<VertexRef> v;
    while (p < size) {
        const uint8_t op = dl[p++];
        if (op == 0x00 || op == 0x48) continue;  // NOP / invalidate vertex cache
        if (op == 0x08 || op == 0x61) {
            const size_t bytes = op == 0x08 ? 5 : 4;
            if (bytes > size - p) { err = "truncated register command"; return false; }
            p += bytes;
            continue;
        }
        if (op == 0x10) {  // XF
            if (size - p < 4) { err = "truncated XF command"; return false; }
            const size_t n = be16(dl + p) + 1;
            p += 4;
            if (n > (size - p) / 4) { err = "truncated XF data"; return false; }
            p += n * 4;
            continue;
        }
        if (op == 0x20 || op == 0x28 || op == 0x30 || op == 0x38) {
            if (size - p < 4) { err = "truncated indexed-load command"; return false; }
            p += 4;
            continue;
        }
        if (op == 0x40) {  // CALL_DL: address and byte count
            if (size - p < 8) { err = "truncated display-list call"; return false; }
            p += 8;
            continue;
        }
        if (op == 0x50) {  // Aurora command
            if (size - p < 9) { err = "truncated Aurora command"; return false; }
            const uint16_t subcommand = static_cast<uint16_t>(be16(dl + p));
            if (subcommand != 0x0041) {
                err = "unsupported Aurora display-list command";
                return false;
            }

            const int prim = dl[p + 2] & 0xF8;
            const uint32_t vertexCount = be16(dl + p + 3);
            const uint32_t indexCount = be32(dl + p + 5);
            logf(0,
                "Aurora DRAW_INDEXED: primitive=0x%02X vertices=%u indices=%u vertexStride=%u "
                "indexOffset=%zu vertexOffset=%zu",
                prim, vertexCount, indexCount, layout.stride, p + 9,
                p + 9 + static_cast<size_t>(indexCount) * sizeof(uint16_t));
            if (prim != 0x90 || indexCount % 3 != 0) {
                err = "invalid Aurora indexed triangle command";
                return false;
            }

            const size_t indexStart = p + 9;
            if (indexCount > (size - indexStart) / sizeof(uint16_t)) {
                err = "truncated Aurora index data";
                return false;
            }
            const size_t vertexStart = indexStart + static_cast<size_t>(indexCount) * sizeof(uint16_t);
            if (vertexCount > (size - vertexStart) / layout.stride) {
                err = "truncated Aurora vertex data";
                return false;
            }

            v.clear();
            v.reserve(vertexCount);
            for (uint32_t i = 0; i < vertexCount; ++i) {
                v.push_back(read_vertex_ref(dl + vertexStart + static_cast<size_t>(i) * layout.stride, layout));
            }
            for (uint32_t i = 0; i < indexCount; i += 3) {
                const uint16_t a = host16(dl + indexStart + static_cast<size_t>(i) * sizeof(uint16_t));
                const uint16_t b = host16(dl + indexStart + static_cast<size_t>(i + 1) * sizeof(uint16_t));
                const uint16_t c = host16(dl + indexStart + static_cast<size_t>(i + 2) * sizeof(uint16_t));
                if (a >= vertexCount || b >= vertexCount || c >= vertexCount) {
                    err = "Aurora index is outside the vertex data";
                    return false;
                }
                tris.push_back(v[a]);
                tris.push_back(v[b]);
                tris.push_back(v[c]);
            }
            p = vertexStart + static_cast<size_t>(vertexCount) * layout.stride;
            continue;
        }
        const int prim = op & 0xF8;
        if (prim == 0x80 || prim == 0x90 || prim == 0x98 || prim == 0xA0 ||
            prim == 0xA8 || prim == 0xB0 || prim == 0xB8) {
            if (size - p < 2) { err = "truncated primitive header"; return false; }
            uint32_t n = be16(dl + p);
            p += 2;
            if (n > (size - p) / layout.stride) { err = "truncated primitive data"; return false; }

            v.clear();
            v.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                const uint8_t* r = dl + p + static_cast<size_t>(i) * layout.stride;
                const VertexRef ref = read_vertex_ref(r, layout);
                if (i < 3) {
                    logf(0, "  vert %u: posIdx=%d uv0Idx=%d pnmtx=%d",
                        i, ref.posIdx, ref.uvIdx[0], ref.pnmtx);
                }
                v.push_back(ref);
            }
            p += static_cast<size_t>(n) * layout.stride;

            auto tri = [&](size_t a, size_t b, size_t c) {
                tris.push_back(v[a]); tris.push_back(v[b]); tris.push_back(v[c]);
            };
            switch (prim) {
            case 0x80: for (size_t i = 0; i + 3 < n; i += 4) { tri(i, i + 1, i + 2); tri(i, i + 2, i + 3); } break;
            case 0x90: for (size_t i = 0; i + 2 < n; i += 3) tri(i, i + 1, i + 2); break;
            case 0x98: for (size_t i = 2; i < n; ++i) { if (i & 1) tri(i - 1, i - 2, i); else tri(i - 2, i - 1, i); } break;
            case 0xA0: for (size_t i = 2; i < n; ++i) tri(0, i - 1, i); break;
            }
            continue;
        }
        char buf[64];
        std::snprintf(buf, sizeof(buf), "unknown display list opcode 0x%02X at %zu", op, p - 1);
        err = buf;
        return false;
    }
    return true;
}

}  // namespace mi
