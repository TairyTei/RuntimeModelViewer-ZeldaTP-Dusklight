#include "dlparse.hpp"
#include "common.hpp"

#include <algorithm>
#include <cstdio>

namespace mi {
namespace {

inline uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}
inline uint32_t be16(const uint8_t* p) { return (uint32_t(p[0]) << 8) | p[1]; }

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
inline uint32_t le16(const uint8_t* p) { return p[0] | (p[1] << 8); }
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
        if (d.attr == 13 && d.type != 1) {
            out.uvOff = static_cast<int>(off);
            out.uvBytes = d.type == 2 ? 1 : 2;
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

    if (depth > 10) return false;
    if (dl == nullptr || size == 0) return false;

    size_t p = 0;
    std::vector<VertexRef> v;
    while (p < size) {
        const uint8_t op = dl[p++];
        if (op == 0x00) continue;  // NOP
        if (op == 0x08) { p += 5; continue; }
        if (op == 0x10) {  // XF
            if (p + 4 > size) break;
            const size_t n = be16(dl + p) + 1;
            p += 4 + n * 4;
            continue;
        }
        if (op == 0x20 || op == 0x28 || op == 0x30 || op == 0x38) { p += 4; continue; }
        if (op == 0x61) { p += 4; continue; }

        // ДОБАВЛЯЕМ 0x68 и 0x70 (LOAD_INDX_C / D)
        if (op == 0x50 || op == 0x60 || op == 0x68 || op == 0x70) {
            p += 2;
            continue;
        }
        if (op == 0x40 || op == 0x48) {  // CALL_DL
            p += 3;
            continue;
        }
        if (op & 0x80) { // Примитивы
            const int prim = op & 0xF8;
            if (p + 2 > size) { err = "truncated"; return false; }
            uint32_t n = be16(dl + p);
            if (n > 10000) { err = "too many vertices"; return false; } // Защита
            p += 2;
            if (p + static_cast<size_t>(n) * layout.stride > size) { err = "truncated"; return false; }

            v.clear();
            v.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                const uint8_t* r = dl + p + static_cast<size_t>(i) * layout.stride;
                VertexRef ref;
                if (layout.pnmtxOff >= 0) ref.pnmtx = r[layout.pnmtxOff];
                ref.posIdx = layout.posBytes == 1 ? r[layout.posOff]
                                                  : static_cast<int>(le16(r + layout.posOff)); // Должно быть le16!
                if (layout.uvOff >= 0) {
                    ref.uvIdx = layout.uvBytes == 1 ? r[layout.uvOff]
                                                    : static_cast<int>(le16(r + layout.uvOff)); // Должно быть le16!
                }
                if (i < 3) {
                    logf(0, "  vert %u: posIdx=%d uvIdx=%d pnmtx=%d", i, ref.posIdx, ref.uvIdx, ref.pnmtx);
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
