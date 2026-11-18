// The ONLY file that touches game (J3D) types. If the port's headers differ from what this was
// written against, the compile errors will all be in here, mostly in the small accessor templates
// at the top.
// Include order matters: the game's global.h redefines nullptr, which breaks standard library
// headers included after it. So: standard headers and our own headers first, game headers after.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mods/svc/hook.h"
#include "mods/svc/hook.hpp"

#include "common.hpp"
#include "core.hpp"
#include "dlparse.hpp"
#include "gctex.hpp"
#include "j3d.hpp"

#include "global.h"

#include "JSystem/J3DGraphAnimator/J3DModel.h"
#include "JSystem/J3DGraphAnimator/J3DModelData.h"
#include "JSystem/J3DGraphBase/J3DMaterial.h"
#include "JSystem/J3DGraphBase/J3DShape.h"
#include "JSystem/J3DGraphBase/J3DShapeDraw.h"
#include "JSystem/J3DGraphBase/J3DTexture.h"
#include "JSystem/J3DGraphBase/J3DVertex.h"

DEFINE_HOOK(&J3DModel::entry, J3DModelEntryHook);

namespace mi {
namespace {

// ---------- field accessors (tolerate either naming style of the GX list structs) ----------
template <class E> int d_attr(const E& e) {
    if constexpr (requires { e.attr; }) return static_cast<int>(e.attr);
    else return static_cast<int>(e.mAttrib);
}
template <class E> int d_type(const E& e) {
    if constexpr (requires { e.type; }) return static_cast<int>(e.type);
    else return static_cast<int>(e.mType);
}
template <class E> int f_attr(const E& e) {
    if constexpr (requires { e.attr; }) return static_cast<int>(e.attr);
    else return static_cast<int>(e.mAttrib);
}
template <class E> int f_cnt(const E& e) {
    if constexpr (requires { e.cnt; }) return static_cast<int>(e.cnt);
    else return static_cast<int>(e.mCompCnt);
}
template <class E> int f_type(const E& e) {
    if constexpr (requires { e.type; }) return static_cast<int>(e.type);
    else return static_cast<int>(e.mCompType);
}
template <class E> int f_frac(const E& e) {
    if constexpr (requires { e.frac; }) return static_cast<int>(e.frac);
    else return static_cast<int>(e.mCompShift);
}
template <class M> const char* mat_name(const M& m) {
    if constexpr (requires { m.mMaterialName; }) return m.mMaterialName;
    else return nullptr;
}
template <class V> uint32_t array_stride(V& vd, int attr, uint32_t fallback) {
    if constexpr (requires { vd.getVtxArrStride(static_cast<GXAttr>(attr)); }) {
        const uint32_t s = vd.getVtxArrStride(static_cast<GXAttr>(attr));
        return s != 0 ? s : fallback;
    } else {
        return fallback;
    }
}
template <class D> const uint8_t* dl_data(D* d) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(d->getDisplayList());
    return ptr; // Возвращаем как есть. Если там мусор, мы просто пропустим шейп, но не вылетим.
}
template <class D> uint32_t dl_size(D* d) { return static_cast<uint32_t>(d->getDisplayListSize()); }

// ---------- raw array reading ----------
inline uint16_t rd16(const uint8_t* p, bool be) {
    return be ? static_cast<uint16_t>((p[0] << 8) | p[1]) : static_cast<uint16_t>((p[1] << 8) | p[0]);
}
inline float rd_f32(const uint8_t* p, bool be) {
    const uint32_t v = be ? (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]
                          : (uint32_t(p[3]) << 24) | (uint32_t(p[2]) << 16) | (uint32_t(p[1]) << 8) | p[0];
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}
int comp_bytes(int type) {
    switch (type) {
    case 0:
    case 1: return 1;
    case 2:
    case 3: return 2;
    case 4: return 4;
    default: return 4;
    }
}
// GXCompType: U8=0 S8=1 U16=2 S16=3 F32=4
float rd_comp(const uint8_t* p, int type, int frac, bool be) {
    const float scale = 1.0f / static_cast<float>(1u << (frac & 31));
    switch (type) {
    case 0: return static_cast<float>(p[0]) * scale;
    case 1: return static_cast<float>(static_cast<int8_t>(p[0])) * scale;
    case 2: return static_cast<float>(rd16(p, be)) * scale;
    case 3: return static_cast<float>(static_cast<int16_t>(rd16(p, be))) * scale;
    default: return rd_f32(p, be);
    }
}

// Array data is expected in original (big-endian) byte order; verify against the shape bounds.
bool detect_big_endian(const uint8_t* arr, uint32_t count, uint32_t stride, int type, int frac,
    float extentHint) {
    if (arr == nullptr || count == 0) return true;
    const uint32_t n = std::min<uint32_t>(count, 512);
    auto sample = [&](bool be, int& good, float& maxAbs) {
        good = 0;
        maxAbs = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            bool ok = true;
            for (int c = 0; c < 3; ++c) {
                const float v = rd_comp(arr + i * stride + c * comp_bytes(type), type, frac, be);
                // wrongly swapped floats typically come out as denormals or huge values
                const float a = std::fabs(v);
                if (!std::isfinite(v) || a > 1e6f || (type == 4 && a != 0.0f && a < 1e-9f)) ok = false;
                else maxAbs = std::max(maxAbs, a);
            }
            if (ok) ++good;
        }
    };
    int gb, gl;
    float mb, ml;
    sample(true, gb, mb);
    sample(false, gl, ml);
    if (type == 4) return gb >= gl;
    if (extentHint > 0.0f && mb > 0.0f && ml > 0.0f) {
        return std::fabs(std::log(mb / extentHint)) <= std::fabs(std::log(ml / extentHint));
    }
    return true;
}

struct M34 {
    float m[3][4];
    bool valid = false;
};

template <class P> void copy_mtx(P src, M34& out) {
    if (src == nullptr) return;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) out.m[r][c] = src[r][c];
    out.valid = true;
}

float model_radius(J3DModelData* md) {
    static std::map<uintptr_t, float> cache;
    const uintptr_t k = reinterpret_cast<uintptr_t>(md);
    auto it = cache.find(k);
    if (it != cache.end()) return it->second;
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    const uint16_t n = md->getShapeNum();
    for (uint16_t i = 0; i < n; ++i) {
        J3DShape* s = md->getShapeNodePointer(i);
        if (s == nullptr) continue;
        const Vec* mn = s->getMin();
        const Vec* mx = s->getMax();
        if (!mn || !mx) continue;
        lo[0] = std::min(lo[0], mn->x); lo[1] = std::min(lo[1], mn->y); lo[2] = std::min(lo[2], mn->z);
        hi[0] = std::max(hi[0], mx->x); hi[1] = std::max(hi[1], mx->y); hi[2] = std::max(hi[2], mx->z);
    }
    float r = 0.0f;
    if (lo[0] <= hi[0]) {
        const float dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2];
        r = 0.5f * std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (!std::isfinite(r)) r = 0.0f;
    cache[k] = r;
    return r;
}

// ---------- snapshot ----------
bool snapshot_model(J3DModel* model, MeshData& out, std::string& err) {
    J3DModelData* md = model->getModelData();
    if (md == nullptr) {
        err = "model has no model data";
        return false;
    }
    J3DVertexData& vd = md->getVertexData();
    const uint16_t shapeNum = md->getShapeNum();

    // ---- textures (decode all; shapes refer to them by index) ----
    // ---- textures ----
    J3DTexture* texTable = md->getTexture();
        const uint32_t texCount = texTable != nullptr ? texTable->getNum() : 0;
        for (uint32_t i = 0; i < texCount; ++i) {
            TexImage ti;
            const ResTIMG* res = texTable->getResTIMG(static_cast<uint16_t>(i));
            if (!res) continue;
            const uint32_t w = static_cast<uint32_t>(res->width);
            const uint32_t h = static_cast<uint32_t>(res->height);

            // Включаем чтение текстур обратно:
            const uint8_t fmt = static_cast<uint8_t>(res->format);
            const uint32_t imgOff = static_cast<uint32_t>(res->imageOffset);
            const uint8_t* base = reinterpret_cast<const uint8_t*>(res);
            const uint8_t* pal = nullptr;
            uint32_t palCount = 0;
            uint8_t palFmt = static_cast<uint8_t>(res->colorFormat);
            if (fmt == 8 || fmt == 9 || fmt == 10) {
                palCount = static_cast<uint32_t>(res->numColors);
                pal = base + static_cast<uint32_t>(res->paletteOffset);
            }
            if (imgOff != 0 && decode_gc_texture(base + imgOff, w, h, fmt, pal, palFmt, palCount, ti.rgba)) {
                ti.w = w;
                ti.h = h;
            }

            char d[64];
            std::snprintf(d, sizeof(d), "%ux%u %s", w, h, gc_format_name(fmt));
            ti.desc = d;
            out.textures.push_back(std::move(ti));
        }

    // ---- vertex arrays ----
    const uint8_t* posArr = reinterpret_cast<const uint8_t*>(vd.getVtxPosArray());
    const uint8_t* uvArr = reinterpret_cast<const uint8_t*>(vd.getVtxTexCoordArray(0));
    const uint32_t vtxNum = vd.getVtxNum();
    if (posArr == nullptr || vtxNum == 0) {
        err = "model has no position array";
        return false;
    }
    const int posType = vd.getVtxPosType();
    const int posFrac = vd.getVtxPosFrac();
    const uint32_t posStride = array_stride(vd, 9, 3 * comp_bytes(posType));

    // TEX0 format from the vertex attribute format list
    AttrFmt uvFmt;
    std::vector<AttrFmt> fmts(32);
    if (auto* list = vd.getVtxAttrFmtList()) {
        for (int i = 0; i < 40; ++i) {
            const int a = f_attr(list[i]);
            if (a == 0xFF) break;
            if (a >= 0 && a < 32) {
                fmts[a].found = true;
                fmts[a].cnt = f_cnt(list[i]);
                fmts[a].type = f_type(list[i]);
                fmts[a].frac = f_frac(list[i]);
            }
        }
        logf(0, "fmts: pos found=%d cnt=%d type=%d frac=%d, uv found=%d",
             fmts[9].found, fmts[9].cnt, fmts[9].type, fmts[9].frac, fmts[13].found);
    }
    uvFmt = fmts[13];
    if (!uvFmt.found) {
        uvFmt.found = true;
        uvFmt.cnt = 1;
        uvFmt.type = 4;
    }
    const uint32_t uvStride = array_stride(vd, 13, 2 * comp_bytes(uvFmt.type));

    // endianness guess from the shape bounds
    float extent = 0.0f;
    for (uint16_t i = 0; i < shapeNum; ++i) {
        J3DShape* s = md->getShapeNodePointer(i);
        if (s == nullptr) continue;
        const Vec* mn = s->getMin();
        const Vec* mx = s->getMax();
        if (!mn || !mx) continue;
        for (float v : {mn->x, mn->y, mn->z, mx->x, mx->y, mx->z}) {
            if (std::isfinite(v)) extent = std::max(extent, std::fabs(v));
        }
    }
    // const bool be = detect_big_endian(posArr, vtxNum, posStride, posType, posFrac, extent);
    const bool be = false;
    // const bool be = true;
    if (!be) logf(1, "vertex arrays look little-endian; reading them that way");

    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    std::vector<VertexRef> tris;
    std::string dlErr;
    uint32_t badIndex = 0;

    for (uint16_t si = 0; si < shapeNum; ++si) {
        J3DShape* shape = md->getShapeNodePointer(si);
        if (shape == nullptr) continue;

        // vertex descriptor -> display list layout
        std::vector<AttrDesc> descs;
        if (auto* dlist = shape->getVtxDesc()) {
            for (int i = 0; i < 40; ++i) {
                const int a = d_attr(dlist[i]);
                if (a == 0xFF) break;
                descs.push_back({a, d_type(dlist[i])});
            }
        }
        VertexLayout layout;
        std::string lerr;
        if (!build_vertex_layout(descs, [&](int a) { return (a >= 0 && a < 32) ? fmts[a] : AttrFmt{}; },
                layout, lerr)) {
            logf(1, "shape %u skipped: %s", si, lerr.c_str());
            continue;
        }

        std::string dbg;
        for (const AttrDesc& d : descs) {
            dbg += std::to_string(d.attr) + ":" + std::to_string(d.type) + " ";
        }
        logf(0, "shape %u: stride=%u | posOff=%d posBytes=%d | descs: %s",
             si, layout.stride, layout.posOff, layout.posBytes, dbg.c_str());

        SubMesh sub;
        sub.shapeIndex = si;
        sub.firstVertex = static_cast<uint32_t>(out.vertices.size() / 6);
        sub.texture = -1;
        if (J3DMaterial* mat = shape->getMaterial()) {
            if (const char* nm = mat_name(*mat)) sub.label = nm;
            if (mat->getTevBlock() != nullptr) {
                for (uint32_t k = 0; k < 8; ++k) {
                    const uint16_t t = mat->getTexNo(k);
                    if (t != 0xFFFF && t < texCount) {
                        sub.texture = static_cast<int32_t>(t);
                        break;
                    }
                }
            }
        }
                logf(0, "parsing shape %u...", si);
        const uint16_t groups = shape->getMtxGroupNum();
        for (uint16_t g = 0; g < groups; ++g) {
            J3DShapeDraw* draw = shape->getShapeDraw(g);
            J3DShapeMtx* smtx = shape->getShapeMtx(g);
            if (draw == nullptr) continue;

            // matrices for the (up to 10) position-matrix slots of this group
            M34 slots[10] = {}; // Обнуляем, чтобы valid был false
            if (smtx != nullptr) {
                const uint16_t n = std::min<uint16_t>(smtx->getUseMtxNum(), 10);
                for (uint16_t s = 0; s < n; ++s) {
                    const uint16_t d = smtx->getUseMtxIndex(s);
                    if (d == 0xFFFF || d >= 1024) continue;
                    const uint16_t idx = md->getDrawMtxIndex(d);
                    if (idx >= 1024) continue; // Спасаем игру от чтения за границей
                    if (md->getDrawMtxFlag(d) == 0) copy_mtx(model->getAnmMtx(idx), slots[s]);
                    else copy_mtx(model->getWeightAnmMtx(idx), slots[s]);
                }
            }

            tris.clear();
            std::string perr;
            const uint8_t* dl_ptr = dl_data(draw);
            uint32_t dl_len = dl_size(draw);
            if (dl_len > 1024 * 1024) dl_len = 1024 * 1024; // Защита от чтения гигабайтов
            if (dl_ptr != nullptr && dl_len > 0) {
                logf(0, "DL ptr=%p len=%u bytes: %02X %02X %02X %02X %02X %02X %02X %02X",
                    (void*)dl_ptr, dl_len, dl_ptr[0], dl_ptr[1], dl_ptr[2], dl_ptr[3],
                    dl_ptr[4], dl_ptr[5], dl_ptr[6], dl_ptr[7]);

                if (!parse_display_list(dl_ptr, dl_len, layout, tris, perr) && dlErr.empty()) {
                    dlErr = perr;
                }
            }

            if (tris.size() > 100000) {
                logf(1, "Too many triangles in one draw call (%zu), parsing likely failed!", tris.size());
                tris.resize(100000); // Обрезаем, чтобы не зависнуть
            }
            for (size_t t = 0; t + 2 < tris.size(); t += 3) {
                bool invalid = false;
                for (int k = 0; k < 3; ++k) {
                    if (tris[t + k].posIdx < 0 || static_cast<uint32_t>(tris[t + k].posIdx) >= vtxNum) {
                        invalid = true;
                        break;
                    }
                }
                if (invalid) {
                    ++badIndex;
                    // Создаем вырожденный треугольник (все 3 вершины в 0), чтобы не ломать индексы
                    for (int v_idx = 0; v_idx < 3; ++v_idx) {
                        out.vertices.push_back(0.0f); // x
                        out.vertices.push_back(0.0f); // y
                        out.vertices.push_back(0.0f); // z
                        out.vertices.push_back(0.0f); // u
                        out.vertices.push_back(0.0f); // v
                        out.vertices.push_back(static_cast<float>(si)); // id
                    }
                    continue;
                }

                for (int k = 0; k < 3; ++k) {
                    const VertexRef& vr = tris[t + k];
                    const uint8_t* pp = posArr + static_cast<size_t>(vr.posIdx) * posStride;

                    const float x = rd_comp(pp, posType, posFrac, be);
                    const float y = rd_comp(pp + comp_bytes(posType), posType, posFrac, be);
                    const float z = rd_comp(pp + 2 * comp_bytes(posType), posType, posFrac, be);
                    float wx = x, wy = y, wz = z;
                    const int slot = vr.pnmtx >= 0 ? std::min(vr.pnmtx / 3, 9) : 0;
                    const M34& m = slots[slot];
                    if (m.valid) {
                        wx = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
                        wy = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
                        wz = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
                    }
                    float u = 0.0f, v = 0.0f;
                    if (vr.uvIdx >= 0 && static_cast<uint32_t>(vr.uvIdx) < vtxNum && uvArr != nullptr) {
                        const uint8_t* up = uvArr + static_cast<size_t>(vr.uvIdx) * uvStride;
                        u = rd_comp(up, uvFmt.type, uvFmt.frac, be);
                        v = uvFmt.cnt == 0 ? 0.0f
                                           : rd_comp(up + comp_bytes(uvFmt.type), uvFmt.type, uvFmt.frac, be);
                    }
                    if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz)) {
                        wx = wy = wz = 0.0f;
                    }

                    out.vertices.push_back(wx);
                    out.vertices.push_back(wy);
                    out.vertices.push_back(wz);
                    out.vertices.push_back(u);
                    out.vertices.push_back(v);
                    out.vertices.push_back(static_cast<float>(si));

                    lo[0] = std::min(lo[0], wx); lo[1] = std::min(lo[1], wy); lo[2] = std::min(lo[2], wz);
                    hi[0] = std::max(hi[0], wx); hi[1] = std::max(hi[1], wy); hi[2] = std::max(hi[2], wz);
                }
            }
            if (out.vertices.size() > 6u * 4000000u) {
                err = "model too large";
                return false;
            }
        }
        sub.vertexCount = static_cast<uint32_t>(out.vertices.size() / 6) - sub.firstVertex;
        if (sub.vertexCount > 0) out.subs.push_back(std::move(sub));
    }

    if (!dlErr.empty()) logf(1, "display list problem: %s", dlErr.c_str());
    if (badIndex > 0) logf(1, "%u vertex indices were out of range (wrong array layout?)", badIndex);
    if (out.vertices.empty()) {
        err = dlErr.empty() ? "no triangles found" : dlErr;
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        // Если координаты вышли за разумные пределы, сбрасываем их в 0
        out.bmin[i] = (std::isfinite(lo[i]) && std::abs(lo[i]) < 10000.0f) ? lo[i] : 0.0f;
        out.bmax[i] = (std::isfinite(hi[i]) && std::abs(hi[i]) < 10000.0f) ? hi[i] : 0.0f;
    }
    out.joints = md->getJointNum();
    out.shapes = shapeNum;
    out.materials = md->getMaterialNum();
    out.vertexCount = static_cast<uint32_t>(out.vertices.size() / 6);
    out.triCount = out.vertexCount / 3;
    return true;
}

HookAction on_model_entry(ModContext*, void* args, void*, void*) {
    J3DModel* model = mods::arg<J3DModel*>(args, 0);
    if (model == nullptr) return HOOK_CONTINUE;
    J3DModelData* md = model->getModelData();
    if (md == nullptr) return HOOK_CONTINUE;

    ModelInfo info;
    info.key = reinterpret_cast<uintptr_t>(model);
    info.dataKey = reinterpret_cast<uintptr_t>(md);
    Mtx& base = model->getBaseTRMtx();
    info.pos[0] = base[0][3];
    info.pos[1] = base[1][3];
    info.pos[2] = base[2][3];
    info.radius = model_radius(md);
    info.joints = md->getJointNum();
    info.shapes = md->getShapeNum();
    info.materials = md->getMaterialNum();
    info.verts = md->getVtxNum();
    registry_note(info);

    // logf(0, "hook entry: key=%llx", (unsigned long long)info.key);

    if (take_pending(info.key)) {
        auto mesh = std::make_shared<MeshData>();
        mesh->key = info.key;
        char title[96];
        std::snprintf(title, sizeof(title), "model %06llx (data %06llx)",
            static_cast<unsigned long long>(info.key & 0xFFFFFF),
            static_cast<unsigned long long>(info.dataKey & 0xFFFFFF));
        mesh->title = title;
        std::string err;

        logf(0, "snapshot starting for %llx", (unsigned long long)info.key);

        if (snapshot_model(model, *mesh, err)) {
            logf(0, "snapshot %s: %u tris, %zu shapes, %zu textures", title, mesh->triCount,
                mesh->subs.size(), mesh->textures.size());
            publish_mesh(std::move(mesh));
        } else {
            logf(2, "snapshot of %s failed: %s", title, err.c_str());
        }
    }
    return HOOK_CONTINUE;
}

}  // namespace

ModResult j3d_install_hooks() {
    return mods::hook::add_pre<J3DModelEntryHook>(on_model_entry);
}

}  // namespace mi
