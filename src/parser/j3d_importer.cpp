// Translates live J3D model data into shared snapshots. Game lifecycle and hook registration live
// in game/j3d_hook.cpp; parser algorithms are isolated in the other parser sources.
// Include order matters: the game's global.h redefines nullptr, which breaks standard library
// headers included after it. So: standard headers and our own headers first, game headers after.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "shared/common.hpp"
#include "shared/mesh.hpp"
#include "parser/dlparse.hpp"
#include "parser/gctex.hpp"
#include "parser/j3d_importer.hpp"

#include "global.h"

#include "JSystem/J3DGraphAnimator/J3DModel.h"
#include "JSystem/J3DGraphAnimator/J3DModelData.h"
#include "JSystem/J3DGraphBase/J3DMaterial.h"
#include "JSystem/J3DGraphBase/J3DShape.h"
#include "JSystem/J3DGraphBase/J3DShapeDraw.h"
#include "JSystem/J3DGraphBase/J3DTexture.h"
#include "JSystem/J3DGraphBase/J3DVertex.h"

namespace mi::parser {
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
template <class V> uint32_t array_count(V& vd, int attr, uint32_t fallback) {
    if constexpr (requires { vd.getVtxArrNum(static_cast<GXAttr>(attr)); }) {
        return vd.getVtxArrNum(static_cast<GXAttr>(attr));
    } else {
        return fallback;
    }
}
template <class D> const uint8_t* dl_data(D* d) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(d->getDisplayList());
    return ptr; // Возвращаем как есть. Если там мусор, мы просто пропустим шейп, но не вылетим.
}
template <class D> uint32_t dl_size(D* d) { return static_cast<uint32_t>(d->getDisplayListSize()); }
template <class S> bool original_has_pnmtx(S* shape) {
    if constexpr (requires { shape->hasPNMTXIdx(); }) return shape->hasPNMTXIdx();
    else return false;
}

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

template <class P> bool inverse_affine(P src, M34& out) {
    const float a = src[0][0], b = src[0][1], c = src[0][2];
    const float d = src[1][0], e = src[1][1], f = src[1][2];
    const float g = src[2][0], h = src[2][1], i = src[2][2];
    const float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (!std::isfinite(det) || std::abs(det) < 1e-8f) return false;

    const float invDet = 1.0f / det;
    out.m[0][0] = (e * i - f * h) * invDet;
    out.m[0][1] = (c * h - b * i) * invDet;
    out.m[0][2] = (b * f - c * e) * invDet;
    out.m[1][0] = (f * g - d * i) * invDet;
    out.m[1][1] = (a * i - c * g) * invDet;
    out.m[1][2] = (c * d - a * f) * invDet;
    out.m[2][0] = (d * h - e * g) * invDet;
    out.m[2][1] = (b * g - a * h) * invDet;
    out.m[2][2] = (a * e - b * d) * invDet;
    for (int r = 0; r < 3; ++r) {
        out.m[r][3] = 0.0f;
        for (int k = 0; k < 3; ++k) out.m[r][3] -= out.m[r][k] * src[k][3];
    }
    out.valid = true;
    return true;
}

M34 multiply_affine(const M34& a, const M34& b) {
    M34 out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            out.m[r][c] = c == 3 ? a.m[r][3] : 0.0f;
            for (int k = 0; k < 3; ++k) out.m[r][c] += a.m[r][k] * b.m[k][c];
        }
    }
    out.valid = a.valid && b.valid;
    return out;
}

void transform_point(const M34& m, float x, float y, float z, float& ox, float& oy, float& oz) {
    ox = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
    oy = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
    oz = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
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
template <typename... Args>
void snapshot_log(bool verbose, int level, const char* format, Args... args) {
    if (verbose) logf(level, format, args...);
}

bool snapshot_model(J3DModel* model, MeshData& out, std::string& err,
    const MeshData* previous, bool verbose) {
    J3DModelData* md = model->getModelData();
    if (md == nullptr) {
        err = "model has no model data";
        return false;
    }
    J3DVertexData& vd = md->getVertexData();
    const uint16_t shapeNum = md->getShapeNum();
    const uint8_t* sourcePosArr = reinterpret_cast<const uint8_t*>(vd.getVtxPosArray());
    void* currentPos = model->getVertexBuffer()->getCurrentVtxPos();
    const bool cpuSkinPos = model->checkFlag(J3DMdlFlag_SkinPosCpu);
    snapshot_log(verbose, 0,
        "snapshot begin: model=%p data=%p shapes=%u joints=%u materials=%u verts=%u "
        "fastSkin=%u skinDeform=%p cpuSkinPos=%u sourcePos=%p currentPos=%p",
        static_cast<void*>(model), static_cast<void*>(md), static_cast<unsigned>(shapeNum),
        static_cast<unsigned>(md->getJointNum()), static_cast<unsigned>(md->getMaterialNum()),
        vd.getVtxNum(), md->checkFlag(0x100) ? 1u : 0u,
        static_cast<void*>(model->getSkinDeform()), cpuSkinPos ? 1u : 0u,
        static_cast<const void*>(sourcePosArr), currentPos);
    M34 modelBaseInverse{};
    const bool hasModelBaseInverse = inverse_affine(model->getBaseTRMtx(), modelBaseInverse);
    Mtx& modelBase = model->getBaseTRMtx();
    snapshot_log(verbose, 0, "model base: invertible=%u translation=(%.3f,%.3f,%.3f)",
        hasModelBaseInverse ? 1u : 0u, modelBase[0][3], modelBase[1][3], modelBase[2][3]);

    J3DTexture* texTable = md->getTexture();
    const uint32_t texCount = texTable != nullptr ? texTable->getNum() : 0;
    if (previous != nullptr && previous->textures != nullptr &&
        previous->textures->size() == texCount) {
        out.textures = previous->textures;
    } else {
        auto textureImages = std::make_shared<std::vector<TexImage>>();
        for (uint32_t i = 0; i < texCount; ++i) {
            TexImage ti;
            const ResTIMG* res = texTable->getResTIMG(static_cast<uint16_t>(i));
            if (!res) continue;
            const uint32_t w = static_cast<uint32_t>(res->width);
            const uint32_t h = static_cast<uint32_t>(res->height);

            // Включаем чтение текстур обратно:
            const uint8_t fmt = static_cast<uint8_t>(res->format);
            const int32_t imgOff = static_cast<int32_t>(res->imageOffset);
            const int32_t palOff = static_cast<int32_t>(res->paletteOffset);
            const uint8_t* base = reinterpret_cast<const uint8_t*>(res);
            const uint8_t* image = texTable->getImgDataPtr(static_cast<uint16_t>(i));
            const bool runtimeImagePointer = image != nullptr;
            const uint8_t* dataBase = runtimeImagePointer
                ? image - imgOff
                : base;
            if (image == nullptr) {
                image = base + (imgOff == 0 ? sizeof(*res) : imgOff);
            }
            const uint8_t* pal = nullptr;
            uint32_t palCount = 0;
            uint8_t palFmt = static_cast<uint8_t>(res->colorFormat);
            if (fmt == 8 || fmt == 9 || fmt == 10) {
                palCount = static_cast<uint32_t>(res->numColors);
                pal = dataBase + palOff;
            }
            const bool decoded = decode_gc_texture(image, w, h, fmt, pal, palFmt, palCount, ti.rgba);
            if (decoded) {
                ti.w = w;
                ti.h = h;
            } else {
                snapshot_log(verbose, 1, "texture %u not decoded: format=%u size=%ux%u offsets image=%d palette=%d",
                    i, fmt, w, h, imgOff, palOff);
            }
            snapshot_log(verbose, 0,
                "texture %u import: decoded=%u format=%u size=%ux%u imageOffset=%d paletteOffset=%d "
                "paletteFmt=%u colors=%u rgbaBytes=%zu runtimeImage=%u dataBase=%p image=%p palette=%p",
                i, decoded ? 1u : 0u, static_cast<unsigned>(fmt), w, h, imgOff, palOff,
                static_cast<unsigned>(palFmt), palCount,
                ti.rgba.size(), runtimeImagePointer ? 1u : 0u, static_cast<const void*>(dataBase),
                static_cast<const void*>(image), static_cast<const void*>(pal));

            char d[64];
            std::snprintf(d, sizeof(d), "%ux%u %s", w, h, gc_format_name(fmt));
            ti.desc = d;
            textureImages->push_back(std::move(ti));
        }
        out.textures = std::move(textureImages);
    }

    // ---- vertex arrays ----
    const uint8_t* posArr = sourcePosArr;
    bool positionsDeformed = false;
    if (currentPos != nullptr) {
        if (currentPos != sourcePosArr) {
            posArr = reinterpret_cast<const uint8_t*>(currentPos);
            positionsDeformed = cpuSkinPos;
        }
    }
    const uint32_t vtxNum = vd.getVtxNum();
    const uint32_t posCount = array_count(vd, 9, vtxNum);
    if (posArr == nullptr || vtxNum == 0 || posCount == 0) {
        err = "model has no position array";
        return false;
    }
    const int posType = vd.getVtxPosType();
    const int posFrac = vd.getVtxPosFrac();
    const uint32_t posStride = array_stride(vd, 9, 3 * comp_bytes(posType));

    // TEXn formats from the vertex attribute format list
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
        snapshot_log(verbose, 0, "fmts: pos found=%d cnt=%d type=%d frac=%d, uv found=%d",
             fmts[9].found, fmts[9].cnt, fmts[9].type, fmts[9].frac, fmts[13].found);
    }
    const uint8_t* uvArr[8] = {};
    uint32_t uvStrides[8] = {};
    uint32_t uvCounts[8] = {};
    AttrFmt uvFmts[8] = {};
    for (int i = 0; i < 8; ++i) {
        uvArr[i] = reinterpret_cast<const uint8_t*>(vd.getVtxTexCoordArray(static_cast<uint8_t>(i)));
        uvFmts[i] = fmts[13 + i];
        if (!uvFmts[i].found) {
            uvFmts[i].found = true;
            uvFmts[i].cnt = 1;
            uvFmts[i].type = 4;
        }
        uvStrides[i] = array_stride(vd, 13 + i, 2 * comp_bytes(uvFmts[i].type));
        uvCounts[i] = uvArr[i] != nullptr ? array_count(vd, 13 + i, vtxNum) : 0;
    }
    snapshot_log(verbose, 0,
        "vertex arrays: posType=%d frac=%d count=%u stride=%u current=%p usingCurrent=%u cpuDeformed=%u",
        posType, posFrac, posCount, posStride, static_cast<const void*>(posArr),
        posArr != sourcePosArr ? 1u : 0u, positionsDeformed ? 1u : 0u);
    for (int i = 0; i < 8; ++i) {
        if (uvArr[i] != nullptr) {
            snapshot_log(verbose, 0, "TEX%u array=%p count=%u stride=%u type=%d frac=%d components=%d",
                static_cast<unsigned>(i), static_cast<const void*>(uvArr[i]), uvCounts[i],
                uvStrides[i], uvFmts[i].type, uvFmts[i].frac, uvFmts[i].cnt);
        }
    }
    // Dusklight swaps J3D vertex-array components to host byte order when loading VTX1.
    constexpr bool arrayBigEndian = false;
    const auto indicesValid = [&](const std::vector<VertexRef>& refs) {
        return std::all_of(refs.begin(), refs.end(), [&](const VertexRef& ref) {
            if (ref.posIdx < 0 || static_cast<uint32_t>(ref.posIdx) >= posCount) return false;
            for (size_t i = 0; i < ref.uvIdx.size(); ++i) {
                if (ref.uvIdx[i] >= 0 &&
                    (uvArr[i] == nullptr ||
                     static_cast<uint32_t>(ref.uvIdx[i]) >= uvCounts[i])) return false;
            }
            return true;
        });
    };

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
        const bool hasPnmtxDesc = std::any_of(descs.begin(), descs.end(),
            [](const AttrDesc& d) { return d.attr == 0 && d.type != 0; });
        const bool originalPnm = original_has_pnmtx(shape);
        VertexLayout layout;
        std::string lerr;
        if (!build_vertex_layout(descs, [&](int a) { return (a >= 0 && a < 32) ? fmts[a] : AttrFmt{}; },
                layout, lerr)) {
            snapshot_log(verbose, 1, "shape %u skipped: %s", si, lerr.c_str());
            continue;
        }

        std::string dbg;
        for (const AttrDesc& d : descs) {
            dbg += std::to_string(d.attr) + ":" + std::to_string(d.type) + " ";
        }
        snapshot_log(verbose, 0, "shape %u: stride=%u | posOff=%d posBytes=%d | pnmDesc=%u originalPnm=%u | descs: %s",
             static_cast<unsigned>(si), layout.stride, layout.posOff, layout.posBytes,
             hasPnmtxDesc ? 1u : 0u, originalPnm ? 1u : 0u, dbg.c_str());

        SubMesh sub;
        sub.shapeIndex = si;
        sub.firstVertex = static_cast<uint32_t>(out.vertices.size() / 6);
        sub.texture = -1;
        int uvSource = 0;
        int materialIndex = -1;
        int selectedTevStage = -1;
        int selectedTexMap = -1;
        int selectedTexCoord = -1;
        int selectedTexSource = -1;
        int selectedTexMtxReg = -1;
        float uvMatrix[2][4] = {{1.0f, 0.0f, 0.0f, 0.0f},
                                {0.0f, 1.0f, 0.0f, 0.0f}};
        bool applyUvMatrix = false;
        if (J3DMaterial* mat = shape->getMaterial()) {
            materialIndex = mat->getIndex();
            if (const char* nm = mat_name(*mat)) sub.label = nm;
            if (mat->getTevBlock() != nullptr) {
                int firstActiveTexture = -1;
                int firstActiveStage = -1;
                int firstActiveMap = -1;
                for (uint32_t stage = 0; stage < mat->getTevStageNum(); ++stage) {
                    J3DTevOrder* order = mat->getTevBlock()->getTevOrder(stage);
                    if (order == nullptr || order->getTexMap() >= 8) continue;

                    const uint16_t t = mat->getTexNo(order->getTexMap());
                    if (t == 0xFFFF || t >= texCount) continue;
                    if (firstActiveTexture < 0) {
                        firstActiveTexture = static_cast<int>(t);
                        firstActiveStage = static_cast<int>(stage);
                        firstActiveMap = order->getTexMap();
                    }

                    const uint8_t coord = order->mTexCoord;
                    if (coord >= mat->getTexGenNum()) continue;
                    J3DTexCoord* texCoord = mat->getTexCoord(coord);
                    if (texCoord == nullptr) continue;
                    const uint8_t source = texCoord->getTexGenSrc();
                    if (source < GX_TG_TEX0 || source > GX_TG_TEX7) continue;
                    const int sourceIndex = source - GX_TG_TEX0;
                    if (uvArr[sourceIndex] == nullptr) {
                        snapshot_log(verbose, 1, "shape %u material texture uses missing TEX%u array",
                            si, static_cast<unsigned>(sourceIndex));
                        continue;
                    }

                    sub.texture = static_cast<int32_t>(t);
                    uvSource = sourceIndex;
                    selectedTevStage = static_cast<int>(stage);
                    selectedTexMap = order->getTexMap();
                    selectedTexCoord = coord;
                    selectedTexSource = source;
                    const uint8_t texGenType = texCoord->getTexGenType();
                    const uint32_t texMtxReg = texCoord->getTexMtxReg();
                    selectedTexMtxReg = static_cast<int>(texMtxReg);
                    if (texGenType == GX_MTX2x4 && texMtxReg >= GX_TEXMTX0 &&
                        texMtxReg <= GX_TEXMTX9 && (texMtxReg - GX_TEXMTX0) % 3 == 0) {
                        if (J3DTexMtx* texMtx = mat->getTexMtx((texMtxReg - GX_TEXMTX0) / 3)) {
                            Mtx& matrix = texMtx->getMtx();
                            bool finite = true;
                            for (int row = 0; row < 2; ++row) {
                                for (int col = 0; col < 4; ++col) {
                                    uvMatrix[row][col] = matrix[row][col];
                                    finite = finite && std::isfinite(uvMatrix[row][col]);
                                }
                            }
                            applyUvMatrix = finite;
                            if (!finite) {
                                snapshot_log(verbose, 1, "shape %u has a non-finite texture matrix; using raw UVs", si);
                            }
                        }
                    }
                    break;
                }
                if (sub.texture < 0) {
                    sub.texture = firstActiveTexture;
                    selectedTevStage = firstActiveStage;
                    selectedTexMap = firstActiveMap;
                }
            }
        }
        snapshot_log(verbose, 0,
            "shape %u material map: material=%d label=%s texture=%d tevStage=%d texMap=%d "
            "texCoord=%d source=%d uvArray=%p texMtxReg=%d matrixApplied=%u",
            static_cast<unsigned>(si), materialIndex, sub.label.empty() ? "-" : sub.label.c_str(),
            sub.texture, selectedTevStage, selectedTexMap, selectedTexCoord, selectedTexSource,
            static_cast<const void*>(uvArr[uvSource]), selectedTexMtxReg, applyUvMatrix ? 1u : 0u);
        if (applyUvMatrix) {
            snapshot_log(verbose, 0, "shape %u UV matrix: [%g %g %g %g] [%g %g %g %g]",
                static_cast<unsigned>(si), uvMatrix[0][0], uvMatrix[0][1], uvMatrix[0][2],
                uvMatrix[0][3], uvMatrix[1][0], uvMatrix[1][1], uvMatrix[1][2], uvMatrix[1][3]);
        }
        snapshot_log(verbose, 0, "parsing shape %u...", si);
        const uint16_t groups = shape->getMtxGroupNum();
        M34 slots[10] = {};
        for (uint16_t g = 0; g < groups; ++g) {
            J3DShapeDraw* draw = shape->getShapeDraw(g);
            J3DShapeMtx* smtx = shape->getShapeMtx(g);
            if (draw == nullptr) continue;

            // J3D uses 0xFFFF entries to retain the slot value loaded by an earlier group.
            uint32_t inheritedMatrixSlots = 0;
            uint32_t unresolvedMatrixSlots = 0;
            if (smtx != nullptr) {
                const uint16_t n = std::min<uint16_t>(smtx->getUseMtxNum(), 10);
                for (uint16_t s = 0; s < n; ++s) {
                    const uint16_t d = smtx->getUseMtxIndex(s);
                    if (d == 0xFFFF) {
                        if (slots[s].valid) ++inheritedMatrixSlots;
                        else ++unresolvedMatrixSlots;
                        continue;
                    }
                    if (d >= 1024) {
                        slots[s].valid = false;
                        ++unresolvedMatrixSlots;
                        continue;
                    }
                    const uint16_t idx = md->getDrawMtxIndex(d);
                    if (idx >= 1024) {
                        slots[s].valid = false;
                        ++unresolvedMatrixSlots;
                        continue;
                    }
                    M34 worldMatrix{};
                    if (md->getDrawMtxFlag(d) == 0) copy_mtx(model->getAnmMtx(idx), worldMatrix);
                    else copy_mtx(model->getWeightAnmMtx(idx), worldMatrix);
                    if (worldMatrix.valid && hasModelBaseInverse) {
                        slots[s] = multiply_affine(modelBaseInverse, worldMatrix);
                    } else if (worldMatrix.valid) {
                        worldMatrix.valid = false;
                        slots[s] = worldMatrix;
                    }
                    snapshot_log(verbose, 0,
                        "shape %u group %u matrix slot %u: draw=%u matrix=%u weighted=%u valid=%u "
                        "worldTranslation=(%.3f,%.3f,%.3f) localTranslation=(%.3f,%.3f,%.3f)",
                        static_cast<unsigned>(si), static_cast<unsigned>(g), static_cast<unsigned>(s),
                        static_cast<unsigned>(d), static_cast<unsigned>(idx),
                        md->getDrawMtxFlag(d) ? 1u : 0u, worldMatrix.valid ? 1u : 0u,
                        worldMatrix.m[0][3], worldMatrix.m[1][3], worldMatrix.m[2][3],
                        slots[s].m[0][3], slots[s].m[1][3], slots[s].m[2][3]);
                }
            }
            snapshot_log(verbose, 0, "shape %u group %u matrix state: inherited=%u unresolved=%u",
                static_cast<unsigned>(si), static_cast<unsigned>(g),
                inheritedMatrixSlots, unresolvedMatrixSlots);

            tris.clear();
            std::string perr;
            bool parsed = false;
            bool fastSkinFallbackCandidate = false;
            bool fastSkinFallbackUsed = false;
            const uint8_t* dl_ptr = dl_data(draw);
            uint32_t dl_len = dl_size(draw);
            if (dl_len > 1024 * 1024) dl_len = 1024 * 1024; // Защита от чтения гигабайтов
            snapshot_log(verbose, 0, "shape %u group %u draw: ptr=%p len=%u groups=%u usedMatrices=%u cpuSkinPos=%u",
                static_cast<unsigned>(si), static_cast<unsigned>(g), static_cast<const void*>(dl_ptr),
                dl_len, static_cast<unsigned>(groups),
                smtx != nullptr ? static_cast<unsigned>(smtx->getUseMtxNum()) : 0u,
                positionsDeformed ? 1u : 0u);
            if (dl_ptr != nullptr && dl_len > 0) {
                if (dl_len >= 8) {
                    snapshot_log(verbose, 0, "DL prefix: %02X %02X %02X %02X %02X %02X %02X %02X",
                        dl_ptr[0], dl_ptr[1], dl_ptr[2], dl_ptr[3],
                        dl_ptr[4], dl_ptr[5], dl_ptr[6], dl_ptr[7]);
                } else {
                    snapshot_log(verbose, 0, "DL prefix: only %u byte(s) available", dl_len);
                }

                VertexLayout drawLayout = layout;
                parsed = parse_display_list(dl_ptr, dl_len, drawLayout, tris, perr);
                fastSkinFallbackCandidate =
                    md->checkFlag(0x100) && model->getSkinDeform() != nullptr &&
                    !hasPnmtxDesc && dl_ptr[0] == 0x50;
                if (fastSkinFallbackCandidate) {
                    std::vector<AttrDesc> drawDescs = descs;
                    drawDescs.push_back({0, 1});
                    std::string drawLayoutErr;
                    VertexLayout fallbackLayout;
                    if (build_vertex_layout(drawDescs,
                            [&](int a) { return (a >= 0 && a < 32) ? fmts[a] : AttrFmt{}; },
                            fallbackLayout, drawLayoutErr)) {
                        std::vector<VertexRef> fallbackTris;
                        std::string fallbackErr;
                        const bool fallbackParsed =
                            parse_display_list(dl_ptr, dl_len, fallbackLayout, fallbackTris, fallbackErr);
                        const uint16_t matrixSlots = smtx != nullptr
                            ? std::min<uint16_t>(smtx->getUseMtxNum(), 10)
                            : 0;
                        const bool fallbackPnmValid = fallbackParsed && !fallbackTris.empty() &&
                            matrixSlots != 0 &&
                            std::all_of(fallbackTris.begin(), fallbackTris.end(), [&](const VertexRef& ref) {
                                return ref.pnmtx >= 0 && ref.pnmtx % 3 == 0 &&
                                    static_cast<uint16_t>(ref.pnmtx / 3) < matrixSlots;
                            });
                        if (fallbackPnmValid && indicesValid(fallbackTris)) {
                            drawLayout = fallbackLayout;
                            tris = std::move(fallbackTris);
                            parsed = true;
                            fastSkinFallbackUsed = true;
                            perr.clear();
                            snapshot_log(verbose, 1, "shape %u: restored PNMTXIDX byte for fast-skin display list", si);
                        } else {
                            snapshot_log(verbose, 1, "shape %u: fast-skin layout fallback rejected: pnmValid=%u matrixSlots=%u error=%s",
                                si, fallbackPnmValid ? 1u : 0u, static_cast<unsigned>(matrixSlots),
                                fallbackErr.c_str());
                        }
                    } else {
                        snapshot_log(verbose, 1, "shape %u: fast-skin layout fallback failed: %s",
                            si, drawLayoutErr.c_str());
                    }
                }
            }
            uint32_t noPnmtx = 0;
            uint32_t invalidPosRefs = 0;
            uint32_t missingUvRefs = 0;
            uint32_t invalidUvRefs = 0;
            int minPos = std::numeric_limits<int>::max();
            int maxPos = -1;
            int minUv = std::numeric_limits<int>::max();
            int maxUv = -1;
            uint32_t pnmtxCounts[10] = {};
            uint32_t pnmtxUnresolved = 0;
            for (const VertexRef& ref : tris) {
                minPos = std::min(minPos, ref.posIdx);
                maxPos = std::max(maxPos, ref.posIdx);
                if (ref.posIdx < 0 || static_cast<uint32_t>(ref.posIdx) >= posCount) ++invalidPosRefs;
                if (ref.pnmtx < 0) {
                    ++noPnmtx;
                } else {
                    if (ref.pnmtx % 3 != 0 || ref.pnmtx / 3 >= 10) {
                        ++pnmtxUnresolved;
                    } else {
                        const int slot = ref.pnmtx / 3;
                        ++pnmtxCounts[slot];
                        if (!slots[slot].valid) ++pnmtxUnresolved;
                    }
                }
                const int uvIdx = ref.uvIdx[uvSource];
                if (uvIdx < 0) {
                    ++missingUvRefs;
                } else {
                    minUv = std::min(minUv, uvIdx);
                    maxUv = std::max(maxUv, uvIdx);
                    if (uvArr[uvSource] == nullptr ||
                        static_cast<uint32_t>(uvIdx) >= uvCounts[uvSource]) ++invalidUvRefs;
                }
            }
            std::string pnmtxSummary;
            for (int slot = 0; slot < 10; ++slot) {
                if (pnmtxCounts[slot] != 0) {
                    pnmtxSummary += std::to_string(slot) + ":" + std::to_string(pnmtxCounts[slot]) + " ";
                }
            }
            snapshot_log(verbose, 0,
                "shape %u group %u parse: ok=%u triangles=%zu fastSkinCandidate=%u fallbackUsed=%u "
                "posRefsInvalid=%u posRange=%d..%d uvRefsMissing=%u uvRefsInvalid=%u uvRange=%d..%d "
                "pnmtxMissing=%u pnmtxUnresolved=%u pnmtxSlots=[%s] error=%s",
                static_cast<unsigned>(si), static_cast<unsigned>(g), parsed ? 1u : 0u, tris.size() / 3,
                fastSkinFallbackCandidate ? 1u : 0u, fastSkinFallbackUsed ? 1u : 0u,
                invalidPosRefs, minPos == std::numeric_limits<int>::max() ? -1 : minPos, maxPos,
                missingUvRefs, invalidUvRefs, minUv == std::numeric_limits<int>::max() ? -1 : minUv,
                maxUv, noPnmtx, pnmtxUnresolved, pnmtxSummary.c_str(), perr.empty() ? "-" : perr.c_str());
            if (!parsed && dlErr.empty()) {
                dlErr = perr.empty() ? "display list could not be parsed" : perr;
            }

            if (tris.size() > 100000) {
                snapshot_log(verbose, 1, "Too many triangles in one draw call (%zu), parsing likely failed!", tris.size());
                tris.resize(100000); // Обрезаем, чтобы не зависнуть
            }
            for (size_t t = 0; t + 2 < tris.size(); t += 3) {
                bool invalid = false;
                for (int k = 0; k < 3; ++k) {
                    if (tris[t + k].posIdx < 0 || static_cast<uint32_t>(tris[t + k].posIdx) >= posCount) {
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

                    const float x = rd_comp(pp, posType, posFrac, arrayBigEndian);
                    const float y = rd_comp(pp + comp_bytes(posType), posType, posFrac, arrayBigEndian);
                    const float z = rd_comp(pp + 2 * comp_bytes(posType), posType, posFrac, arrayBigEndian);
                    float wx = x, wy = y, wz = z;
                    const int slot = vr.pnmtx < 0
                        ? 0
                        : (vr.pnmtx % 3 == 0 ? vr.pnmtx / 3 : -1);
                    const M34* m = slot >= 0 && slot < 10 && slots[slot].valid
                        ? &slots[slot]
                        : nullptr;
                    if (!positionsDeformed && m != nullptr) {
                        transform_point(*m, x, y, z, wx, wy, wz);
                    } else if (positionsDeformed && hasModelBaseInverse) {
                        transform_point(modelBaseInverse, x, y, z, wx, wy, wz);
                    }
                    float u = 0.0f, v = 0.0f;
                    const int uvIdx = vr.uvIdx[uvSource];
                    if (uvIdx >= 0 && static_cast<uint32_t>(uvIdx) < uvCounts[uvSource] &&
                        uvArr[uvSource] != nullptr) {
                        const uint8_t* up = uvArr[uvSource] +
                            static_cast<size_t>(uvIdx) * uvStrides[uvSource];
                        u = rd_comp(up, uvFmts[uvSource].type, uvFmts[uvSource].frac, arrayBigEndian);
                        v = uvFmts[uvSource].cnt == 0 ? 0.0f
                            : rd_comp(up + comp_bytes(uvFmts[uvSource].type),
                                      uvFmts[uvSource].type, uvFmts[uvSource].frac, arrayBigEndian);
                        if (applyUvMatrix) {
                            const float transformedU =
                                uvMatrix[0][0] * u + uvMatrix[0][1] * v + uvMatrix[0][3];
                            const float transformedV =
                                uvMatrix[1][0] * u + uvMatrix[1][1] * v + uvMatrix[1][3];
                            u = transformedU;
                            v = transformedV;
                        }
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

    if (!dlErr.empty()) snapshot_log(verbose, 1, "display list problem: %s", dlErr.c_str());
    if (badIndex > 0) snapshot_log(verbose, 1, "%u vertex indices were out of range (wrong array layout?)", badIndex);
    if (out.vertices.empty()) {
        err = dlErr.empty() ? "no triangles found" : dlErr;
        snapshot_log(verbose, 1, "snapshot failed: model=%p error=%s", static_cast<void*>(model), err.c_str());
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        out.bmin[i] = std::isfinite(lo[i]) ? lo[i] : 0.0f;
        out.bmax[i] = std::isfinite(hi[i]) ? hi[i] : 0.0f;
    }
    out.joints = md->getJointNum();
    out.shapes = shapeNum;
    out.materials = md->getMaterialNum();
    out.vertexCount = static_cast<uint32_t>(out.vertices.size() / 6);
    out.triCount = out.vertexCount / 3;
    uint32_t decodedTextures = 0;
    for (const TexImage& texture : *out.textures) {
        if (texture.w != 0 && texture.h != 0 && !texture.rgba.empty()) ++decodedTextures;
    }
    snapshot_log(verbose, 0,
        "snapshot complete: model=%p vertices=%u triangles=%u submeshes=%zu textures=%zu "
        "decodedTextures=%u badTriangles=%u bounds=[(%g,%g,%g)-(%g,%g,%g)]",
        static_cast<void*>(model), out.vertexCount, out.triCount, out.subs.size(),
        out.textures->size(), decodedTextures, badIndex,
        out.bmin[0], out.bmin[1], out.bmin[2], out.bmax[0], out.bmax[1], out.bmax[2]);
    return true;
}

}  // namespace

ModelInfo describe_j3d_model(::J3DModel* model, bool includePoseSignature) {
    ModelInfo info{};
    if (model == nullptr) return info;
    J3DModelData* md = model->getModelData();
    if (md == nullptr) return info;
    info.key = reinterpret_cast<uintptr_t>(model);
    info.dataKey = reinterpret_cast<uintptr_t>(md);
    Mtx& base = model->getBaseTRMtx();
    info.pos[0] = base[0][3];
    info.pos[1] = base[1][3];
    info.pos[2] = base[2][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            info.transform[row * 4 + column] = base[row][column];
        }
    }
    info.radius = model_radius(md);
    info.joints = md->getJointNum();
    info.shapes = md->getShapeNum();
    info.materials = md->getMaterialNum();
    info.verts = md->getVtxNum();

    J3DVertexBuffer* vertexBuffer = model->getVertexBuffer();
    const void* sourcePositions = md->getVtxPosArray();
    const bool cpuSkinPos = model->checkFlag(J3DMdlFlag_SkinPosCpu);
    M34 baseInverse{};
    if (includePoseSignature && sourcePositions != nullptr && vertexBuffer != nullptr &&
        vertexBuffer->getCurrentVtxPos() == sourcePositions && !cpuSkinPos &&
        model->getSkinDeform() == nullptr &&
        inverse_affine(model->getBaseTRMtx(), baseInverse)) {
        uint64_t poseKey = 14695981039346656037ull;
        const auto hashValue = [&poseKey](uint32_t value) {
            poseKey ^= value;
            poseKey *= 1099511628211ull;
        };
        hashValue(static_cast<uint32_t>(md->getDrawMtxNum()));
        bool validPose = md->getDrawMtxNum() <= 1024;
        for (uint16_t draw = 0; validPose && draw < md->getDrawMtxNum(); ++draw) {
            const uint16_t index = md->getDrawMtxIndex(draw);
            const uint8_t flag = md->getDrawMtxFlag(draw);
            if (index >= 1024) {
                validPose = false;
                break;
            }
            M34 worldMatrix{};
            if (flag == 0) copy_mtx(model->getAnmMtx(index), worldMatrix);
            else copy_mtx(model->getWeightAnmMtx(index), worldMatrix);
            if (!worldMatrix.valid) {
                validPose = false;
                break;
            }
            const M34 localMatrix = multiply_affine(baseInverse, worldMatrix);
            hashValue(index);
            hashValue(flag);
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 4; ++column) {
                    const float component = localMatrix.m[row][column];
                    if (!std::isfinite(component)) {
                        validPose = false;
                        break;
                    }
                    uint32_t bits = 0;
                    std::memcpy(&bits, &component, sizeof(bits));
                    hashValue(bits);
                }
                if (!validPose) break;
            }
        }
        if (validPose) {
            info.poseKey = poseKey;
            info.geometryCacheable = true;
        }
    }
    return info;
}

bool snapshot_j3d_model(::J3DModel* model, MeshData& out, std::string& err,
    const MeshData* previous, bool verbose) {
    return snapshot_model(model, out, err, previous, verbose);
}

}  // namespace mi::parser
