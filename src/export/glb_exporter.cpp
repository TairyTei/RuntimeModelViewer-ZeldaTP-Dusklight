#include "export/glb_exporter.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <iterator>
#include <limits>
#include <system_error>

namespace mi::gltf {
namespace {

struct BufferView {
    uint32_t offset = 0;
    uint32_t length = 0;
    uint32_t stride = 0;
    uint32_t target = 0;
};

struct Accessor {
    uint32_t bufferView = 0;
    uint32_t byteOffset = 0;
    uint32_t count = 0;
    const char* type = nullptr;
    bool hasBounds = false;
    float min[3] = {};
    float max[3] = {};
};

struct Primitive {
    uint32_t positionAccessor = 0;
    uint32_t uvAccessor = 0;
    int32_t textureIndex = -1;
    int32_t material = -1;
};

struct Image {
    uint32_t bufferView = 0;
};

void append_u32_le(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value >> 16));
    out.push_back(static_cast<uint8_t>(value >> 24));
}

void append_u32_be(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value >> 24));
    out.push_back(static_cast<uint8_t>(value >> 16));
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value));
}

void append_float_le(std::vector<uint8_t>& out, float value) {
    append_u32_le(out, std::bit_cast<uint32_t>(value));
}

bool append_json_float(std::string& out, float value) {
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
        std::chars_format::general, std::numeric_limits<float>::max_digits10);
    if (result.ec != std::errc{}) return false;
    out.append(buffer, result.ptr);
    return true;
}

template <typename Integer>
void append_json_integer(std::string& out, Integer value) {
    char buffer[32];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    out.append(buffer, result.ptr);
}

void align4(std::vector<uint8_t>& data) {
    while ((data.size() & 3u) != 0) data.push_back(0);
}

uint32_t crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

void append_png_chunk(std::vector<uint8_t>& png, const char type[4],
    const uint8_t* data, size_t size) {
    append_u32_be(png, static_cast<uint32_t>(size));
    const size_t crcStart = png.size();
    for (int i = 0; i < 4; ++i) png.push_back(static_cast<uint8_t>(type[i]));
    if (size != 0) png.insert(png.end(), data, data + size);
    append_u32_be(png, crc32(png.data() + crcStart, png.size() - crcStart));
}

bool encode_png(const TexImage& texture, std::vector<uint8_t>& png) {
    if (texture.w == 0 || texture.h == 0) return false;
    if (texture.w > std::numeric_limits<size_t>::max() / 4u) return false;
    const size_t rowBytes = static_cast<size_t>(texture.w) * 4u;
    if (static_cast<size_t>(texture.h) >
        std::numeric_limits<size_t>::max() / rowBytes) return false;
    if (rowBytes * texture.h != texture.rgba.size()) return false;
    if (static_cast<size_t>(texture.h) >
        std::numeric_limits<size_t>::max() / (rowBytes + 1u)) {
        return false;
    }

    std::vector<uint8_t> raw;
    raw.reserve((rowBytes + 1u) * texture.h);
    for (uint32_t y = 0; y < texture.h; ++y) {
        raw.push_back(0);
        const size_t offset = static_cast<size_t>(y) * rowBytes;
        raw.insert(raw.end(), texture.rgba.begin() + offset,
            texture.rgba.begin() + offset + rowBytes);
    }

    std::vector<uint8_t> deflate;
    deflate.reserve(raw.size() + raw.size() / 65535u * 5u + 16u);
    deflate.push_back(0x78);
    deflate.push_back(0x01);
    size_t cursor = 0;
    while (cursor < raw.size()) {
        const size_t blockSize = std::min<size_t>(65535u, raw.size() - cursor);
        const bool finalBlock = cursor + blockSize == raw.size();
        deflate.push_back(finalBlock ? 0x01 : 0x00);
        const uint16_t length = static_cast<uint16_t>(blockSize);
        const uint16_t inverseLength = static_cast<uint16_t>(~length);
        deflate.push_back(static_cast<uint8_t>(length));
        deflate.push_back(static_cast<uint8_t>(length >> 8));
        deflate.push_back(static_cast<uint8_t>(inverseLength));
        deflate.push_back(static_cast<uint8_t>(inverseLength >> 8));
        deflate.insert(deflate.end(), raw.begin() + cursor, raw.begin() + cursor + blockSize);
        cursor += blockSize;
    }
    uint32_t adlerA = 1;
    uint32_t adlerB = 0;
    for (uint8_t byte : raw) {
        adlerA = (adlerA + byte) % 65521u;
        adlerB = (adlerB + adlerA) % 65521u;
    }
    append_u32_be(deflate, (adlerB << 16) | adlerA);

    png.clear();
    constexpr uint8_t signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    png.insert(png.end(), std::begin(signature), std::end(signature));
    std::vector<uint8_t> ihdr;
    append_u32_be(ihdr, texture.w);
    append_u32_be(ihdr, texture.h);
    ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});
    append_png_chunk(png, "IHDR", ihdr.data(), ihdr.size());
    append_png_chunk(png, "IDAT", deflate.data(), deflate.size());
    append_png_chunk(png, "IEND", nullptr, 0);
    return true;
}

bool add_buffer_view(std::vector<BufferView>& views, size_t offset, size_t length,
    uint32_t stride, uint32_t target, std::string& error) {
    if (offset > UINT32_MAX || length > UINT32_MAX) {
        error = "GLB buffer exceeds the 32-bit format limit";
        return false;
    }
    views.push_back({static_cast<uint32_t>(offset), static_cast<uint32_t>(length), stride, target});
    return true;
}

bool append_accessor_bounds(std::string& json, const Accessor& accessor) {
    if (!accessor.hasBounds) return true;
    json += ",\"min\":[";
    for (int i = 0; i < 3; ++i) {
        if (i != 0) json += ',';
        if (!append_json_float(json, accessor.min[i])) return false;
    }
    json += "],\"max\":[";
    for (int i = 0; i < 3; ++i) {
        if (i != 0) json += ',';
        if (!append_json_float(json, accessor.max[i])) return false;
    }
    json += ']';
    return true;
}

}  // namespace

bool build_glb(const MeshData& mesh, int32_t soloShape, std::vector<uint8_t>& out,
    std::string& error) {
    out.clear();
    error.clear();
    if (mesh.textures == nullptr) {
        error = "model texture list is unavailable";
        return false;
    }

    std::vector<uint8_t> binary;
    std::vector<BufferView> bufferViews;
    std::vector<Accessor> accessors;
    std::vector<Primitive> primitives;
    std::vector<Image> images;
    std::vector<int32_t> materialByTexture(mesh.textures->size(), -1);
    bool foundShape = soloShape < 0;

    for (const SubMesh& sub : mesh.subs) {
        if (soloShape >= 0 && sub.shapeIndex != static_cast<uint32_t>(soloShape)) continue;
        foundShape = true;
        if (sub.vertexCount == 0 || sub.vertexCount % 3 != 0) continue;
        const uint64_t endVertex = static_cast<uint64_t>(sub.firstVertex) + sub.vertexCount;
        if (endVertex > mesh.vertices.size() / 6u ||
            static_cast<uint64_t>(sub.vertexCount) * 20u > UINT32_MAX) {
            error = "shape vertex range is outside the model snapshot";
            return false;
        }

        align4(binary);
        const size_t viewOffset = binary.size();
        float min[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()};
        float max[3] = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max()};
        for (uint32_t i = 0; i < sub.vertexCount; ++i) {
            const size_t vertex = (static_cast<size_t>(sub.firstVertex) + i) * 6u;
            for (int axis = 0; axis < 5; ++axis) {
                const float value = mesh.vertices[vertex + static_cast<size_t>(axis)];
                if (!std::isfinite(value)) {
                    error = "model contains a non-finite vertex attribute";
                    return false;
                }
                append_float_le(binary, value);
                if (axis < 3) {
                    min[axis] = std::min(min[axis], value);
                    max[axis] = std::max(max[axis], value);
                }
            }
        }
        if (!add_buffer_view(bufferViews, viewOffset, binary.size() - viewOffset, 20, 34962, error)) {
            return false;
        }
        const uint32_t viewIndex = static_cast<uint32_t>(bufferViews.size() - 1);
        const uint32_t positionAccessor = static_cast<uint32_t>(accessors.size());
        accessors.push_back({viewIndex, 0, sub.vertexCount, "VEC3", true,
            {min[0], min[1], min[2]}, {max[0], max[1], max[2]}});
        const uint32_t uvAccessor = static_cast<uint32_t>(accessors.size());
        accessors.push_back({viewIndex, 12, sub.vertexCount, "VEC2", false, {}, {}});
        primitives.push_back({positionAccessor, uvAccessor, sub.texture, -1});
    }

    if (!foundShape) {
        error = "selected shape is not present in the model";
        return false;
    }
    if (primitives.empty()) {
        error = "model has no exportable triangles";
        return false;
    }

    for (const Primitive& primitive : primitives) {
        const int32_t textureIndex = primitive.textureIndex;
        if (textureIndex < 0) continue;
        if (static_cast<size_t>(textureIndex) >= mesh.textures->size()) {
            error = "shape references a texture outside the model snapshot";
            return false;
        }
        if (materialByTexture[static_cast<size_t>(textureIndex)] >= 0) continue;
        const TexImage& texture = (*mesh.textures)[static_cast<size_t>(textureIndex)];
        std::vector<uint8_t> png;
        if (!encode_png(texture, png)) {
            error = "shape texture has no valid decoded RGBA image";
            return false;
        }
        align4(binary);
        const size_t viewOffset = binary.size();
        binary.insert(binary.end(), png.begin(), png.end());
        if (!add_buffer_view(bufferViews, viewOffset, binary.size() - viewOffset, 0, 0, error)) {
            return false;
        }
        materialByTexture[static_cast<size_t>(textureIndex)] =
            static_cast<int32_t>(images.size());
        images.push_back({static_cast<uint32_t>(bufferViews.size() - 1)});
    }
    for (Primitive& primitive : primitives) {
        if (primitive.textureIndex >= 0 &&
            static_cast<size_t>(primitive.textureIndex) < materialByTexture.size()) {
            primitive.material = materialByTexture[static_cast<size_t>(primitive.textureIndex)];
        }
    }
    if (binary.size() > UINT32_MAX) {
        error = "GLB buffer exceeds the 32-bit format limit";
        return false;
    }

    std::string json;
    json.reserve(1024u + primitives.size() * 180u + bufferViews.size() * 90u);
    json += "{\"asset\":{\"version\":\"2.0\",\"generator\":\"Dusklight Model Inspector\"},"
        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
        "\"meshes\":[{\"primitives\":[";
    for (size_t i = 0; i < primitives.size(); ++i) {
        if (i != 0) json += ',';
        const Primitive& primitive = primitives[i];
        json += "{\"attributes\":{\"POSITION\":";
        append_json_integer(json, primitive.positionAccessor);
        json += ",\"TEXCOORD_0\":";
        append_json_integer(json, primitive.uvAccessor);
        json += "},\"mode\":4";
        if (primitive.material >= 0) {
            json += ",\"material\":";
            append_json_integer(json, primitive.material);
        }
        json += '}';
    }
    json += "]}],\"accessors\":[";
    for (size_t i = 0; i < accessors.size(); ++i) {
        if (i != 0) json += ',';
        const Accessor& accessor = accessors[i];
        json += "{\"bufferView\":";
        append_json_integer(json, accessor.bufferView);
        json += ",\"byteOffset\":";
        append_json_integer(json, accessor.byteOffset);
        json += ",\"componentType\":5126,\"count\":";
        append_json_integer(json, accessor.count);
        json += ",\"type\":\"";
        json += accessor.type;
        json += '"';
        if (!append_accessor_bounds(json, accessor)) {
            error = "could not encode vertex bounds in GLB JSON";
            return false;
        }
        json += '}';
    }
    json += "],\"bufferViews\":[";
    for (size_t i = 0; i < bufferViews.size(); ++i) {
        if (i != 0) json += ',';
        const BufferView& view = bufferViews[i];
        json += "{\"buffer\":0,\"byteOffset\":";
        append_json_integer(json, view.offset);
        json += ",\"byteLength\":";
        append_json_integer(json, view.length);
        if (view.stride != 0) {
            json += ",\"byteStride\":";
            append_json_integer(json, view.stride);
        }
        if (view.target != 0) {
            json += ",\"target\":";
            append_json_integer(json, view.target);
        }
        json += '}';
    }
    json += "],\"buffers\":[{\"byteLength\":";
    append_json_integer(json, binary.size());
    json += "}]";
    if (!images.empty()) {
        json += ",\"samplers\":[{\"magFilter\":9729,\"minFilter\":9729,"
            "\"wrapS\":10497,\"wrapT\":10497}],\"images\":[";
        for (size_t i = 0; i < images.size(); ++i) {
            if (i != 0) json += ',';
            json += "{\"bufferView\":";
            append_json_integer(json, images[i].bufferView);
            json += ",\"mimeType\":\"image/png\"}";
        }
        json += "],\"textures\":[";
        for (size_t i = 0; i < images.size(); ++i) {
            if (i != 0) json += ',';
            json += "{\"sampler\":0,\"source\":";
            append_json_integer(json, i);
            json += '}';
        }
        json += "],\"materials\":[";
        for (size_t i = 0; i < images.size(); ++i) {
            if (i != 0) json += ',';
            json += "{\"doubleSided\":true,\"alphaMode\":\"BLEND\","
                "\"pbrMetallicRoughness\":{\"baseColorTexture\":{\"index\":";
            append_json_integer(json, i);
            json += "},\"metallicFactor\":0,\"roughnessFactor\":1}}";
        }
        json += ']';
    }
    json += '}';

    while ((json.size() & 3u) != 0) json.push_back(' ');
    align4(binary);
    const uint64_t totalSize = 12u + 8u + json.size() + 8u + binary.size();
    if (totalSize > UINT32_MAX) {
        error = "GLB exceeds the 32-bit file size limit";
        return false;
    }
    out.reserve(static_cast<size_t>(totalSize));
    append_u32_le(out, 0x46546C67u);
    append_u32_le(out, 2u);
    append_u32_le(out, static_cast<uint32_t>(totalSize));
    append_u32_le(out, static_cast<uint32_t>(json.size()));
    append_u32_le(out, 0x4E4F534Au);
    out.insert(out.end(), json.begin(), json.end());
    append_u32_le(out, static_cast<uint32_t>(binary.size()));
    append_u32_le(out, 0x004E4942u);
    out.insert(out.end(), binary.begin(), binary.end());
    return true;
}

}  // namespace mi::gltf
