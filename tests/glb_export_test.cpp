#include "export/glb_exporter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {

uint32_t read_u32_le(const std::vector<uint8_t>& data, size_t offset) {
    return static_cast<uint32_t>(data[offset]) |
        (static_cast<uint32_t>(data[offset + 1]) << 8) |
        (static_cast<uint32_t>(data[offset + 2]) << 16) |
        (static_cast<uint32_t>(data[offset + 3]) << 24);
}

size_t count_occurrences(const std::string& text, const std::string& needle) {
    size_t count = 0;
    for (size_t offset = 0; (offset = text.find(needle, offset)) != std::string::npos;
         offset += needle.size()) {
        ++count;
    }
    return count;
}

uint32_t read_u32_be(const std::vector<uint8_t>& data, size_t offset) {
    return (static_cast<uint32_t>(data[offset]) << 24) |
        (static_cast<uint32_t>(data[offset + 1]) << 16) |
        (static_cast<uint32_t>(data[offset + 2]) << 8) |
        static_cast<uint32_t>(data[offset + 3]);
}

uint32_t crc32(const std::vector<uint8_t>& data, size_t offset, size_t length) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < length; ++i) {
        crc ^= data[offset + i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

bool valid_test_png(const std::vector<uint8_t>& data, size_t start,
    const std::array<uint8_t, 4>& expectedRgba, size_t& pngLength) {
    constexpr uint8_t signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (start + sizeof(signature) > data.size() ||
        !std::equal(std::begin(signature), std::end(signature), data.begin() + start)) {
        return false;
    }

    size_t cursor = start + sizeof(signature);
    bool foundHeader = false;
    bool foundData = false;
    bool foundEnd = false;
    while (cursor + 12 <= data.size()) {
        const uint32_t chunkLength = read_u32_be(data, cursor);
        if (static_cast<uint64_t>(cursor) + chunkLength + 12u > data.size()) return false;
        const size_t typeOffset = cursor + 4;
        const size_t payloadOffset = cursor + 8;
        const size_t crcOffset = payloadOffset + chunkLength;
        if (crc32(data, typeOffset, static_cast<size_t>(chunkLength) + 4u) !=
            read_u32_be(data, crcOffset)) {
            return false;
        }

        if (data[typeOffset] == 'I' && data[typeOffset + 1] == 'H' &&
            data[typeOffset + 2] == 'D' && data[typeOffset + 3] == 'R') {
            if (chunkLength != 13 || read_u32_be(data, payloadOffset) != 1 ||
                read_u32_be(data, payloadOffset + 4) != 1 ||
                data[payloadOffset + 8] != 8 || data[payloadOffset + 9] != 6) {
                return false;
            }
            foundHeader = true;
        } else if (data[typeOffset] == 'I' && data[typeOffset + 1] == 'D' &&
            data[typeOffset + 2] == 'A' && data[typeOffset + 3] == 'T') {
            if (chunkLength != 16 || data[payloadOffset] != 0x78 ||
                data[payloadOffset + 1] != 0x01 || data[payloadOffset + 2] != 0x01 ||
                data[payloadOffset + 3] != 5 || data[payloadOffset + 4] != 0 ||
                data[payloadOffset + 5] != 0xFA || data[payloadOffset + 6] != 0xFF ||
                data[payloadOffset + 7] != 0 ||
                !std::equal(expectedRgba.begin(), expectedRgba.end(),
                    data.begin() + payloadOffset + 8)) {
                return false;
            }
            uint32_t adlerA = 1;
            uint32_t adlerB = 0;
            for (size_t i = 0; i < 5; ++i) {
                const uint8_t byte = data[payloadOffset + 7 + i];
                adlerA = (adlerA + byte) % 65521u;
                adlerB = (adlerB + adlerA) % 65521u;
            }
            if (read_u32_be(data, payloadOffset + 12) != ((adlerB << 16) | adlerA)) {
                return false;
            }
            foundData = true;
        } else if (data[typeOffset] == 'I' && data[typeOffset + 1] == 'E' &&
            data[typeOffset + 2] == 'N' && data[typeOffset + 3] == 'D') {
            if (chunkLength != 0) return false;
            cursor += 12;
            foundEnd = true;
            break;
        }
        cursor += static_cast<size_t>(chunkLength) + 12u;
    }
    pngLength = cursor - start;
    return foundHeader && foundData && foundEnd;
}

bool valid_glb(const std::vector<uint8_t>& data, size_t expectedPrimitives,
    const std::vector<std::array<uint8_t, 4>>& expectedImages) {
    if (data.size() < 28 || read_u32_le(data, 0) != 0x46546C67u ||
        read_u32_le(data, 4) != 2u || read_u32_le(data, 8) != data.size()) {
        return false;
    }
    const uint32_t jsonLength = read_u32_le(data, 12);
    if (read_u32_le(data, 16) != 0x4E4F534Au || 20u + jsonLength + 8u > data.size()) {
        return false;
    }
    const std::string json(data.begin() + 20, data.begin() + 20 + jsonLength);
    const size_t binaryHeader = 20u + jsonLength;
    if (read_u32_le(data, binaryHeader + 4) != 0x004E4942u ||
        binaryHeader + 8u + read_u32_le(data, binaryHeader) != data.size()) {
        return false;
    }
    constexpr char pngSignature[] = "\x89PNG\r\n\x1A\n";
    const std::string binary(data.begin() + binaryHeader + 8, data.end());
    if (count_occurrences(json, "\"mode\":4") != expectedPrimitives ||
        count_occurrences(json, "\"mimeType\":\"image/png\"") != expectedImages.size() ||
        count_occurrences(json, "\"baseColorTexture\"") != expectedImages.size() ||
        count_occurrences(binary, std::string(pngSignature, sizeof(pngSignature) - 1)) !=
            expectedImages.size()) {
        return false;
    }

    const std::vector<uint8_t> signature(std::begin(pngSignature), std::end(pngSignature) - 1);
    size_t searchFrom = binaryHeader + 8;
    for (const auto& expected : expectedImages) {
        const auto image = std::search(data.begin() + searchFrom, data.end(),
            signature.begin(), signature.end());
        if (image == data.end()) return false;
        const size_t imageOffset = static_cast<size_t>(image - data.begin());
        size_t imageLength = 0;
        if (!valid_test_png(data, imageOffset, expected, imageLength)) return false;
        searchFrom = imageOffset + imageLength;
    }
    return true;
}

}  // namespace

int main() {
    mi::MeshData mesh;
    mesh.vertices = {
        0, 0, 0, 0, 0, 4,
        1, 0, 0, 1, 0, 4,
        0, 1, 0, 0, 1, 4,
        2, 0, 0, 0, 0, 9,
        3, 0, 0, 1, 0, 9,
        2, 1, 0, 0, 1, 9,
    };
    mesh.subs = {
        {0, 3, 0, 4, "first"},
        {3, 3, 1, 9, "second"},
    };
    mesh.textures = std::make_shared<const std::vector<mi::TexImage>>(
        std::vector<mi::TexImage>{
            {1, 1, {255, 0, 0, 255}, "red"},
            {1, 1, {0, 255, 0, 255}, "green"},
        });

    std::vector<uint8_t> glb;
    std::string error;
    if (!mi::gltf::build_glb(mesh, -1, glb, error) ||
        !valid_glb(glb, 2, {{{255, 0, 0, 255}}, {{0, 255, 0, 255}}})) {
        std::cerr << "all-shapes GLB export failed: " << error << '\n';
        return 1;
    }
    if (!mi::gltf::build_glb(mesh, 9, glb, error) ||
        !valid_glb(glb, 1, {{{0, 255, 0, 255}}})) {
        std::cerr << "single-shape GLB export failed: " << error << '\n';
        return 1;
    }
    if (mi::gltf::build_glb(mesh, 99, glb, error) || error.empty()) {
        std::cerr << "missing-shape export did not report an error\n";
        return 1;
    }
    const auto textures = mesh.textures;
    mesh.textures.reset();
    if (mi::gltf::build_glb(mesh, -1, glb, error) || error.empty()) {
        std::cerr << "missing texture list did not report an error\n";
        return 1;
    }
    mesh.textures = textures;
    return 0;
}
