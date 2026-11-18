#pragma once
#include <cstdint>
#include <vector>

namespace mi::parser {

// Decodes the base level of a GameCube texture (tiled, big-endian) into RGBA8.
// fmt: GXTexFmt (I4=0 I8=1 IA4=2 IA8=3 RGB565=4 RGB5A3=5 RGBA8=6 C4=8 C8=9 C14X2=10 CMPR=14).
// pal/palFmt/palCount are only used by the C* formats (palFmt: IA8=0 RGB565=1 RGB5A3=2).
bool decode_gc_texture(const uint8_t* img, uint32_t w, uint32_t h, uint8_t fmt,
    const uint8_t* pal, uint8_t palFmt, uint32_t palCount, std::vector<uint8_t>& rgba);

const char* gc_format_name(uint8_t fmt);

}  // namespace mi::parser
