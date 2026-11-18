#include "gctex.hpp"

#include <cstring>

namespace mi {
namespace {

inline uint16_t be16(const uint8_t* p) { return static_cast<uint16_t>((p[0] << 8) | p[1]); }
inline uint8_t e5(int v) { return static_cast<uint8_t>((v << 3) | (v >> 2)); }
inline uint8_t e6(int v) { return static_cast<uint8_t>((v << 2) | (v >> 4)); }

void c_rgb565(uint16_t v, uint8_t* c) {
    c[0] = e5((v >> 11) & 31);
    c[1] = e6((v >> 5) & 63);
    c[2] = e5(v & 31);
    c[3] = 255;
}

void c_rgb5a3(uint16_t v, uint8_t* c) {
    if (v & 0x8000) {
        c[0] = e5((v >> 10) & 31);
        c[1] = e5((v >> 5) & 31);
        c[2] = e5(v & 31);
        c[3] = 255;
    } else {
        const int a = (v >> 12) & 7;
        c[0] = static_cast<uint8_t>(((v >> 8) & 15) * 17);
        c[1] = static_cast<uint8_t>(((v >> 4) & 15) * 17);
        c[2] = static_cast<uint8_t>((v & 15) * 17);
        c[3] = static_cast<uint8_t>((a << 5) | (a << 2) | (a >> 1));
    }
}

void c_ia8(uint16_t v, uint8_t* c) {
    c[0] = c[1] = c[2] = static_cast<uint8_t>(v & 0xFF);
    c[3] = static_cast<uint8_t>(v >> 8);
}

void pal_color(uint8_t palFmt, const uint8_t* pal, uint32_t palCount, uint32_t idx, uint8_t* c) {
    if (pal == nullptr || idx >= palCount) {
        c[0] = c[1] = c[2] = c[3] = 0;
        return;
    }
    const uint16_t v = be16(pal + idx * 2);
    switch (palFmt) {
    case 0: c_ia8(v, c); break;
    case 1: c_rgb565(v, c); break;
    default: c_rgb5a3(v, c); break;
    }
}

struct BlockInfo {
    uint32_t bw, bh, bytes;
};

bool block_info(uint8_t fmt, BlockInfo& b) {
    switch (fmt) {
    case 0: b = {8, 8, 32}; return true;   // I4
    case 1: b = {8, 4, 32}; return true;   // I8
    case 2: b = {8, 4, 32}; return true;   // IA4
    case 3: b = {4, 4, 32}; return true;   // IA8
    case 4: b = {4, 4, 32}; return true;   // RGB565
    case 5: b = {4, 4, 32}; return true;   // RGB5A3
    case 6: b = {4, 4, 64}; return true;   // RGBA8
    case 8: b = {8, 8, 32}; return true;   // C4
    case 9: b = {8, 4, 32}; return true;   // C8
    case 10: b = {4, 4, 32}; return true;  // C14X2
    case 14: b = {8, 8, 32}; return true;  // CMPR
    default: return false;
    }
}

// Decodes one block into tile[bh][bw][4].
void decode_block(uint8_t fmt, const uint8_t* s, const BlockInfo& b, const uint8_t* pal,
    uint8_t palFmt, uint32_t palCount, uint8_t* tile) {
    auto px = [&](uint32_t x, uint32_t y) { return tile + (y * b.bw + x) * 4; };
    switch (fmt) {
    case 0:  // I4
        for (uint32_t y = 0; y < 8; ++y)
            for (uint32_t x = 0; x < 8; ++x) {
                const uint8_t byte = s[y * 4 + x / 2];
                const uint8_t n = (x & 1) ? (byte & 15) : (byte >> 4);
                uint8_t* c = px(x, y);
                c[0] = c[1] = c[2] = static_cast<uint8_t>(n * 17);
                c[3] = 255;
            }
        break;
    case 1:  // I8
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 8; ++x) {
                uint8_t* c = px(x, y);
                c[0] = c[1] = c[2] = s[y * 8 + x];
                c[3] = 255;
            }
        break;
    case 2:  // IA4
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 8; ++x) {
                const uint8_t v = s[y * 8 + x];
                uint8_t* c = px(x, y);
                c[0] = c[1] = c[2] = static_cast<uint8_t>((v & 15) * 17);
                c[3] = static_cast<uint8_t>((v >> 4) * 17);
            }
        break;
    case 3:  // IA8
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 4; ++x) c_ia8(be16(s + (y * 4 + x) * 2), px(x, y));
        break;
    case 4:  // RGB565
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 4; ++x) c_rgb565(be16(s + (y * 4 + x) * 2), px(x, y));
        break;
    case 5:  // RGB5A3
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 4; ++x) c_rgb5a3(be16(s + (y * 4 + x) * 2), px(x, y));
        break;
    case 6:  // RGBA8: 16 x (A,R) then 16 x (G,B)
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 4; ++x) {
                const uint32_t i = y * 4 + x;
                uint8_t* c = px(x, y);
                c[3] = s[i * 2];
                c[0] = s[i * 2 + 1];
                c[1] = s[32 + i * 2];
                c[2] = s[32 + i * 2 + 1];
            }
        break;
    case 8:  // C4
        for (uint32_t y = 0; y < 8; ++y)
            for (uint32_t x = 0; x < 8; ++x) {
                const uint8_t byte = s[y * 4 + x / 2];
                const uint32_t idx = (x & 1) ? (byte & 15) : (byte >> 4);
                pal_color(palFmt, pal, palCount, idx, px(x, y));
            }
        break;
    case 9:  // C8
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 8; ++x)
                pal_color(palFmt, pal, palCount, s[y * 8 + x], px(x, y));
        break;
    case 10:  // C14X2
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 4; ++x)
                pal_color(palFmt, pal, palCount, be16(s + (y * 4 + x) * 2) & 0x3FFF, px(x, y));
        break;
    case 14:  // CMPR (DXT1-like), four 4x4 sub-blocks per 8x8 block
        for (uint32_t sb = 0; sb < 4; ++sb) {
            const uint8_t* d = s + sb * 8;
            const uint32_t ox = (sb & 1) * 4, oy = (sb >> 1) * 4;
            const uint16_t c0 = be16(d), c1 = be16(d + 2);
            uint8_t col[4][4];
            c_rgb565(c0, col[0]);
            c_rgb565(c1, col[1]);
            if (c0 > c1) {
                for (int k = 0; k < 3; ++k) {
                    col[2][k] = static_cast<uint8_t>((2 * col[0][k] + col[1][k]) / 3);
                    col[3][k] = static_cast<uint8_t>((col[0][k] + 2 * col[1][k]) / 3);
                }
                col[2][3] = col[3][3] = 255;
            } else {
                for (int k = 0; k < 3; ++k) {
                    col[2][k] = static_cast<uint8_t>((col[0][k] + col[1][k]) / 2);
                    col[3][k] = 0;
                }
                col[2][3] = 255;
                col[3][3] = 0;
            }
            for (uint32_t y = 0; y < 4; ++y) {
                const uint8_t bits = d[4 + y];
                for (uint32_t x = 0; x < 4; ++x) {
                    std::memcpy(px(ox + x, oy + y), col[(bits >> (6 - 2 * x)) & 3], 4);
                }
            }
        }
        break;
    default: break;
    }
}

}  // namespace

bool decode_gc_texture(const uint8_t* img, uint32_t w, uint32_t h, uint8_t fmt,
    const uint8_t* pal, uint8_t palFmt, uint32_t palCount, std::vector<uint8_t>& rgba) {
    BlockInfo b{};
    if (img == nullptr || w == 0 || h == 0 || w > 4096 || h > 4096 || !block_info(fmt, b)) {
        return false;
    }
    rgba.assign(static_cast<size_t>(w) * h * 4, 0);
    const uint32_t bx = (w + b.bw - 1) / b.bw;
    const uint32_t by = (h + b.bh - 1) / b.bh;
    uint8_t tile[64 * 4];
    for (uint32_t y = 0; y < by; ++y) {
        for (uint32_t x = 0; x < bx; ++x) {
            const uint8_t* src = img + static_cast<size_t>(y * bx + x) * b.bytes;
            decode_block(fmt, src, b, pal, palFmt, palCount, tile);
            for (uint32_t ty = 0; ty < b.bh; ++ty) {
                const uint32_t py = y * b.bh + ty;
                if (py >= h) break;
                for (uint32_t tx = 0; tx < b.bw; ++tx) {
                    const uint32_t pxl = x * b.bw + tx;
                    if (pxl >= w) break;
                    std::memcpy(&rgba[(static_cast<size_t>(py) * w + pxl) * 4],
                        tile + (ty * b.bw + tx) * 4, 4);
                }
            }
        }
    }
    return true;
}

const char* gc_format_name(uint8_t fmt) {
    switch (fmt) {
    case 0: return "I4";
    case 1: return "I8";
    case 2: return "IA4";
    case 3: return "IA8";
    case 4: return "RGB565";
    case 5: return "RGB5A3";
    case 6: return "RGBA8";
    case 8: return "C4";
    case 9: return "C8";
    case 10: return "C14X2";
    case 14: return "CMPR";
    default: return "?";
    }
}

}  // namespace mi
