#pragma once
// Tiny column-major matrix helpers (WebGPU clip space, right-handed view, -Z forward).
#include <cmath>

namespace mi {

struct Mat4 {
    float m[16];  // element (row r, col c) = m[c * 4 + r]
};

inline Mat4 mat_mul(const Mat4& a, const Mat4& b) {
    Mat4 o{};
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) {
                s += a.m[k * 4 + r] * b.m[c * 4 + k];
            }
            o.m[c * 4 + r] = s;
        }
    }
    return o;
}

inline Mat4 mat_perspective_reversed_z(float fovyRad, float aspect, float zn, float zf) {
    const float f = 1.0f / std::tan(fovyRad * 0.5f);
    Mat4 o{};
    o.m[0] = f / aspect;
    o.m[5] = f;
    o.m[10] = zn / (zf - zn);
    o.m[11] = -1.0f;
    o.m[14] = zn * zf / (zf - zn);
    return o;
}

inline Mat4 mat_look_at(const float eye[3], const float ctr[3], const float up[3]) {
    float f[3] = {ctr[0] - eye[0], ctr[1] - eye[1], ctr[2] - eye[2]};
    float fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    if (fl < 1e-6f) fl = 1.0f;
    for (float& v : f) v /= fl;
    float s[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2],
        f[0] * up[1] - f[1] * up[0]};
    float sl = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    if (sl < 1e-6f) sl = 1.0f;
    for (float& v : s) v /= sl;
    const float u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2],
        s[0] * f[1] - s[1] * f[0]};
    Mat4 o{};
    o.m[0] = s[0]; o.m[4] = s[1]; o.m[8] = s[2];
    o.m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    o.m[1] = u[0]; o.m[5] = u[1]; o.m[9] = u[2];
    o.m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    o.m[2] = -f[0]; o.m[6] = -f[1]; o.m[10] = -f[2];
    o.m[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
    o.m[15] = 1.0f;
    return o;
}

}  // namespace mi
