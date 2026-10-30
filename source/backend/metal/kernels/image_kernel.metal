// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/backend/metal/kernels/image_kernel.metal
// Description: Implements the fused host-image preprocessing Metal kernel.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;
struct Params {
  uint sw, sh, dw, dh, rw, rh, left, top;
  uint src_row, dst_row, dst_plane;
  uint swap, interpolation, normalization, source, origin_x, origin_y, chroma, float_output;
  float fill[3], mean[3], scale[3];
};
float3 fetch(device const uchar* src, constant Params& p, uint x, uint y) {
  if (p.source == 0) {
    uint offset = y * p.src_row + x * 3;
    return float3(src[offset], src[offset + 1], src[offset + 2]);
  }
  uint gx = x + p.origin_x, gy = y + p.origin_y;
  uint uv = p.chroma + (gy / 2) * p.src_row + (gx / 2) * 2;
  int u = int(src[uv + (p.source == 1 ? 0 : 1)]) - 128;
  int v = int(src[uv + (p.source == 1 ? 1 : 0)]) - 128;
  int luma = max(0, int(src[gy * p.src_row + gx]) - 16) * 1220542;
  int rounding = 1 << 19;
  return float3(clamp((luma + rounding + 2116026 * u) >> 20, 0, 255),
                clamp((luma + rounding - 852492 * v - 409993 * u) >> 20, 0, 255),
                clamp((luma + rounding + 1673527 * v) >> 20, 0, 255));
}

float4 cubic_weights(float f) {
  const float a = -0.75f;
  float c0 = ((a * (f + 1) - 5 * a) * (f + 1) + 8 * a) * (f + 1) - 4 * a;
  float c1 = ((a + 2) * f - (a + 3)) * f * f + 1;
  float t = 1 - f;
  float c2 = ((a + 2) * t - (a + 3)) * t * t + 1;
  return rint(float4(c0, c1, c2, 1 - c0 - c1 - c2) * 2048.0f);
}

kernel void preprocess(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                       constant Params& p [[buffer(2)]], uint2 xy [[thread_position_in_grid]]) {
  if (xy.x >= p.dw || xy.y >= p.dh) return;
  const uint index = xy.y * p.dst_row + (p.dst_plane == 0 ? xy.x * 3 : xy.x);
  const bool inside = xy.x >= p.left && xy.x < p.left + p.rw &&
                      xy.y >= p.top && xy.y < p.top + p.rh;
  float3 value = float3(p.fill[0], p.fill[1], p.fill[2]);
  if (inside) {
    if (p.interpolation == 1) {
      const uint x = uint(ulong(xy.x - p.left) * p.sw / p.rw);
      const uint y = uint(ulong(xy.y - p.top) * p.sh / p.rh);
      value = fetch(src, p, min(x, p.sw - 1), min(y, p.sh - 1));
    } else if (p.interpolation == 2) {
      float fx = (float(xy.x - p.left) + 0.5f) * float(p.sw) / float(p.rw) - 0.5f;
      float fy = (float(xy.y - p.top) + 0.5f) * float(p.sh) / float(p.rh) - 0.5f;
      int ix = int(floor(fx)), iy = int(floor(fy));
      float4 wx = cubic_weights(fx - float(ix)), wy = cubic_weights(fy - float(iy));
      float3 sum = float3(0);
      for (int j = 0; j < 4; ++j) {
        float3 row = float3(0);
        for (int i = 0; i < 4; ++i)
          row += fetch(src, p, uint(clamp(ix + i - 1, 0, int(p.sw) - 1)),
                       uint(clamp(iy + j - 1, 0, int(p.sh) - 1))) * wx[i];
        sum += row * wy[j] / (2048.0f * 2048.0f);
      }
      value = clamp(rint(sum), 0.0f, 255.0f);
    } else {
      const float fx = (float(xy.x - p.left) + 0.5f) * float(p.sw) / float(p.rw) - 0.5f;
      const float fy = (float(xy.y - p.top) + 0.5f) * float(p.sh) / float(p.rh) - 0.5f;
      const int ix = int(floor(fx)), iy = int(floor(fy));
      const float ax = fx - float(ix), ay = fy - float(iy);
      const uint x0 = uint(clamp(ix, 0, int(p.sw) - 1)), x1 = uint(clamp(ix + 1, 0, int(p.sw) - 1));
      const uint y0 = uint(clamp(iy, 0, int(p.sh) - 1)), y1 = uint(clamp(iy + 1, 0, int(p.sh) - 1));
      const float3 v0 = fetch(src, p, x0, y0), v1 = fetch(src, p, x1, y0);
      const float3 v2 = fetch(src, p, x0, y1), v3 = fetch(src, p, x1, y1);
      // shortcut: float interpolation may differ by one uint8 level; use fixed-point for bit-exact OpenCV output.
      value = floor(mix(mix(v0, v1, ax), mix(v2, v3, ax), ay) + 0.5f);
    }
  }
  for (uint channel = 0; channel < 3; ++channel) {
    float result = value[p.swap ? 2 - channel : channel];
    if (p.normalization == 1 || (p.normalization == 2 && inside))
      result = (result - p.mean[channel]) * p.scale[channel];
    uint out = index + (p.dst_plane == 0 ? channel : channel * p.dst_plane);
    if (p.float_output)
      reinterpret_cast<device float*>(dst)[out] = result;
    else
      dst[out] = uchar(clamp(rint(result), 0.0f, 255.0f));
  }
}
