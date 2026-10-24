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
  uint swap, nearest, normalization;
  float fill[3], mean[3], scale[3];
};
kernel void preprocess(device const uchar* src [[buffer(0)]], device float* dst [[buffer(1)]],
                       constant Params& p [[buffer(2)]], uint2 xy [[thread_position_in_grid]]) {
  if (xy.x >= p.dw || xy.y >= p.dh) return;
  const uint index = xy.y * p.dst_row + xy.x;
  const bool inside = xy.x >= p.left && xy.x < p.left + p.rw &&
                      xy.y >= p.top && xy.y < p.top + p.rh;
  float3 value = float3(p.fill[0], p.fill[1], p.fill[2]);
  if (inside) {
    if (p.nearest) {
      const uint x = uint(ulong(xy.x - p.left) * p.sw / p.rw);
      const uint y = uint(ulong(xy.y - p.top) * p.sh / p.rh);
      const uint offset = min(y, p.sh - 1) * p.src_row + min(x, p.sw - 1) * 3;
      value = float3(src[offset], src[offset + 1], src[offset + 2]);
    } else {
      const float fx = (float(xy.x - p.left) + 0.5f) * float(p.sw) / float(p.rw) - 0.5f;
      const float fy = (float(xy.y - p.top) + 0.5f) * float(p.sh) / float(p.rh) - 0.5f;
      const int ix = int(floor(fx)), iy = int(floor(fy));
      const float ax = fx - float(ix), ay = fy - float(iy);
      const uint x0 = uint(clamp(ix, 0, int(p.sw) - 1)), x1 = uint(clamp(ix + 1, 0, int(p.sw) - 1));
      const uint y0 = uint(clamp(iy, 0, int(p.sh) - 1)), y1 = uint(clamp(iy + 1, 0, int(p.sh) - 1));
      const uint a = y0 * p.src_row + x0 * 3, b = y0 * p.src_row + x1 * 3;
      const uint c = y1 * p.src_row + x0 * 3, d = y1 * p.src_row + x1 * 3;
      const float3 v0 = float3(src[a], src[a + 1], src[a + 2]), v1 = float3(src[b], src[b + 1], src[b + 2]);
      const float3 v2 = float3(src[c], src[c + 1], src[c + 2]), v3 = float3(src[d], src[d + 1], src[d + 2]);
      // shortcut: float interpolation may differ by one uint8 level; use fixed-point for bit-exact OpenCV output.
      value = floor(mix(mix(v0, v1, ax), mix(v2, v3, ax), ay) + 0.5f);
    }
  }
  for (uint channel = 0; channel < 3; ++channel) {
    float result = value[p.swap ? 2 - channel : channel];
    if (p.normalization == 1 || (p.normalization == 2 && inside))
      result = (result - p.mean[channel]) * p.scale[channel];
    dst[channel * p.dst_plane + index] = result;
  }
}
