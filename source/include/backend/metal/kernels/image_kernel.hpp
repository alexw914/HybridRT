// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: source/include/backend/metal/kernels/image_kernel.hpp
// Description: Declares the source of the fused host-image preprocessing Metal
//              kernel, embedded from image_kernel.metal at build time.
// -----------------------------------------------------------------------------

#ifndef HRT_METAL_KERNELS_IMAGE_KERNEL_HPP
#define HRT_METAL_KERNELS_IMAGE_KERNEL_HPP

namespace hrt::backend {

// Metal Shading Language source of source/backend/metal/kernels/image_kernel.metal, compiled at run time with
// newLibraryWithSource; null-terminated.
auto metal_image_kernel() -> const char*;

} // namespace hrt::backend

#endif // HRT_METAL_KERNELS_IMAGE_KERNEL_HPP
