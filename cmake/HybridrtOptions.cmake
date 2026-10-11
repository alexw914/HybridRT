# ---------------------------------------------------------------------------
# HybridRT build options
# ---------------------------------------------------------------------------
# Inference backends are enabled by pointing <SDK>_ROOT at an installed SDK,
# see HybridrtBackends.cmake.
# ---------------------------------------------------------------------------

option(HYBRIDRT_BUILD_TOOLS "Build the hrt-compress packaging tool" ${HYBRIDRT_IS_TOP_LEVEL})
option(HYBRIDRT_BUILD_TESTS "Build test suite" ${HYBRIDRT_IS_TOP_LEVEL})
option(HYBRIDRT_BUILD_BENCHMARKS "Build benchmarks (needs OpenCV imgcodecs)" OFF)
option(HYBRIDRT_BUILD_EXAMPLES "Build examples (needs OpenCV imgcodecs)" ${HYBRIDRT_IS_TOP_LEVEL})
option(HYBRIDRT_BUILD_FLATC "Build flatc and the hrtc_schema regeneration target" OFF)
option(HYBRIDRT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(HYBRIDRT_WITH_CUDA "Enable CUDA memory and image operators without TensorRT" OFF)
option(HYBRIDRT_RGA3_ONLY "Schedule RGA work on RGA3 cores only" OFF)
