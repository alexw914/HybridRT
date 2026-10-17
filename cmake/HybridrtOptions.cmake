# ---------------------------------------------------------------------------
# HybridRT build options
# ---------------------------------------------------------------------------
# Inference backends are enabled by pointing <SDK>_ROOT at an installed SDK,
# see HybridrtBackends.cmake.
# ---------------------------------------------------------------------------

option(HYBRIDRT_BUILD_TOOLS "Build the hrt-compress packaging tool" ${HYBRIDRT_IS_TOP_LEVEL})
option(HYBRIDRT_BUILD_TESTS "Build test suite" OFF)
option(HYBRIDRT_BUILD_BENCHMARKS "Build benchmarks (needs OpenCV imgcodecs)" OFF)
option(HYBRIDRT_BUILD_EXAMPLES "Build examples (needs OpenCV imgcodecs)" ${HYBRIDRT_IS_TOP_LEVEL})
option(HYBRIDRT_BUILD_C_API "Build the hybridrt_c shared library with the C API" OFF)
option(HYBRIDRT_BUILD_PYTHON "Build the C API for the Python package and register its tests" OFF)
option(HYBRIDRT_BUILD_FLATC "Build flatc and the hrtc_schema regeneration target" OFF)
option(HYBRIDRT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(HYBRIDRT_WITH_CUDA "Enable CUDA memory and image operators without TensorRT" OFF)
option(HYBRIDRT_RGA3_ONLY "Schedule RGA work on RGA3 cores only" OFF)

if(HYBRIDRT_BUILD_PYTHON)
    set(HYBRIDRT_BUILD_C_API ON)
endif()

# The C API library links static dependencies into a shared library that
# other processes dlopen: they need position-independent code, must not bring
# kitzoo's process-wide mimalloc override, which only executables may carry,
# and mimalloc's thread-locals must not use the static TLS block.
if(HYBRIDRT_BUILD_C_API)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(KITZOO_MIMALLOC_OVERRIDE OFF)
    set(MI_TLS_MODEL "LOCAL_DYNAMIC" CACHE STRING "mimalloc TLS model; dlopen-compatible for hybridrt_c" FORCE)
endif()
