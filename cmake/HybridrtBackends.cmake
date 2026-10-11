# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------
# A backend is enabled by setting its SDK root, e.g. -DORT_ROOT=<path>.
# cmake/module/<sdk>.cmake imports the SDK targets; this file sets
# HRT_USE_<BACKEND>, which source/CMakeLists.txt turns into sources and the
# matching compile definition.
# ---------------------------------------------------------------------------

set(_module_dir "${CMAKE_CURRENT_LIST_DIR}/module")

if(ORT_ROOT)
    include(${_module_dir}/onnxruntime.cmake)
    set(HRT_USE_ORT ON)
endif()

if(ncnn_ROOT)
    include(${_module_dir}/ncnn.cmake)
    set(HRT_USE_NCNN ON)
endif()

if(MNN_ROOT)
    include(${_module_dir}/MNN.cmake)
    set(HRT_USE_MNN ON)
endif()

if(RKNN_ROOT AND RGA_ROOT)
    include(${_module_dir}/rknn.cmake)
    include(${_module_dir}/rga.cmake)
    set(HRT_USE_RKNN ON)
    set(HRT_USE_RGA ON)
elseif(RKNN_ROOT OR RGA_ROOT)
    message(FATAL_ERROR "RKNN_ROOT and RGA_ROOT must be set together")
endif()

if(CANN_ROOT)
    include(${_module_dir}/cann.cmake)
    set(HRT_USE_CANN ON)
endif()

if(TensorRT_ROOT OR HYBRIDRT_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(HRT_USE_CUDA ON)
endif()

if(TensorRT_ROOT)
    include(${_module_dir}/TensorRT.cmake)
    set(HRT_USE_TENSORRT ON)
endif()

unset(_module_dir)
