# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: cmake/module/OpenVINO.cmake
# Description: Finds the OpenVINO Runtime SDK and imports its C API target.
# -----------------------------------------------------------------------------

if(NOT DEFINED OpenVINO_ROOT)
  message(FATAL_ERROR "Please set OpenVINO_ROOT")
endif()

find_package(OpenVINO 2026.4.1 CONFIG REQUIRED PATHS "${OpenVINO_ROOT}/runtime/cmake" NO_DEFAULT_PATH)

if(NOT TARGET openvino::runtime::c)
  message(FATAL_ERROR "OpenVINO Runtime C API target is missing in ${OpenVINO_ROOT}")
endif()

message(STATUS "Found OpenVINO Runtime: ${OpenVINO_ROOT}")
