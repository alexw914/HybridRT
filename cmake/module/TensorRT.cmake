message(STATUS "--------------TensorRT--------------")

include(CheckSymbolExists)

find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${TensorRT_ROOT} PATH_SUFFIXES include)

if (NOT TensorRT_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot found TensorRT include dir")
endif ()

if (EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_VERSION_FILE_CONTENTS)

  check_symbol_exists(TRT_MAJOR_ENTERPRISE "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TRT_MAJOR_ENTERPRISE_FOUND)

  if (TRT_MAJOR_ENTERPRISE_FOUND)
    string(REGEX MATCH "define TRT_MAJOR_ENTERPRISE * +([0-9]+)"
        TensorRT_VERSION_MAJOR "${TensorRT_VERSION_FILE_CONTENTS}")
    set(TensorRT_VERSION_MAJOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define TRT_MINOR_ENTERPRISE * +([0-9]+)"
        TensorRT_VERSION_MINOR "${TensorRT_VERSION_FILE_CONTENTS}")
    set(TensorRT_VERSION_MINOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define TRT_PATCH_ENTERPRISE * +([0-9]+)"
        TensorRT_VERSION_PATCH "${TensorRT_VERSION_FILE_CONTENTS}")
    set(TensorRT_VERSION_PATCH "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define TRT_BUILD_ENTERPRISE * +([0-9]+)"
        TensorRT_VERSION_BUILD "${TensorRT_VERSION_FILE_CONTENTS}")
    set(TensorRT_VERSION_BUILD "${CMAKE_MATCH_1}")
  else ()
    string(REGEX MATCH "define NV_TENSORRT_MAJOR * +([0-9]+)"
        TensorRT_VERSION_MAJOR "${TensorRT_VERSION_FILE_CONTENTS}")
    set(TensorRT_VERSION_MAJOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define NV_TENSORRT_MINOR * +([0-9]+)"
        TensorRT_VERSION_MINOR "${TensorRT_VERSION_FILE_CONTENTS}")
    set(TensorRT_VERSION_MINOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define NV_TENSORRT_PATCH * +([0-9]+)"
        TensorRT_VERSION_PATCH "${TensorRT_VERSION_FILE_CONTENTS}")
    set(TensorRT_VERSION_PATCH "${CMAKE_MATCH_1}")
    string(REGEX MATCH "define NV_TENSORRT_BUILD * +([0-9]+)"
        TensorRT_VERSION_BUILD "${TensorRT_VERSION_FILE_CONTENTS}")
    set(TensorRT_VERSION_BUILD "${CMAKE_MATCH_1}")
  endif ()

  if (NOT TensorRT_VERSION_MAJOR)
    message(FATAL_ERROR "Cannot found TensorRT Version")
  endif ()

  set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}.${TensorRT_VERSION_BUILD}")

  math(EXPR HRT_TENSORRT_VERSION "${TensorRT_VERSION_MAJOR} * 10000 + ${TensorRT_VERSION_MINOR} * 100 + ${TensorRT_VERSION_PATCH}")

  message(STATUS "Found TensorRT version ${TensorRT_VERSION}(${HRT_TENSORRT_VERSION}) in ${TensorRT_ROOT}")

else ()
  message(FATAL_ERROR "Cannot found version file for TensorRT")
endif ()

list(APPEND TRT_LIBS_NAME nvinfer_plugin)
list(APPEND TRT_LIBS_NAME nvinfer)

foreach (trt_name ${TRT_LIBS_NAME})
  find_library(${trt_name}_lib NAMES ${trt_name} HINTS ${TensorRT_ROOT} PATH_SUFFIXES lib)
  if (NOT ${trt_name}_lib)
    set(TensorRT_FOUND OFF)
    message(FATAL_ERROR "Cannot found ${${trt_name}_lib} needed by TensorRT")
  endif ()

  add_library(TensorRT::${trt_name} SHARED IMPORTED)
  set_target_properties(TensorRT::${trt_name} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${TensorRT_INCLUDE_DIR}
      IMPORTED_LOCATION ${${trt_name}_lib}
  )
endforeach ()

set(TensorRT_FOUND ON)