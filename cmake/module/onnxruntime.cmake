message(STATUS "--------------onnxruntime--------------")

if (NOT DEFINED ORT_ROOT)
  message(FATAL_ERROR "Please set ORT_ROOT")
endif ()

if (NOT EXISTS ${ORT_ROOT}/include/onnxruntime_c_api.h)
  message(FATAL_ERROR "Cannot found file ${ORT_ROOT}/include/onnxruntime_c_api.h")
endif ()

file(READ ${ORT_ROOT}/include/onnxruntime_c_api.h ORT_C_API_H)
string(REGEX MATCH "ORT_API_VERSION ([0-9]+)" _ ${ORT_C_API_H})
set(ORT_API_VERSION "${CMAKE_MATCH_1}")
if (NOT ORT_API_VERSION)
  set(ORT_API_VERSION "unknown")
endif ()

find_library(ORT_LIBRARY NAMES onnxruntime HINTS ${ORT_ROOT} PATH_SUFFIXES lib NO_DEFAULT_PATH)
if (NOT ORT_LIBRARY)
  message(FATAL_ERROR "Cannot found onnxruntime library in ${ORT_ROOT}/lib")
endif ()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ORT_LIBRARY}
    IMPORTED_IMPLIB ${ORT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ORT_ROOT}/include
)

if(WIN32)
  find_file(ORT_RUNTIME NAMES onnxruntime.dll HINTS "${ORT_ROOT}" PATH_SUFFIXES lib bin NO_DEFAULT_PATH)
  if(NOT ORT_RUNTIME)
    message(FATAL_ERROR "Cannot find onnxruntime.dll in ${ORT_ROOT}/lib or bin")
  endif()
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ORT_RUNTIME}")
endif()

message(STATUS "Found onnxruntime API Version ${ORT_API_VERSION}: ${ORT_LIBRARY}")

set(ORT_FOUND ON)
