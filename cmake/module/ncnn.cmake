message(STATUS "--------------ncnn--------------")

if (NOT DEFINED ncnn_ROOT)
  message(FATAL_ERROR "Please set ncnn_ROOT")
endif ()

if (NOT EXISTS ${ncnn_ROOT})
  message(FATAL_ERROR "Cannot found directory ${ncnn_ROOT}")
endif ()
if (NOT EXISTS ${ncnn_ROOT}/include)
  message(FATAL_ERROR "Cannot found directory ${ncnn_ROOT}/include")
endif ()

if (NOT EXISTS ${ncnn_ROOT}/include/ncnn/platform.h)
  message(FATAL_ERROR "Cannot found file ${ncnn_ROOT}/include/ncnn/platform.h")
endif ()

file(READ ${ncnn_ROOT}/include/ncnn/platform.h NCNN_PLATFORM_H)
# read ncnn version, format is NCNN_VERSION_STRING "1.0.20230023"
string(REGEX MATCH "NCNN_VERSION_STRING \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" NCNN_VERSION_STRING ${NCNN_PLATFORM_H})
string(REGEX REPLACE "NCNN_VERSION_STRING \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" "\\1" NCNN_VERSION ${NCNN_VERSION_STRING})
if (NOT NCNN_VERSION)
  set(NCNN_VERSION "unknown")
endif ()

set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${ncnn_ROOT})
find_package(ncnn REQUIRED PATHS ${ncnn_ROOT})

if (NOT ncnn_FOUND)
  message(FATAL_ERROR "Cannot found ncnn")
endif ()

message(STATUS "Found ncnn Version ${NCNN_VERSION} in ${ncnn_ROOT}")