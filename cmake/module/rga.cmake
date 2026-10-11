message(STATUS "--------------rga--------------")

if (NOT DEFINED RGA_ROOT)
  message(FATAL_ERROR "Canot get definition for 'RGA_ROOT'")
endif ()

if (NOT EXISTS ${RGA_ROOT})
  message(FATAL_ERROR "Cannot find directory ${RGA_ROOT}")
endif ()

if (NOT EXISTS ${RGA_ROOT}/include)
  message(FATAL_ERROR "Cannot find directory ${RGA_ROOT}/include")
endif ()

if (ANDROID)
  set(rga_lib_dir ${RGA_ROOT}/libs/AndroidNdk/${ANDROID_ABI})
elseif (UNIX AND NOT APPLE)
  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(rga_lib_dir ${RGA_ROOT}/libs/Linux/gcc-aarch64)
  elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
    set(rga_lib_dir ${RGA_ROOT}/libs/Linux/gcc-armhf)
  else ()
    message(FATAL_ERROR "Unsupported processor ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()
else ()
  message(FATAL_ERROR "Unsupported platform ${CMAKE_SYSTEM_NAME}")
endif ()

if (NOT EXISTS ${rga_lib_dir})
  message(FATAL_ERROR "Cannot find directory ${rga_lib_dir}")
endif ()

if (NOT EXISTS ${RGA_ROOT}/include/im2d_version.h)
  message(FATAL_ERROR "Cannot found file ${RGA_ROOT}/include/im2d_version.h")
endif ()

file(READ ${RGA_ROOT}/include/im2d_version.h RGA_IM2D_VERSION_H_CONTENTS)
string(REGEX MATCH "RGA_API_MAJOR_VERSION +([0-9]+)" RGA_API_MAJOR_VERSION ${RGA_IM2D_VERSION_H_CONTENTS})
string(REGEX REPLACE "RGA_API_MAJOR_VERSION +([0-9]+)" "\\1" RGA_API_MAJOR_VERSION ${RGA_API_MAJOR_VERSION})
string(REGEX MATCH "RGA_API_MINOR_VERSION +([0-9]+)" RGA_API_MINOR_VERSION ${RGA_IM2D_VERSION_H_CONTENTS})
string(REGEX REPLACE "RGA_API_MINOR_VERSION +([0-9]+)" "\\1" RGA_API_MINOR_VERSION ${RGA_API_MINOR_VERSION})
string(REGEX MATCH "RGA_API_REVISION_VERSION +([0-9]+)" RGA_API_REVISION_VERSION ${RGA_IM2D_VERSION_H_CONTENTS})
string(REGEX REPLACE "RGA_API_REVISION_VERSION +([0-9]+)" "\\1" RGA_API_REVISION_VERSION ${RGA_API_REVISION_VERSION})
string(REGEX MATCH "RGA_API_BUILD_VERSION +([0-9]+)" RGA_API_BUILD_VERSION ${RGA_IM2D_VERSION_H_CONTENTS})
string(REGEX REPLACE "RGA_API_BUILD_VERSION +([0-9]+)" "\\1" RGA_API_BUILD_VERSION ${RGA_API_BUILD_VERSION})
if (NOT RGA_API_MAJOR_VERSION)
  set(RGA_VERSION "unknown")
else ()
  set(RGA_VERSION "${RGA_API_MAJOR_VERSION}.${RGA_API_MINOR_VERSION}.${RGA_API_REVISION_VERSION}_[${RGA_API_BUILD_VERSION}]")
endif ()


add_library(rga SHARED IMPORTED)
target_link_directories(rga INTERFACE ${rga_lib_dir})
set_target_properties(rga PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${RGA_ROOT}/include
    IMPORTED_LOCATION ${rga_lib_dir}/librga.so
)

message(STATUS "Found RGA Version ${RGA_VERSION} in ${RGA_ROOT}")

set(rga_FOUND ON)
