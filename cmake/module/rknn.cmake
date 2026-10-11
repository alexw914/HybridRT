message(STATUS "--------------rknn--------------")


if (ANDROID)
  message(FATAL_ERROR "current rknn only supported in Linux")
endif ()

if (NOT DEFINED RKNN_ROOT)
  message(FATAL_ERROR "Canot get definition for 'RKNN_ROOT'")
endif ()

if (NOT EXISTS ${RKNN_ROOT})
  message(FATAL_ERROR "Cannot find directory ${RKNN_ROOT}")
endif ()

if (NOT EXISTS ${RKNN_ROOT}/runtime)
  message(FATAL_ERROR "Cannot find directory ${RKNN_ROOT}/runtime")
endif ()

if (NOT DEFINED RKNN_PLATFORM)
  set(RKNN_PLATFORM "")
endif ()

if (NOT EXISTS ${RKNN_ROOT}/runtime/${RKNN_PLATFORM}/Linux/librknn_api/include)
  message(FATAL_ERROR "Cannot find directory ${RKNN_ROOT}/runtime/${RKNN_PLATFORM}/Linux/librknn_api/include")
endif ()

set(rknn_lib_dir ${RKNN_ROOT}/runtime/${RKNN_PLATFORM}/Linux/librknn_api/aarch64)

if (NOT EXISTS ${RKNN_ROOT}/runtime/${RKNN_PLATFORM}/Linux/librknn_api/aarch64)
  message(FATAL_ERROR "Cannot find lib directory ${RKNN_ROOT}/runtime/${RKNN_PLATFORM}/Linux/librknn_api/aarch64")
endif ()

execute_process(
    COMMAND "sh" "-c" "strings ${rknn_lib_dir}/librknnrt.so | grep 'librknnrt version'"
    OUTPUT_VARIABLE RKNN_VERSION
    RESULT_VARIABLE result
)

if (result EQUAL 0)
  string(STRIP ${RKNN_VERSION} RKNN_VERSION)
  message(STATUS "Found ${RKNN_VERSION} in ${RKNN_ROOT}")
else ()
  message(WARNING "Failed to get rknn version.")
endif ()

add_library(rknn SHARED IMPORTED)
target_link_directories(rknn INTERFACE ${rknn_lib_dir})
set_target_properties(rknn PROPERTIES
    IMPORTED_LOCATION ${rknn_lib_dir}/librknnrt.so
    INTERFACE_INCLUDE_DIRECTORIES ${RKNN_ROOT}/runtime/${RKNN_PLATFORM}/Linux/librknn_api/include
)


set(RKNN_FOUND ON)
