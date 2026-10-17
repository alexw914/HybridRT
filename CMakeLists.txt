cmake_minimum_required(VERSION 3.25)

project(
    hybridrt
    VERSION 0.1.0
    DESCRIPTION "A C++20 model inference framework"
    LANGUAGES CXX)

set(HYBRIDRT_IS_TOP_LEVEL ${PROJECT_IS_TOP_LEVEL})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Deduplicate static libraries on link lines; Apple ld warns on repeated archives.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
    set(CMAKE_LINK_LIBRARIES_STRATEGY REORDER_FREELY)
endif()

if(HYBRIDRT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(HybridrtOptions)
include(HybridrtWarnings)
include(HybridrtRuntime)

# -- Dependencies --------------------------------------------------------------
include(external/kitzoo)
include(external/flatbuffers)
include(external/opencv)
if(HYBRIDRT_BUILD_TESTS)
    include(external/googletest)
endif()
if(HYBRIDRT_BUILD_BENCHMARKS)
    include(external/benchmark)
endif()
include(HybridrtBackends)

# -- Targets -------------------------------------------------------------------
add_subdirectory(source)

if(HYBRIDRT_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(HYBRIDRT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(HYBRIDRT_BUILD_TESTS OR HYBRIDRT_BUILD_BENCHMARKS)
    enable_testing()
endif()

if(HYBRIDRT_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

if(HYBRIDRT_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(HYBRIDRT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(HYBRIDRT_BUILD_FLATC)
    add_custom_target(
        hrtc_schema
        COMMAND flatc --cpp -o ${PROJECT_SOURCE_DIR}/source/include/hrtc
                ${PROJECT_SOURCE_DIR}/fbs/hrtc.fbs
        DEPENDS flatc
        COMMENT "Regenerating source/include/hrtc/hrtc_generated.h")
endif()
