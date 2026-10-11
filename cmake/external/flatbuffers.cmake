# flatbuffers: hrtc package metadata.
# Use a local checkout with -DFETCHCONTENT_SOURCE_DIR_FLATBUFFERS=<path>.

include(FetchContent)

set(HYBRIDRT_FLATBUFFERS_GIT_TAG "v25.2.10" CACHE STRING "flatbuffers git tag or commit")

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ${HYBRIDRT_BUILD_FLATC} CACHE BOOL "" FORCE)

FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG ${HYBRIDRT_FLATBUFFERS_GIT_TAG}
    GIT_SHALLOW TRUE
    SYSTEM)
FetchContent_MakeAvailable(flatbuffers)
