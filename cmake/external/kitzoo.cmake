# kitzoo: logging, JSON, CLI, memory, threading and OS utilities.
# Use a local checkout with -DFETCHCONTENT_SOURCE_DIR_KITZOO=<path>.

include(FetchContent)

set(HYBRIDRT_KITZOO_GIT_TAG "v0.5.1" CACHE STRING "kitzoo git tag or commit")

FetchContent_Declare(
    kitzoo
    GIT_REPOSITORY https://github.com/alexw914/kitzoo.git
    GIT_TAG ${HYBRIDRT_KITZOO_GIT_TAG}
    GIT_SHALLOW TRUE
    SYSTEM
    FIND_PACKAGE_ARGS CONFIG)
FetchContent_MakeAvailable(kitzoo)
