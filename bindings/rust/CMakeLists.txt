# ---------------------------------------------------------------------------
# Rust crates: hybridrt-sys (FFI) and hybridrt (safe API) over hybridrt_c
# ---------------------------------------------------------------------------
# Tests run cargo against the built library through HYBRIDRT_LIB_DIR; the
# first run fetches the crates' dependencies.
# ---------------------------------------------------------------------------

if(HYBRIDRT_BUILD_TESTS)
    find_program(CARGO_EXECUTABLE cargo)
    if(CARGO_EXECUTABLE)
        add_test(NAME hybridrt_test_rust
                 COMMAND ${CARGO_EXECUTABLE} test --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml)
        set_tests_properties(hybridrt_test_rust PROPERTIES TIMEOUT 1800 ENVIRONMENT_MODIFICATION
            "HYBRIDRT_LIB_DIR=set:$<TARGET_FILE_DIR:hybridrt_c>;CARGO_TARGET_DIR=set:${CMAKE_CURRENT_BINARY_DIR}/target;PATH=path_list_prepend:$<TARGET_FILE_DIR:hybridrt_c>")
    else()
        message(STATUS "Rust tests skipped: cargo not found")
    endif()
endif()
