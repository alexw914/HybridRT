# ---------------------------------------------------------------------------
# Python package: ctypes over hybridrt_c
# ---------------------------------------------------------------------------
# scikit-build-core (pyproject.toml) installs the "python" component:
# hybridrt_c and the backend runtimes it links, into the package. Tests load
# the built library through HYBRIDRT_LIBRARY.
# ---------------------------------------------------------------------------

if(SKBUILD)
    if(APPLE)
        set_target_properties(hybridrt_c PROPERTIES INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(hybridrt_c PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
    install(TARGETS hybridrt_c DESTINATION hybridrt COMPONENT python)
    # Backend SDK libraries ship beside it, each once under the name the loader
    # asks for, as wheels hold no symlinks; system libraries are left to the
    # environment or a wheel repair tool.
    install(CODE [[
        file(GET_RUNTIME_DEPENDENCIES
             RESOLVED_DEPENDENCIES_VAR hybridrt_runtime
             LIBRARIES "$<TARGET_FILE:hybridrt_c>"
             DIRECTORIES "$<TARGET_FILE_DIR:hybridrt_c>"
             PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-"
             POST_EXCLUDE_REGEXES "^/lib" "^/usr/" "^/System/" "[Ss]ystem32")
        foreach(dependency IN LISTS hybridrt_runtime)
            get_filename_component(name "${dependency}" NAME)
            file(REAL_PATH "${dependency}" resolved)
            file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/hybridrt" TYPE FILE FILES "${resolved}" RENAME "${name}")
        endforeach()
    ]] COMPONENT python)
endif()

if(HYBRIDRT_BUILD_TESTS)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, pytest"
                        RESULT_VARIABLE python_test_deps OUTPUT_QUIET ERROR_QUIET)
    endif()
    if(Python3_Interpreter_FOUND AND python_test_deps EQUAL 0)
        add_test(NAME hybridrt_test_python
                 COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        set_tests_properties(hybridrt_test_python PROPERTIES ENVIRONMENT_MODIFICATION
            "HYBRIDRT_LIBRARY=set:$<TARGET_FILE:hybridrt_c>;PYTHONPATH=path_list_prepend:${CMAKE_CURRENT_SOURCE_DIR}")
    else()
        message(STATUS "Python tests skipped: needs a Python 3 interpreter with numpy and pytest")
    endif()
endif()
