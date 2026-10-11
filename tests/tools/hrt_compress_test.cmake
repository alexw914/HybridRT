# -----------------------------------------------------------------------------
# HybridRT | C++20 Model Inference Framework
# File: tests/tools/hrt_compress_test.cmake
# Description: Tests CLI extraction with safe and unsafe algorithm names.
# -----------------------------------------------------------------------------

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")

foreach(name IN ITEMS "detect-v1.0" "../escaped" "nested/escaped" "nested\\escaped"
                      "/absolute" "C:escaped" "." "..")
    set(manifest "${TEST_DIR}/manifest.json")
    string(REPLACE "\\" "\\\\" json_name "${name}")
    file(WRITE "${manifest}" "{\"name\":\"test\",\"version\":\"1\",\"algos\":[{\"name\":\"${json_name}\",\"config\":{\"engine\":{\"backends\":[{\"engine_type\":\"ort\"}]}}}]}")
    execute_process(COMMAND "${HRT_COMPRESS}" -j "${manifest}" -o "${TEST_DIR}/test.hrtc"
                    RESULT_VARIABLE packed OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT packed EQUAL 0)
        message(FATAL_ERROR "Cannot pack fixture ${name}: ${output}${error}")
    endif()
    file(REMOVE_RECURSE "${TEST_DIR}/out")
    execute_process(COMMAND "${HRT_COMPRESS}" -i "${TEST_DIR}/test.hrtc" -x "${TEST_DIR}/out"
                    RESULT_VARIABLE extracted OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(name STREQUAL "detect-v1.0")
        if(NOT extracted EQUAL 0 OR NOT EXISTS "${TEST_DIR}/out/detect-v1.0.config")
            message(FATAL_ERROR "Safe name failed: ${output}${error}")
        endif()
        file(WRITE "${TEST_DIR}/outside.config" "unchanged")
        file(REMOVE "${TEST_DIR}/out/detect-v1.0.config")
        file(CREATE_LINK "${TEST_DIR}/outside.config" "${TEST_DIR}/out/detect-v1.0.config" SYMBOLIC)
        execute_process(COMMAND "${HRT_COMPRESS}" -i "${TEST_DIR}/test.hrtc" -x "${TEST_DIR}/out"
                        RESULT_VARIABLE linked OUTPUT_VARIABLE output ERROR_VARIABLE error)
        file(READ "${TEST_DIR}/outside.config" content)
        if(linked EQUAL 0 OR NOT content STREQUAL "unchanged")
            message(FATAL_ERROR "Extraction followed a symbolic link: ${output}${error}")
        endif()
    elseif(extracted EQUAL 0 OR EXISTS "${TEST_DIR}/out")
        message(FATAL_ERROR "Unsafe name was not rejected before writing: ${name}")
    endif()
endforeach()

file(REMOVE_RECURSE "${TEST_DIR}")
