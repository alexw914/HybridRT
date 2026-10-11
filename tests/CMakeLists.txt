# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
# hybridrt_add_unit_test(name SOURCES ... [DEPENDS ...])
#
# Creates hybridrt_test_<name> linked against GTest::gtest_main and the given
# targets, registered with CTest via gtest_discover_tests.
# ---------------------------------------------------------------------------

include(GoogleTest)

if(HYBRIDRT_BUILD_TOOLS)
    add_test(NAME HrtCompressTest.ExtractsOnlySafeNames
        COMMAND ${CMAKE_COMMAND}
            -DHRT_COMPRESS=$<TARGET_FILE:hrt_compress>
            -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/hrt_compress_test
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/hrt_compress_test.cmake)
endif()

function(hybridrt_add_unit_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})

    set(test_name hybridrt_test_${name})
    add_executable(${test_name} ${ARG_SOURCES})
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main ${ARG_DEPENDS})
    target_compile_definitions(${test_name} PRIVATE HRT_TEMPLATE_DIR="${PROJECT_SOURCE_DIR}/assets/template")
    hybridrt_apply_warnings(${test_name})
    gtest_discover_tests(${test_name})
endfunction()

hybridrt_add_unit_test(hrtc SOURCES hrtc/hrtc_test.cpp DEPENDS hrt::hrtc kitzoo::os)
hybridrt_add_unit_test(memory SOURCES core/memory_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(image_preprocessor SOURCES preprocess/image_preprocessor_test.cpp DEPENDS hrt::hybridrt)

if(HRT_USE_ORT)
    hybridrt_add_unit_test(onnxruntime_engine SOURCES engine/onnxruntime_engine_test.cpp DEPENDS hrt::hybridrt)
endif()

hybridrt_add_unit_test(detect SOURCES postprocess/detect_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(rotated SOURCES postprocess/rotated_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(general_classify SOURCES postprocess/general_classify_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(general_keypoint SOURCES postprocess/general_keypoint_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(yolo_keypoint SOURCES postprocess/yolo_keypoint_test.cpp DEPENDS hrt::hybridrt)

if(HRT_USE_NCNN)
    hybridrt_add_unit_test(ncnn_engine SOURCES engine/ncnn_engine_test.cpp DEPENDS hrt::hybridrt)
endif()

if(HRT_USE_MNN)
    hybridrt_add_unit_test(mnn_engine SOURCES engine/mnn_engine_test.cpp DEPENDS hrt::hybridrt kitzoo::os MNN)
endif()

if(HRT_USE_ORT)
    hybridrt_add_unit_test(image_model SOURCES model/image_model_test.cpp DEPENDS hrt::hybridrt kitzoo::os)
endif()
hybridrt_add_unit_test(segment SOURCES postprocess/segment_test.cpp DEPENDS hrt::hybridrt)

hybridrt_add_unit_test(view SOURCES core/view_test.cpp DEPENDS hrt::hybridrt)
