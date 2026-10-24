# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
# hybridrt_add_unit_test(name SOURCES ... [DEPENDS ...])
#
# Creates hybridrt_test_<name> linked against GTest::gtest_main and the given
# targets, registered with CTest via gtest_discover_tests.
# ---------------------------------------------------------------------------

include(GoogleTest)

function(hybridrt_add_unit_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})

    set(test_name hybridrt_test_${name})
    add_executable(${test_name} ${ARG_SOURCES})
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main ${ARG_DEPENDS})
    if(CMAKE_CROSSCOMPILING)
        target_compile_definitions(${test_name} PRIVATE HRT_TEMPLATE_DIR="./assets/template")
    else()
        target_compile_definitions(${test_name} PRIVATE HRT_TEMPLATE_DIR="${PROJECT_SOURCE_DIR}/assets/template")
    endif()
    hybridrt_apply_warnings(${test_name})
    hybridrt_deploy_runtime(${test_name})
    # Cross-compiled tests cannot run on the build host, so list them when CTest runs on the target.
    if(CMAKE_CROSSCOMPILING)
        gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 30 DISCOVERY_MODE PRE_TEST)
    else()
        gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 30)
    endif()
endfunction()

# hybridrt_enable_model_test(name)
#
# With ONNX Runtime and OpenCV imgcodecs, defines HRT_ORT_MODEL_TEST so the test
# runs template algos on the local fixtures under assets/ (skipped when absent).
function(hybridrt_enable_model_test name)
    if(HRT_USE_ORT AND TARGET opencv_imgcodecs)
        target_link_libraries(hybridrt_test_${name} PRIVATE opencv_imgcodecs)
        target_compile_definitions(hybridrt_test_${name} PRIVATE HRT_ORT_MODEL_TEST)
    endif()
endfunction()

# hybridrt_add_program_test(name PROGRAM <target> DEFINE <macro> SOURCES ... [DEPENDS ...])
#
# A black-box test of a command-line program: <macro> holds the path of the
# built <target>, which is built before the test.
function(hybridrt_add_program_test name)
    cmake_parse_arguments(ARG "" "PROGRAM;DEFINE" "SOURCES;DEPENDS" ${ARGN})

    hybridrt_add_unit_test(${name} SOURCES ${ARG_SOURCES} DEPENDS kitzoo::os ${ARG_DEPENDS})
    target_compile_definitions(hybridrt_test_${name} PRIVATE ${ARG_DEFINE}="$<TARGET_FILE:${ARG_PROGRAM}>")
    add_dependencies(hybridrt_test_${name} ${ARG_PROGRAM})
endfunction()

if(TARGET hrt_compress)
    hybridrt_add_program_test(hrt_compress PROGRAM hrt_compress DEFINE HRT_COMPRESS
        SOURCES tools/hrt_compress_test.cpp DEPENDS hrt::hrtc)
endif()
if(TARGET hrt_model_bench AND HRT_USE_ORT)
    hybridrt_add_program_test(hrt_model_bench PROGRAM hrt_model_bench DEFINE HRT_MODEL_BENCH
        SOURCES tools/hrt_model_bench_test.cpp DEPENDS hrt::hrtc)
endif()
if(TARGET hrt_validate AND HRT_USE_ORT)
    hybridrt_add_program_test(hrt_validate PROGRAM hrt_validate DEFINE HRT_VALIDATE
        SOURCES tools/hrt_validate_test.cpp DEPENDS hrt::hrtc opencv_imgcodecs)
endif()
if(TARGET hrt_ocr AND HRT_USE_ORT)
    hybridrt_add_program_test(hrt_ocr PROGRAM hrt_ocr DEFINE HRT_OCR SOURCES tools/hrt_ocr_test.cpp DEPENDS hrt::hrtc)
endif()
if(TARGET hybridrt_example_image_model AND HRT_USE_ORT)
    hybridrt_add_program_test(example_image_model PROGRAM hybridrt_example_image_model
        DEFINE HRT_EXAMPLE_IMAGE_MODEL SOURCES examples/image_model_test.cpp DEPENDS hrt::hrtc opencv_imgcodecs)
endif()
if(TARGET hybridrt_example_ocr_model AND HRT_USE_ORT)
    hybridrt_add_program_test(example_ocr_model PROGRAM hybridrt_example_ocr_model
        DEFINE HRT_EXAMPLE_OCR_MODEL SOURCES examples/ocr_model_test.cpp DEPENDS hrt::hrtc)
endif()

if(TARGET hybridrt_c)
    hybridrt_add_unit_test(hrt SOURCES capi/hrt_test.cpp DEPENDS hrt::hybridrt_c hrt::hrtc kitzoo::os)
    hybridrt_enable_model_test(hrt)
    if(HRT_USE_ORT)
        target_compile_definitions(hybridrt_test_hrt PRIVATE HRT_TEST_ORT)
    endif()
endif()

hybridrt_add_unit_test(hrtc SOURCES hrtc/hrtc_test.cpp DEPENDS hrt::hrtc kitzoo::os)
hybridrt_add_unit_test(define SOURCES core/define_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(memory SOURCES core/memory_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(image_preprocessor SOURCES preprocess/image_preprocessor_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(postprocessor_factory SOURCES postprocess/postprocessor_factory_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(ocr SOURCES postprocess/ocr_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(ocr_model SOURCES model/ocr_model_test.cpp DEPENDS hrt::hybridrt)
hybridrt_enable_model_test(ocr_model)

if(HRT_USE_CUDA)
    hybridrt_add_unit_test(image SOURCES core/image_test.cpp DEPENDS hrt::hybridrt CUDA::cudart)
endif()

if(HRT_USE_ORT)
    hybridrt_add_unit_test(onnxruntime_engine SOURCES engine/onnxruntime_engine_test.cpp DEPENDS hrt::hybridrt)
endif()

if(HRT_USE_TENSORRT)
    hybridrt_add_unit_test(tensorrt_engine SOURCES engine/tensorrt_engine_test.cpp DEPENDS hrt::hybridrt CUDA::cudart)
endif()

hybridrt_add_unit_test(detect SOURCES postprocess/detect_test.cpp DEPENDS hrt::hybridrt)
hybridrt_enable_model_test(detect)
hybridrt_add_unit_test(detr_detect SOURCES postprocess/detr_detect_test.cpp DEPENDS hrt::hybridrt)
hybridrt_enable_model_test(detr_detect)
hybridrt_add_unit_test(e2e_detect SOURCES postprocess/e2e_detect_test.cpp DEPENDS hrt::hybridrt)
hybridrt_enable_model_test(e2e_detect)
hybridrt_add_unit_test(rotated SOURCES postprocess/rotated_test.cpp DEPENDS hrt::hybridrt)
target_include_directories(hybridrt_test_rotated PRIVATE ${PROJECT_SOURCE_DIR}/source)
hybridrt_enable_model_test(rotated)
hybridrt_add_unit_test(general_classify SOURCES postprocess/general_classify_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(embedding SOURCES postprocess/embedding_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(general_keypoint SOURCES postprocess/general_keypoint_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(e2e_keypoint SOURCES postprocess/e2e_keypoint_test.cpp DEPENDS hrt::hybridrt)
hybridrt_enable_model_test(e2e_keypoint)
hybridrt_add_unit_test(rtmo_keypoint SOURCES postprocess/rtmo_keypoint_test.cpp DEPENDS hrt::hybridrt)
hybridrt_enable_model_test(rtmo_keypoint)
hybridrt_add_unit_test(general_yolo_pose SOURCES postprocess/general_yolo_pose_test.cpp DEPENDS hrt::hybridrt)
hybridrt_add_unit_test(yolov8_keypoint SOURCES postprocess/yolov8_keypoint_test.cpp DEPENDS hrt::hybridrt)
hybridrt_enable_model_test(yolov8_keypoint)
hybridrt_add_unit_test(yolo26_keypoint SOURCES postprocess/yolo26_keypoint_test.cpp DEPENDS hrt::hybridrt)
hybridrt_enable_model_test(yolo26_keypoint)

if(HRT_USE_NCNN)
    hybridrt_add_unit_test(ncnn_engine SOURCES engine/ncnn_engine_test.cpp DEPENDS hrt::hybridrt)
endif()

if(HRT_USE_MNN)
    hybridrt_add_unit_test(mnn_engine SOURCES engine/mnn_engine_test.cpp DEPENDS hrt::hybridrt kitzoo::os)
    # hybridrt already links MNN (whole-archive when static); linking it again plainly is rejected.
    target_include_directories(hybridrt_test_mnn_engine SYSTEM PRIVATE
        $<TARGET_PROPERTY:MNN,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

if(HRT_USE_ORT OR HRT_USE_TENSORRT OR HRT_USE_RKNN)
    hybridrt_add_unit_test(image_model SOURCES model/image_model_test.cpp DEPENDS hrt::hybridrt kitzoo::log)
    if(HRT_USE_ORT)
        target_link_libraries(hybridrt_test_image_model PRIVATE kitzoo::os)
        target_compile_definitions(hybridrt_test_image_model PRIVATE HRT_TEST_ORT)
    endif()
    hybridrt_enable_model_test(image_model)
    if(HRT_USE_RKNN AND TARGET opencv_imgcodecs)
        target_link_libraries(hybridrt_test_image_model PRIVATE opencv_imgcodecs kitzoo::os rknn)
        target_compile_definitions(hybridrt_test_image_model PRIVATE HRT_RKNN_MODEL_TEST)
    endif()
    if(HRT_USE_TENSORRT)
        target_link_libraries(hybridrt_test_image_model PRIVATE TensorRT::nvinfer CUDA::cudart kitzoo::log)
        target_compile_definitions(hybridrt_test_image_model PRIVATE
            HRT_TEST_TENSORRT HRT_TENSORRT_VERSION=${HRT_TENSORRT_VERSION})
    endif()
endif()
hybridrt_add_unit_test(segment SOURCES postprocess/segment_test.cpp DEPENDS hrt::hybridrt)
hybridrt_enable_model_test(segment)

hybridrt_add_unit_test(canvas SOURCES core/canvas_test.cpp DEPENDS hrt::hybridrt)
