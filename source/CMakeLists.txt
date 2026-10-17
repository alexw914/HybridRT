include(GNUInstallDirs)

configure_file(
    "${PROJECT_SOURCE_DIR}/include/hrt/core/version.hpp.in"
    "${PROJECT_BINARY_DIR}/include/hrt/core/version.hpp"
    @ONLY)

function(hybridrt_configure_target target)
    add_library(hrt::${target} ALIAS ${target})
    target_include_directories(
        ${target}
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
               $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(${target} PUBLIC cxx_std_20)
    set_target_properties(
        ${target}
        PROPERTIES POSITION_INDEPENDENT_CODE ON
                   CXX_VISIBILITY_PRESET hidden
                   VISIBILITY_INLINES_HIDDEN ON)
    hybridrt_apply_warnings(${target})
endfunction()

# -- hrtc: model package format, usable without the inference runtime ---------
add_library(hrtc STATIC hrtc/hrtc.cpp)
hybridrt_configure_target(hrtc)
target_link_libraries(hrtc PUBLIC kitzoo::core kitzoo::json kitzoo::memory PRIVATE flatbuffers kitzoo::log kitzoo::os)

# -- hybridrt: memory, buffers, images, preprocessing and engines -------------
add_library(hybridrt STATIC
    core/define.cpp
    core/context.cpp
    core/memory.cpp
    core/buffer.cpp
    core/image.cpp
    core/config.cpp
    core/view.cpp
    engine/engine.cpp
    memory/cpu_memory.cpp
    memory/memory_copier.cpp
    preprocess/image_preprocessor.cpp
    preprocess/cpu_image_preprocessor.cpp
    postprocess/common.cpp
    postprocess/detect.cpp
    postprocess/detect/yolov5_detect.cpp
    postprocess/detect/yolov8_detect.cpp
    postprocess/detect/yolo26_detect.cpp
    postprocess/detect/general_detect.cpp
    postprocess/detect/detr_detect.cpp
    postprocess/detect/e2e_detect.cpp
    postprocess/rotated.cpp
    postprocess/rotated/yolov5_rotated.cpp
    postprocess/rotated/yolov8_rotated.cpp
    postprocess/rotated/yolo26_rotated.cpp
    postprocess/rotated/general_rotated.cpp
    postprocess/rotated/e2e_rotated.cpp
    postprocess/classify/general_classify.cpp
    postprocess/keypoint/general_keypoint.cpp
    postprocess/keypoint/e2e_keypoint.cpp
    postprocess/keypoint/rtmo_keypoint.cpp
    postprocess/keypoint/general_yolo_pose.cpp
    postprocess/keypoint/yolov8_keypoint.cpp
    postprocess/keypoint/yolo26_keypoint.cpp
    postprocess/segment.cpp
    postprocess/segment/mask_decoder.cpp
    postprocess/segment/general_yolo_segment.cpp
    postprocess/segment/yolov8_segment.cpp
    postprocess/segment/yolo26_segment.cpp
    postprocess/segment/e2e_segment.cpp
    postprocess/segment/semantic_segment.cpp
    model/image_model.cpp)
hybridrt_configure_target(hybridrt)
target_link_libraries(hybridrt
    PUBLIC hrt::hrtc kitzoo::core kitzoo::json kitzoo::memory kitzoo::thread opencv_core
    PRIVATE kitzoo::log kitzoo::utilities opencv_imgproc $<TARGET_NAME_IF_EXISTS:opencv_geometry>)

# Linux dma-bufs are shared by media devices and read by host engines.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(hybridrt PRIVATE memory/dma_buffer_memory.cpp)
    target_compile_definitions(hybridrt PRIVATE HRT_USE_DMA_BUFFER)
endif()

# -- Engines -------------------------------------------------------------------
if(APPLE)
    enable_language(OBJCXX)
    target_sources(hybridrt PRIVATE preprocess/metal_image_preprocessor.cpp)
    set_source_files_properties(preprocess/metal_image_preprocessor.cpp PROPERTIES
        LANGUAGE OBJCXX COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(hybridrt PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED ON)
    target_link_libraries(hybridrt PRIVATE "-framework Metal" "-framework Foundation")
    target_compile_definitions(hybridrt PRIVATE HRT_USE_METAL)
endif()

if(HRT_USE_ORT)
    target_sources(hybridrt PRIVATE engine/onnxruntime_engine.cpp)
    target_link_libraries(hybridrt PRIVATE onnxruntime)
    target_compile_definitions(hybridrt PRIVATE HRT_USE_ORT)
endif()

if(HRT_USE_CUDA)
    target_sources(hybridrt PRIVATE
        backend/cuda/cuda_common.cpp
        backend/cuda/cuda_context.cpp
        backend/cuda/cuda_memory.cpp
        backend/cuda/cuda_host_memory.cpp
        preprocess/cuda_image_preprocessor.cpp)
    target_link_libraries(hybridrt PRIVATE
        CUDA::cudart CUDA::nppc CUDA::nppial CUDA::nppicc CUDA::nppidei CUDA::nppig)
    target_compile_definitions(hybridrt PRIVATE HRT_USE_CUDA)
endif()

if(HRT_USE_TENSORRT)
    target_sources(hybridrt PRIVATE engine/tensorrt_engine.cpp)
    target_link_libraries(hybridrt PRIVATE TensorRT::nvinfer TensorRT::nvinfer_plugin ${CMAKE_DL_LIBS})
    target_compile_definitions(hybridrt PRIVATE HRT_USE_TENSORRT HRT_TENSORRT_VERSION=${HRT_TENSORRT_VERSION})
endif()

if(HRT_USE_RKNN)
    target_sources(hybridrt PRIVATE
        preprocess/rga_image_preprocessor.cpp
        engine/rknn_engine.cpp)
    target_link_libraries(hybridrt PRIVATE rknn rga ${CMAKE_DL_LIBS})
    target_compile_definitions(hybridrt PRIVATE HRT_USE_RKNN $<$<BOOL:${HYBRIDRT_RGA3_ONLY}>:HRT_RGA3_ONLY>)
endif()

if(HRT_USE_CANN)
    target_sources(hybridrt PRIVATE
        backend/ascend/acl_common.cpp
        backend/ascend/acl_context.cpp
        backend/ascend/acl_memory.cpp
        preprocess/dvpp_image_preprocessor.cpp
        engine/cann_engine.cpp)
    target_link_libraries(hybridrt PRIVATE cann::ascendcl cann::acl_dvpp_mpi)
    target_compile_definitions(hybridrt PRIVATE HRT_USE_CANN)
endif()


if(HRT_USE_NCNN)
    target_sources(hybridrt PRIVATE engine/ncnn_engine.cpp)
    target_link_libraries(hybridrt PRIVATE ncnn)
    target_compile_definitions(hybridrt PRIVATE HRT_USE_NCNN)
endif()

if(HRT_USE_MNN)
    target_sources(hybridrt PRIVATE engine/mnn_engine.cpp)
    get_target_property(mnn_library_type MNN TYPE)
    if(mnn_library_type STREQUAL "SHARED_LIBRARY")
        target_link_libraries(hybridrt PRIVATE MNN)
    else()
        # Operators register from static initializers, so keep every object.
        target_link_libraries(hybridrt PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,MNN>")
    endif()
    target_compile_definitions(hybridrt PRIVATE HRT_USE_MNN)
endif()

# -- hybridrt_c: C API for language bindings ----------------------------------
if(HYBRIDRT_BUILD_C_API)
    add_library(hybridrt_c SHARED hrt.cpp)
    hybridrt_configure_target(hybridrt_c)
    target_compile_definitions(hybridrt_c PRIVATE HRT_C_API_BUILD)
    target_link_libraries(hybridrt_c PRIVATE hrt::hybridrt kitzoo::log)
    # Only the hrt_ functions are exported; static dependencies stay private.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(hybridrt_c PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
    hybridrt_deploy_runtime(hybridrt_c)
endif()
