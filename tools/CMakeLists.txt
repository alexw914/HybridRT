set(HYBRIDRT_TOOLS_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/build/tools"
    CACHE PATH "Output directory for command-line tools")

add_executable(hrt_compress hrt_compress.cpp)
target_link_libraries(hrt_compress PRIVATE hrt::hybridrt kitzoo::core kitzoo::log kitzoo::os)
set_target_properties(hrt_compress PROPERTIES OUTPUT_NAME hrt-compress RUNTIME_OUTPUT_DIRECTORY
                                                                        ${HYBRIDRT_TOOLS_OUTPUT_DIR})
hybridrt_apply_warnings(hrt_compress)
hybridrt_deploy_runtime(hrt_compress)

add_executable(hrt_model_bench hrt_model_bench.cpp)
target_link_libraries(hrt_model_bench PRIVATE hrt::hybridrt kitzoo::core kitzoo::log kitzoo::time)
set_target_properties(hrt_model_bench PROPERTIES OUTPUT_NAME hrt-model-bench RUNTIME_OUTPUT_DIRECTORY
                                                                             ${HYBRIDRT_TOOLS_OUTPUT_DIR})
hybridrt_apply_warnings(hrt_model_bench)
hybridrt_deploy_runtime(hrt_model_bench)

if(TARGET opencv_imgcodecs)
    add_executable(hrt_validate hrt_validate.cpp)
    target_link_libraries(hrt_validate PRIVATE hrt::hybridrt kitzoo::core kitzoo::log kitzoo::os kitzoo::thread kitzoo::time opencv_imgcodecs)
    set_target_properties(hrt_validate PROPERTIES OUTPUT_NAME hrt-validate RUNTIME_OUTPUT_DIRECTORY
                                                                           ${HYBRIDRT_TOOLS_OUTPUT_DIR})
    hybridrt_apply_warnings(hrt_validate)
    hybridrt_deploy_runtime(hrt_validate)
endif()
