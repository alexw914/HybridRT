set(HYBRIDRT_TOOLS_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/build/tools"
    CACHE PATH "Output directory for command-line tools")

add_executable(hrt_compress hrt_compress.cpp)
target_link_libraries(hrt_compress PRIVATE hrt::hybridrt kitzoo::core kitzoo::log kitzoo::os)
set_target_properties(hrt_compress PROPERTIES OUTPUT_NAME hrt-compress RUNTIME_OUTPUT_DIRECTORY
                                                                        ${HYBRIDRT_TOOLS_OUTPUT_DIR})
hybridrt_apply_warnings(hrt_compress)

add_executable(hrt_model_bench hrt_model_bench.cpp)
target_link_libraries(hrt_model_bench PRIVATE hrt::hybridrt kitzoo::core kitzoo::log)
set_target_properties(hrt_model_bench PROPERTIES OUTPUT_NAME hrt-model-bench RUNTIME_OUTPUT_DIRECTORY
                                                                             ${HYBRIDRT_TOOLS_OUTPUT_DIR})
hybridrt_apply_warnings(hrt_model_bench)
