# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------
# hybridrt_add_benchmark(name SOURCES ... [DEPENDS ...])
#
# Creates hybridrt_bench_<name> linked against benchmark::benchmark_main and
# registers a short CTest smoke run labeled "benchmark".
# ---------------------------------------------------------------------------

function(hybridrt_add_benchmark name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPENDS" ${ARGN})

    set(bench_name hybridrt_bench_${name})
    add_executable(${bench_name} ${ARG_SOURCES})
    target_link_libraries(${bench_name} PRIVATE benchmark::benchmark benchmark::benchmark_main ${ARG_DEPENDS})
    if(CMAKE_CROSSCOMPILING)
        target_compile_definitions(${bench_name} PRIVATE HRT_SOURCE_DIR=".")
    else()
        target_compile_definitions(${bench_name} PRIVATE HRT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    endif()
    hybridrt_apply_warnings(${bench_name})
    hybridrt_deploy_runtime(${bench_name})

    add_test(NAME bench_${name} COMMAND ${bench_name} --benchmark_min_time=0.01s)
    set_tests_properties(bench_${name} PROPERTIES LABELS benchmark)
endfunction()

# preprocess and postprocess time one stage, pipeline the whole ImageModel call,
# on the template models in assets/test/onnx; engine speed alone is tools/hrt-model-bench.
if(TARGET opencv_imgcodecs)
    hybridrt_add_benchmark(preprocess SOURCES preprocess_bench.cpp DEPENDS hrt::hybridrt opencv_imgcodecs)
endif()
if((HRT_USE_ORT OR HRT_USE_RKNN) AND TARGET opencv_imgcodecs)
    hybridrt_add_benchmark(postprocess SOURCES postprocess_bench.cpp DEPENDS hrt::hybridrt opencv_imgcodecs kitzoo::os)
    target_compile_definitions(hybridrt_bench_postprocess PRIVATE
        $<$<BOOL:${HRT_USE_RKNN}>:HRT_BENCH_RKNN>)
endif()
if((HRT_USE_ORT OR HRT_USE_RKNN) AND TARGET opencv_imgcodecs)
    hybridrt_add_benchmark(pipeline SOURCES pipeline_bench.cpp DEPENDS hrt::hybridrt opencv_imgcodecs)
    target_compile_definitions(hybridrt_bench_pipeline PRIVATE
        $<$<BOOL:${HRT_USE_ORT}>:HRT_BENCH_ORT>
        $<$<BOOL:${HRT_USE_RKNN}>:HRT_BENCH_RKNN>)
endif()
