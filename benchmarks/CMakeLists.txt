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
    target_compile_definitions(${bench_name} PRIVATE HRT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
    hybridrt_apply_warnings(${bench_name})

    add_test(NAME bench_${name} COMMAND ${bench_name} --benchmark_min_time=0.01s)
    set_tests_properties(bench_${name} PROPERTIES LABELS benchmark)
endfunction()

# preprocess and postprocess time one stage, pipeline the whole ImageModel call,
# on the assets/yolo models; engine speed alone is tools/hrt-model-bench.
if(TARGET opencv_imgcodecs)
    hybridrt_add_benchmark(preprocess SOURCES preprocess_bench.cpp DEPENDS hrt::hybridrt opencv_imgcodecs)
endif()
if(HRT_USE_ORT AND TARGET opencv_imgcodecs)
    hybridrt_add_benchmark(postprocess SOURCES postprocess_bench.cpp DEPENDS hrt::hybridrt opencv_imgcodecs)
    hybridrt_add_benchmark(pipeline SOURCES pipeline_bench.cpp DEPENDS hrt::hybridrt opencv_imgcodecs)
endif()
