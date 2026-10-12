# HybridRT

HybridRT is a C++20 model inference framework. It runs the preprocess, infer
and postprocess steps of a model across CPU, GPU and NPU backends, loading
models and their configs from a single `.hrtc` package. It has no inference
engine of its own: inference runs on the integrated frameworks listed under
Backends. Operators are provided by a separate library; foundation utilities
come from [kitzoo](https://github.com/alexw914/kitzoo).

## Build and test

Requirements: CMake 3.25+, Ninja, Git, a C++20 compiler, and OpenCV (core,
imgproc).

```sh
cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DOpenCV_DIR=<opencv>/lib/cmake/opencv4 -DORT_ROOT=<onnxruntime-sdk>
cmake --build build/debug --parallel
ctest --test-dir build/debug --output-on-failure
```

Run an algorithm on an image from a package or straight from its manifest (examples need OpenCV
imgcodecs). `assets/yolo` versions the manifests and test images but not the models: download
those listed in `assets/yolo/sources.json` into `assets/yolo/models/` first. Run from the
repository root:

```sh
build/tools/hrt-compress -j assets/yolo/yolo.json -o assets/yolo/yolo.hrtc
build/debug/examples/hybridrt_example_image_model -m assets/yolo/yolo.hrtc -i assets/yolo/images/bus.jpg -o result.jpg
build/debug/examples/hybridrt_example_image_model -m assets/yolo/yolo.json -a segment -i assets/yolo/images/bus.jpg -o segment.jpg
```

Benchmarks (Release, `-DHYBRIDRT_BUILD_BENCHMARKS=ON`) run on the `assets/yolo` tasks:
preprocessing, postprocessing, and the whole `ImageModel` call including pooled
multi-threaded throughput:

```sh
build/release/benchmarks/hybridrt_bench_preprocess --benchmark_min_time=1s
build/release/benchmarks/hybridrt_bench_postprocess --benchmark_min_time=1s
build/release/benchmarks/hybridrt_bench_pipeline --benchmark_min_time=2s
```

kitzoo, flatbuffers, GoogleTest and Google Benchmark are fetched from git. Use local checkouts
with `-DFETCHCONTENT_SOURCE_DIR_KITZOO=<path>` or
`-DFETCHCONTENT_SOURCE_DIR_FLATBUFFERS=<path>`. After editing `fbs/hrtc.fbs`,
configure with `-DHYBRIDRT_BUILD_FLATC=ON` and build the `hrtc_schema` target.

## Use in another CMake project

```cmake
include(FetchContent)
FetchContent_Declare(hybridrt
    GIT_REPOSITORY <repository-url>
    GIT_TAG <tag>)
FetchContent_MakeAvailable(hybridrt)

target_link_libraries(app PRIVATE hrt::hybridrt)
```

## Modules

| Target | Functionality |
| --- | --- |
| `hrt::hrtc` | The `.hrtc` model package: algorithms, configs, and per-backend model files. |
| `hrt::hybridrt` | Model execution with memory, contexts, engines, image preprocessing, task postprocessing, and OpenCV result drawing in `core/view`. |

`tools/hrt-compress` packs a JSON manifest (every field: `assets/template/hrtc_template.json`) into a
`.hrtc` file, prints or extracts existing packages, and checks their configs:

```sh
hrt-compress -j manifest.json -o model.hrtc
hrt-compress -i model.hrtc -x extracted/
hrt-compress -c -j manifest.json
```

`tools/hrt-model-bench` times engine inference alone, on generated inputs, for every algo and
backend of a package or manifest, using each engine's config:

```sh
hrt-model-bench -m model.hrtc|manifest.json [-a algo] [-e ort] [-n 100] [-w 10]
```

## Backends

A backend is enabled by pointing CMake at its SDK.

| Backend | Engine type | Image operators | CMake |
| --- | --- | --- | --- |
| ONNX Runtime | `ort` | CPU | `-DORT_ROOT=<sdk>` |
| ncnn | `ncnn` | CPU | `-Dncnn_ROOT=<sdk>` |
| MNN | `mnn` | CPU | `-DMNN_ROOT=<sdk>` |
| [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html) | `trt` | CUDA (NPP) | `-DTensorRT_ROOT=<sdk>` |
| [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2) | `rknn` | RGA | `-DRKNN_ROOT=<sdk> -DRGA_ROOT=<sdk>` |
| [CANN](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/index/index.html) | `cann` | DVPP | `-DCANN_ROOT=<sdk>` |

## Reference

| Project | Purpose |
| --- | --- |
| [kitzoo](https://github.com/alexw914/kitzoo) | Foundation utilities. |
| [FlatBuffers](https://github.com/google/flatbuffers) | hrtc package metadata. |
| [OpenCV](https://github.com/opencv/opencv) | Host image operators and rotated-box geometry. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime), [ncnn](https://github.com/Tencent/ncnn), [MNN](https://github.com/alibaba/MNN) | CPU inference backends. |
| [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html), [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2), [CANN](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/index/index.html) | Vendor inference backends. |
| [GoogleTest](https://github.com/google/googletest), [Google Benchmark](https://github.com/google/benchmark) | Unit tests and benchmarks. |
