# HybridRT

HybridRT is a C++20 model inference framework. It runs the preprocess, infer
and postprocess steps of a model across CPU, GPU and NPU backends, loading
models and their configs from a single `.hrtc` package. It has no inference
engine of its own: inference runs on the integrated frameworks listed under
Backends. Foundation utilities come from [kitzoo](https://github.com/alexw914/kitzoo).

## Build and test

Requirements: CMake 3.25+, Ninja, Git, a C++20 compiler, and OpenCV (core,
imgproc).

```sh
cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DOpenCV_DIR=<opencv>/lib/cmake/opencv4 -DORT_ROOT=<onnxruntime-sdk>
cmake --build build/debug --parallel
ctest --test-dir build/debug --output-on-failure
```

CI covers Linux, macOS and Windows, including an Ultralytics ONNX model smoke test on
Windows. Platforms the tests last ran on:

| Platform | Compiler | Backends |
| --- | --- | --- |
| Linux x86_64 (CI) | Clang | ONNX Runtime 1.31.0 |
| Linux x86_64, RTX 4090 | GCC 11.4 | TensorRT 10.16.1.11, ONNX Runtime 1.23.2 |
| Linux x86_64 (WSL2, Ubuntu 24.04) | GCC 13.3 | none (CPU image operators and postprocessing) |
| macOS arm64 | Apple Clang | ONNX Runtime 1.31.0 (CPU, CoreML), MNN 3.0.0 |
| Windows x64 | MSVC 19.50 | ONNX Runtime 1.31.0, MNN 3.0.0, ncnn 20260526; TensorRT 10.16.1.11 on CUDA 12.6 (RTX 3060) |
| RK3588 (Radxa Rock 5B+, Debian 12) | Arm GNU Toolchain 12.2 (cross) | RKNN runtime 2.3.2 with librga 1.10.1 |

Cross-compile for RK3588 with an aarch64 toolchain file whose glibc and libstdc++ are not newer
than the board's (Arm GNU Toolchain 12.2 matches Debian 12). Tools go to `build/tools` unless
`HYBRIDRT_TOOLS_OUTPUT_DIR` is set. Test binaries embed source paths, so run them on the board
from the same directory layout. Under WSL, keep CMake away from Windows packages on the inherited
`PATH`, e.g. with `-DCMAKE_IGNORE_PREFIX_PATH=<windows-prefix>`.

```sh
cmake -S . -B build/rk3588 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<aarch64-toolchain.cmake> \
    -DRKNN_ROOT=<rknpu2> -DRKNN_PLATFORM=RK3588 -DRGA_ROOT=<librga> \
    -DOpenCV_DIR=<aarch64-opencv>/lib/cmake/opencv4 -DHYBRIDRT_TOOLS_OUTPUT_DIR=build/rk3588/tools
cmake --build build/rk3588 --parallel
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

`tools/hrt-model-bench` times engine inference alone, on generated inputs, for every backend of
one algo of a package or manifest, using each engine's config:

```sh
hrt-model-bench -m model.hrtc|manifest.json -n algo [-e ort] [-i 100] [-w 10]
```

`tools/hrt-validate` runs one algo on every image of a directory on 8 threads and writes the
results of any task in one JSON layout (`assets/template/results.json`), by default to
`<input>_<algo>.json` beside the directory; `-r` also saves the images with drawn results. It
exits with 1 if any image fails:

```sh
hrt-validate -m model.hrtc|manifest.json -n algo -i images/ [-e ort] [-o images_algo.json] \
             [-r drawn/] [--preprocess-num 3] [--engine-num 2] [--postprocess-num 3]
```

## Backends

A backend is enabled by pointing CMake at its SDK. Version is the minimum supported release and,
in parentheses, the version tests last ran on; `-` means no minimum is pinned.
A row without a parenthesized version has not been tested.

| Backend | Engine type | Image operators | CMake | Version |
| --- | --- | --- | --- | --- |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | `ort` | CPU | `-DORT_ROOT=<sdk>` | 1.20+ (1.31.0) |
| [ncnn](https://github.com/Tencent/ncnn) | `ncnn` | CPU | `-Dncnn_ROOT=<sdk>` | - (1.0.20261007) |
| [MNN](https://github.com/alibaba/MNN) | `mnn` | CPU | `-DMNN_ROOT=<sdk>` | - (3.0.0) |
| [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html) | `trt` | CUDA (NPP) | `-DTensorRT_ROOT=<sdk>` | 8.5+ (10.16.1) |
| [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2) | `rknn` | RGA | `-DRKNN_ROOT=<sdk> -DRGA_ROOT=<sdk>` | - (2.3.2) |
| [CANN](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/index/index.html) | `cann` | DVPP | `-DCANN_ROOT=<sdk>` | - |

MNN selects Metal with engine config `{"forward_type": "metal"}` and requires an SDK
built with Metal support. The official macOS Metal packages for 3.0.0 and 3.6.1
failed standalone ReLU inference on macOS 27.0.1 arm64 with shader compilation
errors and a crash; CPU 3.0.0 remains the validated configuration.

RKNN takes uint8 images with `{"image_input": true}`; set `dst_image_format` to `inter_rgb`, as
the runtime applies the model's layout and normalization. RGA fills and resizes interleaved RGB/BGR
images whose rows are a multiple of 16 pixels. Host images are first copied into the CMA heap
(RGA2 cannot address memory above 4 GB); other images, and calls whose copy does not fit in the
heap, run on the CPU.
