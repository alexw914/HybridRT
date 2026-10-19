# HybridRT

HybridRT is a C++20 model inference framework for CPU, GPU and NPU deployment.
It combines image preprocessing, inference and postprocessing behind one interface,
with models and configuration stored in a single .hrtc package.

## Supported Models

| Task | Models and output formats |
| --- | --- |
| Object detection | YOLOv5, YOLOv8, YOLO26, RT-DETR / DETR, decoded YOLO outputs, end-to-end detection |
| OBB (Oriented Bounding Box) detection | YOLOv5 OBB, YOLOv8 OBB, YOLO26 OBB, decoded and end-to-end OBB outputs |
| Instance segmentation | YOLOv8-seg, YOLO26-seg, decoded YOLO segmentation, end-to-end segmentation |
| Pose and landmarks | YOLOv8-pose, YOLO26-pose, RTMO, RTMPose / SimCC, HRNet / heatmaps, coordinate outputs, end-to-end pose |
| Classification | Classification scores, including YOLO classification exports |
| Embedding | Raw feature vectors, e.g. ReID or image retrieval |
| Semantic segmentation | Per-pixel class scores or labels |
| OCR | PP-OCR DB text detection and CTC recognition, including PP-OCRv6 tiny/small |

Model support depends on the exported output format. See the
[model and configuration reference](assets/template/README.md) for compatible exports,
parameters and test model sources.

## Build and test

Requirements: CMake 3.25+, Git, a C++20 compiler, and OpenCV with core and imgproc.
Image examples and the validation tool also require imgcodecs.
Install the SDK for each inference backend you want to enable.

| Platform | Build toolchain | Verification coverage |
| --- | --- | --- |
| Linux x86_64 | GCC 11.4 / 13.3; Clang 18.1.3 | CI with ONNX Runtime; additional NVIDIA CUDA / TensorRT validation |
| Windows x64 | MSVC 19.50 | CI with ONNX Runtime and real-model validation; additional MNN, ncnn and TensorRT tests |
| macOS arm64 | Apple Clang 21.0.0 | CI with ONNX Runtime; additional CoreML and MNN CPU validation |
| Linux aarch64 / RK3588 | Arm GNU Toolchain 12.2 | RKNN / RGA, MNN and ncnn board validation |
| Linux aarch64 / Ascend | GCC 12.3 | CANN engine and host preprocessing validation on an Ascend 910B2; DVPP operators validated on an Atlas 300V Pro |

Build with ONNX Runtime, using Ninja:

```sh
cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DORT_ROOT=<onnxruntime-sdk>
cmake --build build/debug --parallel
```

The default build includes tools and examples; tests are opt-in. CMake searches for
OpenCV in the system installation. Set OpenCV_DIR only when it cannot be found.
On Windows, run from an MSVC developer shell and use the matching Windows SDKs.
For cross-compilation, add -DCMAKE_TOOLCHAIN_FILE=<toolchain.cmake> and provide
target-platform OpenCV and backend SDKs.

| CMake option | Purpose | Default in a standalone build |
| --- | --- | --- |
| `HYBRIDRT_BUILD_TESTS` | Build tests | OFF |
| `HYBRIDRT_BUILD_TOOLS` | Build command-line tools | ON |
| `HYBRIDRT_BUILD_EXAMPLES` | Build image examples | ON |
| `HYBRIDRT_BUILD_BENCHMARKS` | Build benchmarks | OFF |
| `HYBRIDRT_BUILD_C_API` | Build the `hybridrt_c` shared library with the C API | OFF |
| `HYBRIDRT_BUILD_PYTHON` | Build the C API for the Python package and register its tests | OFF |
| `HYBRIDRT_BUILD_RUST` | Build the C API for the Rust crates and register their tests | OFF |

To enable and run tests:

```sh
cmake -S . -B build/debug -DHYBRIDRT_BUILD_TESTS=ON
cmake --build build/debug --parallel
ctest --test-dir build/debug --output-on-failure
```

Dependencies are fetched during configuration; local checkouts can be supplied through
CMake's FETCHCONTENT_SOURCE_DIR_<NAME> options. Tests that need unavailable models,
images or SDKs are skipped. Test resources are local files under assets/test/ and
are not included in the repository.

Run an image example after preparing the model referenced by the template:

```sh
build/debug/examples/hybridrt_example_image_model -m assets/template/hrtc_template.json \
    -a yolov8n_segment -i assets/test/images/bus.jpg -o segment.jpg
```

Windows executables have the .exe suffix.

## Use in another CMake project

```cmake
include(FetchContent)
FetchContent_Declare(hybridrt
    GIT_REPOSITORY https://github.com/alexw914/HybridRT.git
    GIT_TAG <tag-or-commit>)
FetchContent_MakeAvailable(hybridrt)

target_link_libraries(app PRIVATE hrt::hybridrt)
```

Set the required backend SDK paths before configuring the parent project.

## Modules

| Component | Purpose |
| --- | --- |
| `hrt::hybridrt` | Image model inference, preprocessing, postprocessing and result drawing |
| `hrt::hrtc` | Model packaging and configuration without an inference runtime dependency |
| `hrt::hybridrt_c` | C API in [include/capi/hrt.h](include/capi/hrt.h): image models on host images, flattened results |
| [Python package](bindings/python/README.md) | numpy interface over the C API, built with `pip install ./bindings/python` |
| [Rust crates](bindings/rust/README.md) | `hybridrt-sys` FFI and the safe `hybridrt` crate over the C API |
| `hrt-compress` | Package, inspect, extract and validate model manifests |
| `hrt-validate` | Run image-directory validation and save results and annotated images |
| `hrt-ocr` | Run OCR on an image directory and save the texts as JSON |
| Image model example | Run a configured model on an image |
| OCR model example | Run two selected algos with `OcrModel`, including ORT CoreML execution |

Start with the [model template](assets/template/hrtc_template.json) and
[backend configuration examples](assets/template/hrtc_device_config.json).
Model and backend parameters are in the [configuration reference](assets/template/README.md).

## Backends

Enable one or more backends by providing their SDK paths at CMake configuration time.

| Backend | Engine name | Model format | CMake settings | Minimum version | Tested version |
| --- | --- | --- | --- | --- | --- |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | ort | ONNX | `-DORT_ROOT=<sdk>` | 1.20+ | 1.31.0 |
| [TensorRT](https://github.com/NVIDIA/TensorRT) | trt | TensorRT engine | `-DTensorRT_ROOT=<sdk>`; CUDA Toolkit required | 8.5+ | 10.16.1 |
| [MNN](https://github.com/alibaba/MNN) | mnn | MNN | `-DMNN_ROOT=<sdk>` | Not pinned | 3.0.0 |
| [ncnn](https://github.com/Tencent/ncnn) | ncnn | ncnn param + bin | `-Dncnn_ROOT=<sdk>` | Not pinned | 1.0.20261007 |
| [RKNN](https://github.com/airockchip/rknn-toolkit2) | rknn | RKNN | `-DRKNN_ROOT=<sdk>` `-DRGA_ROOT=<sdk>` | Not pinned | 2.3.2 |
| [CANN](https://www.hiascend.com/en/software/cann) | cann | Ascend OM | `-DCANN_ROOT=<sdk>` | 8.2 | 8.2 and 9.0.0 |

ONNX Runtime supports CPU and optional CoreML execution. MNN acceleration depends on
the selected SDK and platform; CPU is the validated macOS configuration.
Image preprocessing supports CPU, CUDA, Apple Metal, Rockchip RGA and Ascend DVPP;
postprocessing runs on the CPU. Backend availability depends on the target hardware
and installed SDK, and models must be exported or converted to its required format.
