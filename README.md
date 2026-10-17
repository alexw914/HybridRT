# HybridRT

HybridRT is a C++20 model inference framework for CPU, GPU and NPU deployment.
It combines image preprocessing, inference and postprocessing behind one interface,
with models and configuration stored in a single .hrtc package.

Supported model families and tasks:

| Task | Models and output formats |
| --- | --- |
| Object detection | YOLOv5, YOLOv8, YOLO26, RT-DETR / DETR, decoded YOLO outputs, end-to-end detection |
| OBB (Oriented Bounding Box) detection | YOLOv5 OBB, YOLOv8 OBB, YOLO26 OBB, decoded and end-to-end OBB outputs |
| Instance segmentation | YOLOv8-seg, YOLO26-seg, decoded YOLO segmentation, end-to-end segmentation |
| Pose and landmarks | YOLOv8-pose, YOLO26-pose, RTMO, RTMPose / SimCC, HRNet / heatmaps, coordinate outputs, end-to-end pose |
| Classification | Classification scores, including YOLO classification exports |
| Semantic segmentation | Per-pixel class scores or labels |

Model support depends on the exported output format. See the
[model and configuration reference](assets/template/README.md) for compatible exports,
parameters and test model sources.

## Build and test

Requirements: CMake 3.25+, Git, a C++20 compiler, and OpenCV with core and imgproc.
Image examples and the validation tool also require imgcodecs.
Install the SDK for each inference backend you want to enable.

| Platform | Build toolchain | Verification coverage |
| --- | --- | --- |
| Linux x86_64 | Clang or GCC | CI with ONNX Runtime; additional NVIDIA CUDA / TensorRT validation |
| Windows x64 | MSVC | CI with ONNX Runtime and real-model validation; additional MNN, ncnn and TensorRT tests |
| macOS arm64 | Apple Clang | CI with ONNX Runtime; additional CoreML and MNN CPU validation |
| Linux aarch64 / RK3588 | Native or cross-compilation with an aarch64 C++20 toolchain | RKNN / RGA, MNN and ncnn board validation |

CI coverage does not imply that every backend is tested on every platform.
CANN integration is available but has not been validated on Ascend hardware.

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
| `hrt-compress` | Package, inspect, extract and validate model manifests |
| `hrt-validate` | Run image-directory validation and save results and annotated images |
| Image model example | Run a configured model on an image |

Start with the [model template](assets/template/hrtc_template.json) and
[backend configuration examples](assets/template/hrtc_device_config.json).
Model and backend parameters are in the [configuration reference](assets/template/README.md).

## Backends

Enable one or more backends by providing their SDK paths at CMake configuration time.

| Backend | Engine name | Model format | CMake settings |
| --- | --- | --- | --- |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | ort | ONNX | `-DORT_ROOT=<sdk>` |
| [TensorRT](https://github.com/NVIDIA/TensorRT) | trt | TensorRT engine | `-DTensorRT_ROOT=<sdk>`; CUDA Toolkit required |
| [MNN](https://github.com/alibaba/MNN) | mnn | MNN | `-DMNN_ROOT=<sdk>` |
| [ncnn](https://github.com/Tencent/ncnn) | ncnn | ncnn param + bin | `-Dncnn_ROOT=<sdk>` |
| [RKNN](https://github.com/airockchip/rknn-toolkit2) | rknn | RKNN | `-DRKNN_ROOT=<sdk>` `-DRGA_ROOT=<sdk>` |
| [CANN](https://www.hiascend.com/en/software/cann) | cann | Ascend OM | `-DCANN_ROOT=<sdk>` |

ONNX Runtime supports CPU and optional CoreML execution. MNN acceleration depends on
the selected SDK and platform; CPU is the validated macOS configuration.
Image preprocessing supports CPU, CUDA, Apple Metal, Rockchip RGA and Ascend DVPP;
postprocessing runs on the CPU. Backend availability depends on the target hardware
and installed SDK, and models must be exported or converted to its required format.
