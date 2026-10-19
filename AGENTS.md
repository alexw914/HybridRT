# 代码编写规则

适用于仓库自有代码和 CMake 头文件模板；不直接修改第三方依赖、引擎 SDK 和 build/ 生成文件。
公开接口调整需明确授权，不为遵守新规则破坏兼容性。本文件只保留长期规则，不记录一次性配置、
机器地址、环境路径或安装步骤；可由工具配置一次后持续生效的设置放入对应配置文件。

## 项目与基础能力

- HybridRT 是 C++20 模型推理框架，只负责预处理、推理、后处理及模型封装。
- 通用能力统一使用 kitzoo，不重复实现或包装；缺失能力补充到 kitzoo，只链接所需模块：
  JSON 用 `kitzoo::json::Reader` 解析、nlohmann 原生接口读取，参数结构体用
  `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT` 映射；`get<T>()` 前用
  `source/include/core/config.hpp` 的 `check_param` 无异常检查，未知键也报错。
  主机内存用 `kitzoo::memory::ByteBuffer`，日志用 `KZ_LOG_*`，命令行用
  `kitzoo::core::Options`，文件用 `kitzoo::os`，字符串工具用 `kitzoo::util`，
  属性用 `KZ_NODISCARD`、`KZ_MAYBE_UNUSED`。
- 容器、字符串、智能指针统一用 kitzoo 别名 `Array`、`Vector`、`Map`、`Set`、
  `Pair`、`String`、`UniquePtr`、`SharedPtr`，不写对应 std 类型；别名可直接传给第三方接口。
  `include/hrt/core/containers.hpp` 将其引入 hrt，命名空间外写 `hrt::Vector` 等。
  该头仅由 `define.hpp`、`context.hpp`、`hrtc.hpp`、`config.hpp` 直接包含；
  其余文件不重复包含它或它已引入的标准容器、字符串头。
- 链接 `kitzoo::memory` 的可执行文件由 kitzoo 用 mimalloc 替换进程分配器，库代码不指定分配器。

## 目录与模块

- 公开头文件放 `include/hrt/`，实现放 `source/`；私有头文件放 `source/include/<模块>/`，
  按 `"include/<模块>/<文件>.hpp"` 引用，`source/<模块>/` 只放 .cpp。
  SDK 查找脚本放 `cmake/module/`，第三方依赖配置放 `cmake/external/`。
- `hrt::hrtc` 不依赖推理运行时。主机内存（含 Linux dma-buffer）及拷贝分发放
  `source/memory/`，设备公共代码（context、memory、SDK 辅助）放
  `source/backend/<device>/`，任务解码放 `source/postprocess/<task>/`。
- 后端通过 SDK 根目录启用并定义 `HRT_USE_<BACKEND>`，见 `cmake/HybridrtBackends.cmake`；
  SDK 头文件和类型不得出现在公开头文件。新增后端只扩展
  `source/engine/<backend>_engine.*`、`source/CMakeLists.txt` 和 `Engine::create`；
  新增设备扩展 `Memory::create`/`wrap`、`copy_memory`、`Context::create`、
  `ImagePreprocessor::create`，上层流程不按后端分支。
- 预处理流程只在 `ImagePreprocessor` 基类实现，设备子类只实现算子；
  不支持的步骤由 `supports()` 返回 false 并回退 CPU。后处理只在主机执行，按 `postprocess.type` 注册。
- `ImageModel` 的 algo 只描述模型（preprocess、postprocess），引擎只读取自己的 config。
  各阶段实例数只通过 create 的 `ImageModelOptions` 传入，不写入模型包。
  自定义算子库从引擎 config 加载（ORT/RKNN 的 `custom_op_libraries`、
  TensorRT 的 `plugin_libraries`），可选，加载失败只警告并跳过。
- C API 声明在 `include/hrt.h`（C99 可编译），实现为 `source/hrt.cpp`，由 `HYBRIDRT_BUILD_C_API`
  构建为共享库 `hybridrt_c`，只导出 `hrt_` 函数。类型用 `Hrt` 前缀大驼峰，函数用 `hrt_` 前缀，
  枚举项为显式赋值的 `HRT_<类型>_<值>`；枚举值和库分配的结构体字段只在末尾追加，
  调用方分配的结构体不改布局。可预先检查的错误返回 `HrtStatus`，信息由 `hrt_last_error` 给出；
  C API 只接收主机内存输入，结果为主机数组。该共享库构建关闭 kitzoo 的 mimalloc 进程替换，
  并使用可 dlopen 的 TLS 模型。
- 语言绑定放 `bindings/<语言>/`，只调用 C API，不链接 C++ 库，由 `HYBRIDRT_BUILD_<语言>` 启用；
  Python 绑定用 ctypes，按 Python 惯例抛出 `HybridRTError`，用 scikit-build-core 打包；
  Rust 绑定分 `hybridrt-sys`（手写 FFI，布局与 hrt.h 一致并有测试）和安全封装 `hybridrt`，
  返回 `Result`，unsafe 只在 FFI 调用处并写 SAFETY 注释，用 rustfmt、clippy 检查。
  每种语言提供 `run_models` 示例（`bindings/python/examples/`、`bindings/rust/hybridrt/examples/`
  的 cargo example），每种任务类型各一个 algo 并含 OCR，由绑定测试运行或编译。
- `source/include/hrtc/hrtc_generated.h` 随仓库提交，不手改；修改 `fbs/hrtc.fbs` 后用
  `HYBRIDRT_BUILD_FLATC` 启用 flatc 并构建 `hrtc_schema` 重新生成。
  字段只在表末尾追加；不兼容改动更换 `file_identifier` 并重新打包。

## 模板与资源

- `assets/` 只提交 `template/` 和 `PERF.md`；模型、.hrtc、图片、夹具 manifest、输出结果、
  下载到 `external/` 的 SDK 均不提交。测试资源统一放本地 `assets/test/`，ONNX 放其 onnx 子目录；
  不另建夹具 README、manifest 或 sources 文件。manifest 路径为绝对路径或相对运行时工作目录。
- `assets/template/hrtc_template.json` 展示模型参数，每种 postprocess 至少一个 algo，不要求每个模型都有 algo。
  有测试模型时以模型命名，使用其真实前后处理配置及仓库根目录相对模型路径，测试和基准从模板读取。
  引擎专用 config 示例放 `hrtc_device_config.json`。
- `assets/template/README.md` 说明每种 preprocess、engine、postprocess 的作用、全部参数、
  默认值、可选值及平台必需参数（如 RKNN 的 `image_input`、`dst_image_format`）。
  类型、参数、默认值或平台行为变化时，同一提交同步文档及相关 JSON 模板。
  同类型模型的变体、导出方式、输出列含义在此说明；Test models 记录来源、导出或转换方式、
  sha256 和验证结论，转换脚本只放本地 assets/test/scripts/，不提交。
- 前处理、后处理或 ImageModel pipeline 性能优化在同一提交更新 `assets/PERF.md`，最新在前：
  记录瓶颈与定位、改动、测量条件（平台、模型、输入、线程、次数、工具）、前后数据及剩余问题。
  普通功能不写入。引擎推理耗时只用 `tools/hrt-model-bench` 测量；
  benchmarks/ 只保留 preprocess_bench、postprocess_bench、pipeline_bench。

## 文件、格式与命名

- 自有文件使用以下英文文件头；File 为仓库根目录相对路径，Description 简述职责，续行对齐，
  不写作者、日期和修改历史：

  ```cpp
  // -----------------------------------------------------------------------------
  // HybridRT | C++20 Model Inference Framework
  // File: include/hrt/engine/engine.hpp
  // Description: Declares the inference engine interface.
  // -----------------------------------------------------------------------------
  ```

- Python 文件使用相同内容的 `#` 注释文件头，Rust 文件使用相同的 `//` 文件头。
- 使用 include guard，禁止 pragma once。宏名 `HRT_<目录>_<文件名>_HPP`，全大写、点换下划线，
  冲突时增加一级目录；末尾 endif 注释标明宏名。.hpp.in 按生成的 .hpp 命名，
  保持 `@变量@` 完整，必要时用 clang-format off/on。
- 遵循根目录 `.clang-format`，clang-format >=19，检查用 `--dry-run --Werror`；
  静态检查用 `.clang-tidy` 和 build/ 下的 compile_commands.json。
  include 顺序为 hrt、kitzoo、其他尖括号头、双引号头；限定符在类型左侧。
  非 C++ 文本用 LF，.bat、.cmd 用 CRLF。
- 命名空间按模块：`hrt::core`、engine、preprocess、postprocess、model、hrtc；
  内部另有 memory、backend。跨模块写模块前缀（如 `core::Image`），不用 detail，
  文件内私有代码放匿名命名空间。
- 类型大驼峰，函数和变量小写下划线，常量 `kName`，宏 `HRT_` 前缀；第三方接口沿用原约定。
- 枚举用 enum class，枚举项大驼峰，最后一项保留逗号；只在序列化、ABI 等必要场景指定底层类型和数值。
  枚举项和结构体成员的简短说明统一放在声明后的行尾，与声明保持同一行：

  ```cpp
  enum class CopyPolicy {
    KeepSrcStride, // dst takes src's shape and stride.
    AutoDstStride, // dst takes src's shape with a packed stride.
    DirectCopy,    // dst already has src's shape; only data is copied.
  };

  struct ObjectBox {
    core::Rect2f rect;           // Bounding box.
    float score = 0.0F;          // Confidence score.
    std::int64_t type = 0;       // Class ID.
    std::int64_t track_id = -1;  // Track ID; -1 means unassigned.
  };
  ```

- 注释只解释代码无法表达的约束、原因、数据布局和选项，不描述改动过程或显然逻辑；
  上述行尾形式用于需要说明的成员，不要求逐项添加重复注释。
- 只有持有进程级共享状态的对象才继承 `kitzoo::core::Singleton<T>`；无状态工具用自由函数。

## 函数与错误处理

- 头文件只声明，实现放 .cpp，模板和 constexpr 除外。函数声明间空一行，相关重载可成组；
  函数和有返回值的 lambda 用尾置返回类型。
- 只读大对象传 `const T&`，修改调用方对象传 `T&`，转移所有权按值或 `T&&`；
  数值、指针、std::string_view、std::span 按值传递。
- 所有代码（含工具、示例、测试）不抛异常、不用 try/catch：
  初始化配置错误或无可用引擎等用 `KZ_CHECK_MSG` 记录后 abort；
  推理失败用 `KZ_CHECK_RETURN` 记录后返回空结果（std::nullopt、false）。
  用于挑选的 `Engine::create`/initialize、`Context::create`、
  `ImagePreprocessor::create` 和 `hrtc::load` 返回空值，由调用方决定。
- SDK 状态码直接传检查宏，错误信息取自 SDK；优先用不抛异常的 SDK 接口（如 ORT C API），
  SDK 类型不泄漏到上层。资源回滚用 RAII 或 `KZ_SCOPE_FAIL`，abort 用 `EXPECT_DEATH` 测试。
- 输出只用 `KZ_LOG_*`，不用 cout、printf、spdlog；推理热路径不输出常规日志，
  缓冲区在加载或首次运行时分配并复用。

## README、示例与测试

- 根 README 用英文且简略，依次只保留项目介绍、Build and test、Use in another CMake project、
  Modules、Backends；后端名称链接到各自项目，表格只列现有内容并随变化更新。
- `examples/<name>.cpp` 对应 `hybridrt_example_<name>`，展示典型用法；
  资源通过命令行传入，图片用 OpenCV imgcodecs 读写。
- 测试放 `tests/<模块>/<源文件>_test.cpp`，根目录只留构建入口，同源场景合并；
  C API 测试放 `tests/c/`，绑定测试放 `bindings/python/tests/`（pytest）和
  `bindings/rust/hybridrt/tests/`（cargo test），均经 CTest 运行。
  用例组为源文件大驼峰名加 Test，目标为 `hybridrt_test_<源文件>`，重名时加目录前缀。
  命令行程序和示例用 `tests/tools/run_tool.hpp` 做黑盒测试，检查退出码、输出、产物；
  前后处理用构造的小数据验证。
- 真实模型测试用 `tests/engine/template_config.hpp` 的 `template_algo()` 读取模板，
  由 `hybridrt_enable_model_test()` 启用；缺模型、图片或 SDK 时 GTEST_SKIP。
  新增、合并、删除示例或测试时同步 CMake 和文档。

## 提交与验证

- 只有用户明确要求时才提交或推送，不强制覆盖远程历史。
- 标题为 `<type>: <summary>`，简洁英文、动词开头、最多 72 字符、无句号；
  type 只选 feat、fix、perf、chore、doc、test、ci，perf 正文记录前后基准数据。
- 每次提交围绕完整目的，包含相关调用方、构建、测试、示例、文档和规则，能独立构建。
  较大提交正文说明原因、获授权的接口变化、验证结果及未验证平台和后端。
- 提交前审查差异，排除临时文件、第三方代码、模型、docs/、不允许提交的 assets 内容及生成文件
  （上述 hrtc_generated.h 除外）；执行 `git diff --check`，有暂存时也检查 --cached。
- 修改后同步所有受影响文件，运行相关构建和测试，报告未验证的平台和后端。
- 远程验证前同步当前工作树，保留构建和依赖缓存，排除规则只匹配预期目录，避免遗漏源码。
  使用固定验证工作目录，不另建源码副本；重新配置和构建后测试，不以旧二进制结果代替当前源码验证。
  不降低项目或依赖的 CMake 最低版本来绕过失败；依赖下载失败时使用版本匹配的本地副本。
- CUDA 预处理并发相关改动需重复运行
  `ImageModelTest.KeepsCudaPreprocessorScratchAliveUntilStreamCompletes` 20 次，再运行完整 CTest；
  TensorRT 测试设置正确的 SDK 库路径。
- RKNN 板端验证先重新交叉构建，再同步当前产物；从验证目录运行并配置动态库路径。
  需读取源码文件的测试保持相同目录结构；密码向用户确认，不写入仓库。
  新模型先用 rknn_query 查询输入输出形状及量化参数，并确认模型内置 mean/std、rgb2bgr。
  转换工具包与板端运行时版本保持一致；使用 `assets/test/scripts/rknn_convert.py`，
  无 --dataset 为 fp16，带校准图片列表为 int8，mean/std 写入模型，HybridRT 配 image_input 和 inter_rgb。
