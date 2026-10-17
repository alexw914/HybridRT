# 代码编写规则

适用于仓库自有代码和 CMake 生成头文件的模板；第三方依赖、推理引擎 SDK 及 `build/`
下的生成文件不直接修改。既有公开接口的调整需要明确授权，不为符合新规则而破坏兼容性。

## 项目定位

- HybridRT 是 C++20 模型推理框架，只负责预处理、推理、后处理及串联三者的模型封装。
- 引擎自定义算子库从引擎 `config` 加载（ORT/RKNN `custom_op_libraries`，TensorRT
  `plugin_libraries`），是可选项，加载失败只 `KZ_LOG_WARN` 并跳过。
- 基础能力统一用 kitzoo，不在本仓库重复实现或另行封装，缺少的通用能力补充到 kitzoo：
  JSON 用 `kitzoo::json::Reader` 解析、nlohmann 原生接口读取，参数结构体用
  `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT` 直接映射，`get<T>()` 前先用
  `source/include/core/config.hpp` 的 `check_param` 无异常检查（未知键也是错误）；
  主机内存 `kitzoo::memory::ByteBuffer`；日志 `KZ_LOG_*`；命令行 `kitzoo::core::Options`；
  文件 `kitzoo::os`；字符串工具 `kitzoo::util`；属性 `KZ_NODISCARD`、`KZ_MAYBE_UNUSED`。
  只链接需要的 kitzoo 模块目标。
- 容器、字符串和智能指针在所有代码中用 kitzoo 别名 `Array`、`Vector`、`Map`、`Set`、
  `Pair`、`String`、`UniquePtr`、`SharedPtr`（`include/hrt/core/containers.hpp` 引入 `hrt`，
  命名空间外写 `hrt::Vector`），不写对应的 `std::` 类型；别名即 std 类型，可直接传给第三方
  接口。`containers.hpp` 只由 `define.hpp`、`context.hpp`、`hrtc.hpp`、`config.hpp` 直接包含，
  其余文件不重复包含它或它已引入的 `<vector>`、`<string>` 等标准头文件。
- 链接 `kitzoo::memory` 的可执行文件由 kitzoo 以 mimalloc 替换进程分配器，库代码不指定分配器。

## 目录与模块

- 公开头文件在 `include/hrt/`，实现在 `source/`；私有头文件放 `source/include/<模块>/`，
  按 `"include/<模块>/<文件>.hpp"` 引用，`source/<模块>/` 只放 `.cpp`。后端 SDK 查找脚本在
  `cmake/module/`，第三方依赖在 `cmake/external/`。
- `hrt::hrtc` 不依赖推理运行时；`source/memory/` 放主机内存（含 Linux dma-buffer）与拷贝分发，设备公共代码（context、
  memory、SDK 辅助）放 `source/backend/<device>/`，各任务的解码放 `source/postprocess/<task>/`。
- 后端通过 SDK 根目录启用（`-DORT_ROOT=...` 等，见 `cmake/HybridrtBackends.cmake`）并定义
  `HRT_USE_<BACKEND>`；SDK 头文件和类型不得出现在 `include/hrt/`。新增后端只加
  `source/engine/<backend>_engine.*`、`source/CMakeLists.txt` 一段配置和 `Engine::create` 一个分支；
  新增设备同理扩展 `Memory::create`/`wrap`、`copy_memory`、`Context::create`、
  `ImagePreprocessor::create`。上层流程不出现按后端分支的代码。
- 预处理流程只在 `ImagePreprocessor` 基类写一份，设备子类只实现算子，不支持的步骤由
  `supports()` 返回 false 并退回 CPU。后处理只在主机执行，按 `postprocess.type` 注册。
- `ImageModel`：algo 配置只描述模型（`preprocess`、`postprocess`），每个引擎只读取自己的
  `config`；各阶段实例个数只作为 `create` 的 `ImageModelOptions` 传入，不写进模型包。
- 生成的 `source/include/hrtc/hrtc_generated.h` 随仓库提交、不手改，修改 `fbs/hrtc.fbs` 后用
  `-DHYBRIDRT_BUILD_FLATC=ON` 构建 `hrtc_schema` 重新生成。字段只在表末尾追加；不兼容的改动更换
  `file_identifier` 并重新打包。
- `assets/template/README.md` 说明每种 preprocess、engine、postprocess 类型的作用、全部参数及默认值和可选值，
  以及各平台需要传入的参数（如 RKNN 的 `image_input`、`dst_image_format`）；代码中类型、参数、默认值或平台行为
  变化时，在同一提交中同步更新它和 `hrtc_template.json`。
- `assets/` 下只提交 `template/`。测试、基准和验证用到的每个模型在 `hrtc_template.json` 中有一个以模型命名的
  algo（如 `yolov8n_detect`、`rtmpose_m_pose`），写明该模型的预处理和后处理配置，`data` 为 `assets/<夹具>/` 下的本地
  文件（相对仓库根目录）；其来源、导出或转换方式、sha256 和验证结论写进 `template/README.md` 的 Test models，
  转换脚本放在 `template/`。测试模型、图片、夹具 manifest 和输出结果只放本地、不提交，不另建夹具目录的 README、
  manifest 或 sources 文件；新增或改动模型、测试时只更新 template。
- 引擎推理耗时只由 `tools/hrt-model-bench` 测量；`benchmarks/` 只保留 `preprocess_bench`、
  `postprocess_bench`、`pipeline_bench`。
- 模型文件、`.hrtc`、`assets/` 下除 `template/` 外的文件和下载到 `external/` 的 SDK 不提交；manifest 中的文件
  路径为绝对路径或相对于运行时工作目录。

## 文件头与头文件保护

- 自有文件使用下列英文文件头：`File` 为从仓库根目录开始的路径，`Description` 简述职责，
  续行对齐；不写作者、日期和修改历史。
- 使用 include guard（禁止 `#pragma once`），宏名 `HRT_<目录>_<文件名>_HPP`，全部大写，点换
  下划线，冲突时再加一级目录；`.hpp.in` 按生成的 `.hpp` 命名，模板的 `@变量@` 保持完整（必要时
  用 `// clang-format off/on`）；末尾 `#endif` 注释标明宏名。

```cpp
// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: include/hrt/engine/engine.hpp
// Description: Declares the inference engine interface.
// -----------------------------------------------------------------------------

#ifndef HRT_ENGINE_ENGINE_HPP
#define HRT_ENGINE_ENGINE_HPP

#endif // HRT_ENGINE_ENGINE_HPP
```

## 格式与函数

- 用根目录 `.clang-format` 格式化（`clang-format -i`），检查用 `--dry-run --Werror`；静态检查
  用 `.clang-tidy`（需 `build/` 下的 `compile_commands.json`）。include 分组顺序为 `<hrt/...>`、
  `<kitzoo/...>`、其他 `<...>`、`"..."`；限定符写在类型左侧。非 C++ 文本用 LF，`.bat`、`.cmd` 用 CRLF。
- 函数声明之间空一行，相关重载可成组；函数和有返回值的 lambda 用尾置返回类型。
- 只读的大对象传 `const T&`，修改调用方对象传 `T&`，转移所有权按值或 `T&&`；数值、指针、
  `std::string_view`、`std::span` 按值传递。
- 注释只写代码无法表达的约束、原因、数据布局和选项，不描述改动过程或显而易见的逻辑。
- 不抛异常、不用 `try`/`catch`（含工具、示例和测试）：
  - 初始化（`ImageModel::create`、`create_*_postprocessor` 等）的配置错误、无可用引擎等用
    `KZ_CHECK_MSG` 记录后 abort；
  - 推理过程失败用 `KZ_CHECK_RETURN` 记录后返回空结果（`std::nullopt`、`false`）；
  - 用于挑选的接口（`Engine::create`/`initialize`、`Context::create`、
    `ImagePreprocessor::create`）和 `hrtc::load` 返回空值，由调用方决定；
  - SDK 状态码直接写进上述宏，错误信息取自 SDK；SDK 类型不泄漏到上层；优先用 SDK 的不抛异常
    接口（如 ONNX Runtime C API）；资源回滚用 RAII 或 `KZ_SCOPE_FAIL`；测试用 `EXPECT_DEATH`
    验证 abort。
- 输出只用 `KZ_LOG_*`，不用 `std::cout`、`printf` 或 spdlog；推理热路径不输出常规日志，
  缓冲区在加载或首次运行时分配并复用。
- 头文件只做声明，实现放 `.cpp`；模板和 constexpr 除外。

## 命名

- 命名空间按模块目录：`hrt::core`、`engine`、`preprocess`、`postprocess`、`model`、`hrtc`，
  `source/` 内部的 `memory`、`backend`；跨模块写模块前缀（`core::Image`）；不用 `detail`，
  文件内私有代码放匿名命名空间。
- 类型大驼峰，函数和变量小写下划线，常量 `kName`，宏 `HRT_` 前缀；第三方接口遵循原有约定。
- 枚举用 `enum class`，枚举项大驼峰（`DataType::Float32`），最后一项保留逗号；底层类型和数值
  只在序列化、ABI 等需要时指定。
- 只有持有进程级共享状态的对象才用单例（继承 `kitzoo::core::Singleton<T>`）；无状态工具用自由函数。

## README、示例与测试

- README 用英文、保持简略，依次只保留项目介绍、`Build and test`、`Use in another CMake project`、
  `Modules` 和 `Backends`，后端名称链接到各自项目；表格只列现有内容，变化时同步更新。
- 示例 `examples/<name>.cpp` 对应 `hybridrt_example_<name>`，展示典型用法；资源通过命令行参数
  传入，用 OpenCV imgcodecs 读写图片。
- 测试放 `tests/<module>/<源文件>_test.cpp`（根目录只留构建入口），同一源文件的场景合并到一个
  测试文件；用例组以源文件大驼峰名加 `Test` 命名，目标为 `hybridrt_test_<源文件>`（重名时加目录前缀，如
  `hybridrt_test_example_image_model`）。命令行程序和示例用黑盒测试：运行构建出的程序，检查退出码、
  输出和生成的文件（`tests/tools/run_tool.hpp`）。前后处理用构造的小数据验证；跑真实模型的测试用
  `tests/engine/template_config.hpp` 的 `template_algo()` 读取模板中的模型 algo，由 `hybridrt_enable_model_test()`
  启用，模型、图片或 SDK 缺失时 `GTEST_SKIP()`。
- 新增、合并或删除示例和测试时同步更新 CMake 与文档。

## 提交与验证

- 标题 `<type>: <summary>`，简洁英文、动词开头、不超过 72 字符、无句号；类型为 `feat`、`fix`、
  `perf`（正文记录基准前后数据）、`chore`、`doc`、`test`、`ci`，不混用。
- 提交围绕完整目的，含调用方、构建配置、测试、示例和规则更新，能独立构建；较大提交在正文说明
  改动原因、已授权的接口变化、构建测试结果及未验证的平台和后端。
- 提交前检查差异，不含生成文件、临时文件、第三方代码、模型文件、`assets/` 下除 `template/` 外的文件或 `docs/`；执行
  `git diff --check`（有暂存时加 `--cached`）。
- 只有用户明确要求时才提交或推送；推送不强制覆盖远程历史。
- 修改后同步更新受影响的 CMake、调用方、示例和文档，运行相关构建和测试，报告未验证的平台和后端。

## 远程 TensorRT/CUDA 验证

- GPU 远程主机 SSH 名称为 `gpu4090`，验证工作目录固定为 `/root/WorkSpace/AgentValidation`，源码在其下的 `HybridRT/`；验证过程只使用该目录，不创建临时源码副本或在其他目录构建。
- 每次验证前，从本地仓库根目录将当前工作树同步到远程目录；保留远端 `build/` 和 `deps/` 缓存（排除规则以 `/` 开头，只匹配根目录，避免误排除
  `cmake/external/`）：

  ```sh
  rsync -a --delete --exclude='/.git/' --exclude='/build/' --exclude='/deps/' --exclude='/external/' --exclude='/deps.cmake' --exclude='/spdlog-1.17.0/' ./ gpu4090:/root/WorkSpace/AgentValidation/HybridRT
  ```

- 远端使用 `/root/.local/bin/cmake`（当前为 4.4.3；系统默认 CMake 可能低于项目要求），先检查版本至少为 3.25，再从固定目录配置；TensorRT SDK 根目录为 `/opt/nvidia/TensorRT-10.16.1.11`：

  ```sh
  ssh gpu4090 'cd /root/WorkSpace/AgentValidation/HybridRT && /root/.local/bin/cmake -S . -B build -DTensorRT_ROOT=/opt/nvidia/TensorRT-10.16.1.11 -DHYBRIDRT_BUILD_TESTS=ON'
  ```

- 远端访问 GitHub 不稳定；FetchContent 拉取失败时，在本地按 `cmake/external/` 中的版本克隆，同步到
  `/root/WorkSpace/AgentValidation/deps-local/<name>`，配置时加 `-DFETCHCONTENT_SOURCE_DIR_<NAME>=<该目录>`。

- 构建并重复运行 CUDA 预处理并发回归，然后运行完整 CTest；运行 TensorRT 测试时设置 SDK 库路径：

  ```sh
  ssh gpu4090 'cd /root/WorkSpace/AgentValidation/HybridRT && /root/.local/bin/cmake --build build --target hybridrt_test_image_model -j2'
  ssh gpu4090 'cd /root/WorkSpace/AgentValidation/HybridRT && LD_LIBRARY_PATH=/opt/nvidia/TensorRT-10.16.1.11/lib:$LD_LIBRARY_PATH ./build/tests/hybridrt_test_image_model --gtest_filter=ImageModelTest.KeepsCudaPreprocessorScratchAliveUntilStreamCompletes --gtest_repeat=20 --gtest_break_on_failure'
  ssh gpu4090 'cd /root/WorkSpace/AgentValidation/HybridRT && LD_LIBRARY_PATH=/opt/nvidia/TensorRT-10.16.1.11/lib:$LD_LIBRARY_PATH ctest --test-dir build --output-on-failure'
  ```

- 不降低项目或依赖的 CMake 最低版本来绕过配置失败。若远端 CMake 低于 3.25，先安装或选择符合要求的 CMake，再重新配置和构建；不能把旧 build 中已有的测试二进制运行结果当作当前源码的验证结果。

## RK3588 RKNN 板端验证

- 板端为 Radxa Rock 5B+（Debian 12，librknnrt 2.3.2），SSH `radxa@192.168.5.11`，密码向用户确认，不写进仓库；
  验证目录固定为 `~/hybridrt-test`：`bin/` 放程序，`lib/` 放 `librga.so`，`assets/` 放清单、图片和模型。板端
  Python 只有 numpy 和 `rknnlite`，没有 OpenCV。
- 本地交叉编译使用 Arm GNU Toolchain 12.2 的 `/Users/krisw/Env/rk3588/toolchains/rk3588-gcc12.toolchain.cmake`
  （GCC 10.2 的 `rk3588.toolchain.cmake` 编不过 kitzoo）；第三方依赖用 `external/deps/` 的本地副本：

  ```sh
  E=/Users/krisw/Env/rk3588; D=$PWD/external/deps
  cmake -S . -B build/rk3588-gcc12 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=$E/toolchains/rk3588-gcc12.toolchain.cmake \
      -DRKNN_ROOT=$E/rknpu2 -DRKNN_PLATFORM=RK3588 -DRGA_ROOT=$E/RGA -DOpenCV_DIR=$E/OpenCV/lib/cmake/opencv4 \
      -DHYBRIDRT_BUILD_TESTS=ON -DHYBRIDRT_TOOLS_OUTPUT_DIR=build/rk3588-gcc12/tools \
      -DFETCHCONTENT_SOURCE_DIR_KITZOO=$D/kitzoo-0.5.1 -DFETCHCONTENT_SOURCE_DIR_FLATBUFFERS=$D/flatbuffers-25.2.10 \
      -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$D/googletest-1.18.0 -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=$D/json-3.12.0 \
      -DFETCHCONTENT_SOURCE_DIR_SPDLOG=$D/spdlog-1.17.0 -DFETCHCONTENT_SOURCE_DIR_MIMALLOC=$D/mimalloc-3.5.3 \
      -DFETCHCONTENT_SOURCE_DIR_CXXOPTS=$D/cxxopts-3.3.1 -DFETCHCONTENT_SOURCE_DIR_THREAD_POOL=$D/thread-pool-5.1.0 \
      -DFETCHCONTENT_SOURCE_DIR_CONCURRENTQUEUE=$D/concurrentqueue-1.0.5
  cmake --build build/rk3588-gcc12 --parallel
  ```

- 每次验证都重新构建，再用 tar 经 SSH 同步当前产物，不运行板端旧二进制；模型放 `assets/<名称>/`，清单由模板中
  该模型的 algo 生成（`rknn` 后端），路径相对 `~/hybridrt-test`；从该目录运行并设置 `LD_LIBRARY_PATH=lib`：

  ```sh
  tar czf - -C build/rk3588-gcc12 tests/hybridrt_test_general_keypoint tools/hrt-validate examples/hybridrt_example_image_model \
      | ssh radxa@192.168.5.11 'mkdir -p ~/hybridrt-test/bin && tar xzf - --strip-components=1 -C ~/hybridrt-test/bin'
  ssh radxa@192.168.5.11 'cd ~/hybridrt-test && LD_LIBRARY_PATH=lib ./bin/hybridrt_test_general_keypoint'
  ssh radxa@192.168.5.11 'cd ~/hybridrt-test && LD_LIBRARY_PATH=lib ./bin/hrt-validate -m assets/<名称>/<清单>.json -n <algo> -i assets/yolo/images -r drawn/'
  ```

- 测试程序内嵌源码路径，读取仓库文件的测试需在板端保持相同目录结构；接入新模型前先在板端用 `librknnrt.so` 的
  `rknn_query`（Python ctypes）查询输入输出形状和量化参数，RKNN 模型内置的 mean/std 与 `rgb2bgr` 可用 `strings <模型>.rknn | grep "'mean'"` 查看。
