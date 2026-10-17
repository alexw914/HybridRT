# HybridRT for Rust

Rust bindings of [HybridRT](../../README.md) over its C API (`include/hrt.h`):
`hybridrt-sys` declares the raw FFI and `hybridrt` wraps it safely. Images are
borrowed 8-bit host pixels; results are views into a reusable `Results`, in
source-image coordinates. `Model` is `Send + Sync` and calls may run from
several threads, each with its own `Results`.

## Build

`hybridrt-sys` links `hybridrt_c` from `HYBRIDRT_LIB_DIR` when it is set, for
example a CMake build with `-DHYBRIDRT_BUILD_RUST=ON`:

```sh
HYBRIDRT_LIB_DIR=<build>/source cargo test --manifest-path bindings/rust/Cargo.toml
```

Without it, the build script builds `hybridrt_c` from this repository with
CMake; pass backend SDK paths and other CMake arguments, separated by spaces, in
`HYBRIDRT_CMAKE_ARGS`, e.g. `HYBRIDRT_CMAKE_ARGS="-DORT_ROOT=<onnxruntime-sdk>"`.
Executables need the library and the backend SDK libraries on the loader path
at run time (`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, `PATH`, or an rpath); the
crate's own tests get an rpath to the linked directory.

## Usage

```rust
use hybridrt::{Image, ImageFormat, Model, ModelOptions, Package};

let package = Package::load("model.hrtc")?;
let model = Model::new(&package, Some("yolov8n_detect"), ModelOptions { engine: 2, ..Default::default() })?;
let image = Image::new(&bgr, ImageFormat::InterBgr, width, height)?;
let results = model.run(&image, None)?;
for object in results.boxes() {
    println!("{} {} {} {} {} {}", object.class_id, object.score, object.x, object.y, object.width, object.height);
}
```

| Task | Accessors of `Results` |
| --- | --- |
| Detect | `boxes()` |
| Rotated | `rotated_boxes()`: center, size and clockwise angle in degrees |
| Keypoint | `boxes()`, `keypoints(index)` |
| Segment | `boxes()`, `mask(index)`: rect and row-major pixels, 1 marks the class |
| Classify | `scores(output)`: one slice per engine output |

`Image::with_row_pitch` takes padded rows; image bounds are checked before the
call. Errors carry a `Status` and the library's message. Library logs go to the
console; `set_log_callback` forwards them. An engine whose outputs do not fit
the postprocess config still aborts the process, as in C++.
