// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt-sys/build.rs
// Description: Links hybridrt_c from HYBRIDRT_LIB_DIR, or builds it from the
//              repository with CMake.
// -----------------------------------------------------------------------------

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=HYBRIDRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=HYBRIDRT_CMAKE_ARGS");
    let lib_dir = match env::var_os("HYBRIDRT_LIB_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => build_from_source(),
    };
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=hybridrt_c");
    // Dependents read it as DEP_HYBRIDRT_C_LIB_DIR to find the library at run time.
    println!("cargo:lib_dir={}", lib_dir.display());
}

/// Builds only the hybridrt_c target; HYBRIDRT_CMAKE_ARGS holds extra
/// whitespace-separated CMake arguments such as -DORT_ROOT=<sdk>.
fn build_from_source() -> PathBuf {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../..");
    println!("cargo:rerun-if-changed={}", root.join("include/hrt.h").display());
    let mut config = cmake::Config::new(&root);
    config
        .define("HYBRIDRT_BUILD_C_API", "ON")
        .define("HYBRIDRT_BUILD_TOOLS", "OFF")
        .define("HYBRIDRT_BUILD_EXAMPLES", "OFF")
        .build_target("hybridrt_c");
    if let Ok(args) = env::var("HYBRIDRT_CMAKE_ARGS") {
        for arg in args.split_whitespace() {
            config.configure_arg(arg);
        }
    }
    let source = config.build().join("build").join("source");
    // Multi-config generators (Visual Studio, Xcode) add a configuration directory.
    let profile = source.join(config.get_profile());
    if profile.is_dir() {
        profile
    } else {
        source
    }
}
