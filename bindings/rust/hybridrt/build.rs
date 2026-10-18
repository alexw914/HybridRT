// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: bindings/rust/hybridrt/build.rs
// Description: Lets the crate's tests find hybridrt_c at run time.
// -----------------------------------------------------------------------------

fn main() {
    // Set by hybridrt-sys, which links the library from this directory.
    if let Ok(dir) = std::env::var("DEP_HYBRIDRT_C_LIB_DIR") {
        let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target != "windows" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}
