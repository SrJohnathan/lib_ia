use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LIBIA_BUILD_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let build_dir = env::var("LIBIA_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("../../build"));

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir.display());
}
