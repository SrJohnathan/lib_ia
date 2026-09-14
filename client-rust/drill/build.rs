use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LIBIA_BUILD_DIR");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        return;
    }

    if let Ok(dir) = std::env::var("LIBIA_BUILD_DIR") {
        let path = PathBuf::from(dir);
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", path.display());
    }

    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/../../../build-release");
}