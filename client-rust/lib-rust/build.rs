use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=LIBIA_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=LIBIA_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=LIBIA_SKIP_CPP_BUILD");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.join("../../");
    let source_dir = env::var("LIBIA_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root.clone());

    if let Some(build_dir) = existing_build_dir(&workspace_root) {
        emit_link_directives(&build_dir);
        return;
    }

    if skip_cpp_build() {
        panic!(
            "LIBIA_SKIP_CPP_BUILD is set, but no prebuilt lib_ia was found. Set LIBIA_BUILD_DIR or disable LIBIA_SKIP_CPP_BUILD."
        );
    }

    if !source_dir.join("CMakeLists.txt").exists() {
        panic!(
            "lib_ia CMakeLists.txt not found. Set LIBIA_SOURCE_DIR to the lib_ia root or set LIBIA_BUILD_DIR to a prebuilt library."
        );
    }

    rerun_if_changed(&source_dir);

    let build_dir = cargo_build_dir(&manifest_dir);
    fs::create_dir_all(&build_dir).expect("failed to create build dir");

    configure_cpp(&source_dir, &build_dir);
    build_cpp(&build_dir);

    emit_link_directives(&build_dir);
}

fn skip_cpp_build() -> bool {
    matches!(
        env::var("LIBIA_SKIP_CPP_BUILD").as_deref(),
        Ok("1") | Ok("true") | Ok("yes") | Ok("on")
    )
}

fn existing_build_dir(workspace_root: &Path) -> Option<PathBuf> {
    env::var("LIBIA_BUILD_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|dir| has_library(dir))
        .or_else(|| {
            let default_build = workspace_root.join("build");
            if has_library(&default_build) {
                Some(default_build)
            } else {
                None
            }
        })
}

fn has_library(dir: &Path) -> bool {
    let candidates = [
        dir.join("liblib_ia.so"),
        dir.join("liblib_ia.dylib"),
        dir.join("liblib_ia.dll"),
        dir.join("Release").join("liblib_ia.so"),
        dir.join("Release").join("liblib_ia.dylib"),
        dir.join("Release").join("liblib_ia.dll"),
    ];

    candidates.iter().any(|path| path.exists())
}

fn emit_link_directives(build_dir: &Path) {
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("Release").display()
    );
    println!("cargo:rustc-link-lib=dylib=lib_ia");
}

fn cargo_build_dir(manifest_dir: &Path) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let mut build_dir = out_dir;
    build_dir.push("cpp-build");
    build_dir.push(
        manifest_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("lib-rust"),
    );
    build_dir
}

fn configure_cpp(source_dir: &Path, build_dir: &Path) {
    run_command(
        Command::new("cmake")
            .arg("-S")
            .arg(source_dir)
            .arg("-B")
            .arg(build_dir)
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg("-DLIBIA_BUILD_TEST_INFERENCIA=OFF")
            .arg("-DGGML_CUDA=OFF")
            .arg("-DGGML_VULKAN=ON"),
        "configure",
    );
}

fn build_cpp(build_dir: &Path) {
    run_command(
        Command::new("cmake")
            .arg("--build")
            .arg(build_dir)
            .arg("--config")
            .arg("Release")
            .arg("--target")
            .arg("lib_ia"),
        "build",
    );
}

fn run_command(cmd: &mut Command, step: &str) {
    let status = cmd.status().unwrap_or_else(|err| {
        panic!("failed to spawn cmake {step} command: {err}");
    });

    if !status.success() {
        panic!("cmake {step} failed with status {status}");
    }
}

fn rerun_if_changed(root: &Path) {
    for path in [
        root.join("CMakeLists.txt"),
        root.join("src"),
        root.join("vendor/llama.cpp"),
        root.join("vendor/whisper.cpp"),
        root.join("vendor/stable-diffusion.cpp"),
        root.join("vendor/test-inferencia"),
    ] {
        walk(&path);
    }
}

fn walk(path: &Path) {
    if !path.exists() {
        return;
    }

    if path.is_file() {
        println!("cargo:rerun-if-changed={}", path.display());
        return;
    }

    println!("cargo:rerun-if-changed={}", path.display());
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            walk(&entry.path());
        }
    }
}
