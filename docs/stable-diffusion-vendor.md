# stable-diffusion.cpp vendor notes

This project vendors `stable-diffusion.cpp` at:

```text
vendor/stable-diffusion.cpp
```

Do not clone it with `--recursive` and do not initialize its GGML submodule.

`stable-diffusion.cpp` normally carries its own `ggml` copy, but `lib_ia`
already builds `vendor/llama.cpp`, which also builds GGML. Linking two GGML
copies in the same shared library can cause ABI drift, duplicate symbols, or
backend/linker conflicts, especially with Vulkan enabled.

The intended build rule is:

```text
stable-diffusion.cpp sources
  -> include/link against llama.cpp's GGML
  -> never build vendor/stable-diffusion.cpp/ggml
```

Current backend policy:

- CPU: enabled through llama.cpp/GGML.
- Vulkan: enabled through llama.cpp/GGML.
- CUDA: disabled.
- stable-diffusion.cpp CMake is not added with `add_subdirectory`.
- `lib_ia` owns the integration target and chooses the include/link order.

If the upstream `stable-diffusion.cpp` source starts requiring GGML APIs not
present in the vendored `llama.cpp`, update both vendors together instead of
adding the SD GGML submodule.

