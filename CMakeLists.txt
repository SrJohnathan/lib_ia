cmake_minimum_required(VERSION 4.2)
project(lib_ia)

set(CMAKE_CXX_STANDARD 20)

# stable-diffusion.cpp uses long tensor names (notably Qwen/Z-Image text
# encoder tensors). This must be consistent across every GGML consumer because
# GGML_MAX_NAME changes the ggml_tensor ABI.
add_compile_definitions(GGML_MAX_NAME=128)

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
option(LIBIA_BUILD_TEST_INFERENCIA "Build the lib_ia test inference helper" ON)

add_subdirectory(vendor/llama.cpp)
add_subdirectory(vendor/whisper.cpp)
if(LIBIA_BUILD_TEST_INFERENCIA)
    add_subdirectory(vendor/test-inferencia)
endif()

set(SDCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/stable-diffusion.cpp)
file(GLOB SDCPP_SOURCES CONFIGURE_DEPENDS
    "${SDCPP_DIR}/src/*.h"
    "${SDCPP_DIR}/src/*.cpp"
    "${SDCPP_DIR}/src/*.hpp"
    "${SDCPP_DIR}/src/conditioning/*.h"
    "${SDCPP_DIR}/src/conditioning/*.cpp"
    "${SDCPP_DIR}/src/conditioning/*.hpp"
    "${SDCPP_DIR}/src/core/*.h"
    "${SDCPP_DIR}/src/core/*.cpp"
    "${SDCPP_DIR}/src/core/*.hpp"
    "${SDCPP_DIR}/src/extensions/*.h"
    "${SDCPP_DIR}/src/extensions/*.cpp"
    "${SDCPP_DIR}/src/extensions/*.hpp"
    "${SDCPP_DIR}/src/model/*/*.h"
    "${SDCPP_DIR}/src/model/*/*.cpp"
    "${SDCPP_DIR}/src/model/*/*.hpp"
    "${SDCPP_DIR}/src/runtime/*.h"
    "${SDCPP_DIR}/src/runtime/*.cpp"
    "${SDCPP_DIR}/src/runtime/*.hpp"
    "${SDCPP_DIR}/src/model_io/*.h"
    "${SDCPP_DIR}/src/model_io/*.cpp"
    "${SDCPP_DIR}/src/tokenizers/*.h"
    "${SDCPP_DIR}/src/tokenizers/*.cpp"
    "${SDCPP_DIR}/src/tokenizers/vocab/*.h"
    "${SDCPP_DIR}/src/tokenizers/vocab/*.cpp"
)

add_library(sd_zip OBJECT
    ${SDCPP_DIR}/thirdparty/zip.c
)
target_include_directories(sd_zip PUBLIC
    ${SDCPP_DIR}/thirdparty
)
set_property(TARGET sd_zip PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(stable-diffusion STATIC
    ${SDCPP_SOURCES}
)
target_compile_definitions(stable-diffusion PRIVATE
    GGML_MAX_NAME=128
    SDCPP_BUILD_COMMIT=lib_ia_vendor
    SDCPP_BUILD_VERSION=lib_ia_vendor
)
target_include_directories(stable-diffusion PUBLIC
    ${SDCPP_DIR}/include
    ${SDCPP_DIR}/src
    ${SDCPP_DIR}/thirdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/ggml/src
)
target_include_directories(stable-diffusion PRIVATE
    ${SDCPP_DIR}/src/core
)
target_link_libraries(stable-diffusion PUBLIC ggml sd_zip)
set_property(TARGET stable-diffusion PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(lib_ia SHARED
    src/library.cpp
    src/integration.cpp
    src/sst.cpp
    src/runtime.cpp
    src/image-create.cpp
)

target_compile_definitions(lib_ia PRIVATE LIBIA_BUILD)
target_include_directories(lib_ia PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp/tools/mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/whisper.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/whisper.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/stable-diffusion.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/stable-diffusion.cpp/thirdparty
)
target_link_libraries(lib_ia PRIVATE llama-common mtmd whisper stable-diffusion)
