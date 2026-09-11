cmake_minimum_required(VERSION 3.20)
project(test_image_create CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(LIB_IA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_executable(test-image-create main.cpp)

target_include_directories(test-image-create PRIVATE
    "${LIB_IA_ROOT}/src"
)

add_library(lib_ia_shared SHARED IMPORTED GLOBAL)
set_target_properties(lib_ia_shared PROPERTIES
    IMPORTED_LOCATION "${LIB_IA_ROOT}/build/liblib_ia.so"
)

target_link_libraries(test-image-create PRIVATE lib_ia_shared)

set_target_properties(test-image-create PROPERTIES
    BUILD_RPATH "${LIB_IA_ROOT}/build"
    INSTALL_RPATH "${LIB_IA_ROOT}/build"
)
