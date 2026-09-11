add_executable(test-inferencia
    main.cpp
)

target_compile_features(test-inferencia PRIVATE cxx_std_20)
target_include_directories(test-inferencia PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)
target_link_libraries(test-inferencia PRIVATE lib_ia)
