# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(HEAR VERSION 1.0.0 LANGUAGES CXX)

# Executable will be in ../bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HEAR_SOURCE_DIR}/bin)

add_compile_options(-O2)
add_compile_options(-std=c++17)
add_compile_options(-march=native)
add_compile_options(-g)

add_executable(test "src/main_HEAR.cpp" "src/param.cpp" "src/utils.cpp" "src/thread.cpp" "src/HECNNencryptor_conv1d.cpp" "src/HECNNevaluator_conv1d.cpp" "src/HECNNencryptor_conv2d.cpp" "src/HECNNevaluator_conv2d.cpp" "src/Test_HEAR.cpp")

add_executable(sota "src/main_sota.cpp" "src/param.cpp" "src/utils.cpp" "src/thread.cpp" "src/HECNNevaluator_sota.cpp" "src/HECNNencryptor_conv2d.cpp" "src/HECNNevaluator_conv2d.cpp" "src/Test_ngraph.cpp" "src/Test_lola.cpp")

add_executable(accuracy "src/main_accuracy.cpp" "src/param.cpp" "src/utils.cpp")
 
# Import Microsoft SEAL
find_package(SEAL 3.4.5 EXACT REQUIRED
    PATHS external/seal/native/src/cmake
)

# Import OpenMP 
find_package(OpenMP)
add_definitions("-fopenmp")

# Link Microsoft SEAL
target_link_libraries(test SEAL::seal)
target_link_libraries(sota SEAL::seal ${OpenMP_CXX_LIBRARIES})
