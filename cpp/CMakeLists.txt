cmake_minimum_required(VERSION 3.22)

# CUDA language is enabled later (when we add .cu kernels in Phase 2/4). For
# Phase 0/1 we only need libcudart + TensorRT, which find_package(CUDAToolkit)
# below pulls in without requiring nvcc on PATH.
project(fitra_cam CXX)

# -------- compiler --------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# -------- module path --------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -------- deps --------
# Jetson Orin Nano Super: CUDA 12.6 / TensorRT 10.3 (apt-installed).
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)

# OpenCV (apt: libopencv-dev). We need imgproc + imgcodecs + videoio for the
# correctness check tool; the hot pipeline avoids OpenCV.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

# Threads
find_package(Threads REQUIRED)

# -------- targets --------
add_subdirectory(src)
add_subdirectory(tools)

# -------- summary --------
message(STATUS "")
message(STATUS "fitra_cam build configuration")
message(STATUS "  CMAKE_BUILD_TYPE   = ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX standard       = ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA toolkit       = ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_ROOT})")
message(STATUS "  TensorRT version   = ${TensorRT_VERSION}")
message(STATUS "  TensorRT include   = ${TensorRT_INCLUDE_DIRS}")
message(STATUS "  TensorRT libraries = ${TensorRT_LIBRARIES}")
message(STATUS "  OpenCV version     = ${OpenCV_VERSION}")
message(STATUS "")
