# FindTensorRT.cmake
#
# Locate NVIDIA TensorRT installed via apt on JetPack (libnvinfer-dev etc).
# On Jetson Orin Nano Super / JetPack 6.2.1 the headers live in
# /usr/include/aarch64-linux-gnu and the libs in /usr/lib/aarch64-linux-gnu.
#
# Result variables:
#   TensorRT_FOUND            - whether TRT was located
#   TensorRT_VERSION          - "<major>.<minor>.<patch>"
#   TensorRT_INCLUDE_DIRS     - include path containing NvInfer.h
#   TensorRT_LIBRARIES        - list of TRT libs to link (nvinfer + plugin)
#
# Imported target:
#   TensorRT::TensorRT        - INTERFACE target bundling include + libs

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        /usr/include/aarch64-linux-gnu
        /usr/local/cuda/include
        /usr/include
    DOC "Directory containing NvInfer.h"
)

find_library(TensorRT_NVINFER_LIB
    NAMES nvinfer
    HINTS
        /usr/lib/aarch64-linux-gnu
        /usr/local/cuda/lib64
        /usr/lib
    DOC "Path to libnvinfer.so"
)

find_library(TensorRT_NVINFER_PLUGIN_LIB
    NAMES nvinfer_plugin
    HINTS
        /usr/lib/aarch64-linux-gnu
        /usr/local/cuda/lib64
        /usr/lib
    DOC "Path to libnvinfer_plugin.so"
)

find_library(TensorRT_NVONNXPARSER_LIB
    NAMES nvonnxparser
    HINTS
        /usr/lib/aarch64-linux-gnu
        /usr/local/cuda/lib64
        /usr/lib
    DOC "Path to libnvonnxparser.so (needed for ONNX -> engine build)"
)

# Parse version from NvInferVersion.h if available
if(TensorRT_INCLUDE_DIR)
    if(EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
        file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver_lines
            REGEX "#define[ \t]+NV_TENSORRT_(MAJOR|MINOR|PATCH|BUILD)[ \t]+[0-9]+")
        foreach(_line IN LISTS _trt_ver_lines)
            if(_line MATCHES "NV_TENSORRT_MAJOR[ \t]+([0-9]+)")
                set(TensorRT_VERSION_MAJOR "${CMAKE_MATCH_1}")
            elseif(_line MATCHES "NV_TENSORRT_MINOR[ \t]+([0-9]+)")
                set(TensorRT_VERSION_MINOR "${CMAKE_MATCH_1}")
            elseif(_line MATCHES "NV_TENSORRT_PATCH[ \t]+([0-9]+)")
                set(TensorRT_VERSION_PATCH "${CMAKE_MATCH_1}")
            endif()
        endforeach()
        if(TensorRT_VERSION_MAJOR)
            set(TensorRT_VERSION
                "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIB TensorRT_NVINFER_PLUGIN_LIB
    VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
    set(TensorRT_LIBRARIES
        "${TensorRT_NVINFER_LIB}"
        "${TensorRT_NVINFER_PLUGIN_LIB}"
    )
    if(TensorRT_NVONNXPARSER_LIB)
        list(APPEND TensorRT_LIBRARIES "${TensorRT_NVONNXPARSER_LIB}")
    endif()

    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT INTERFACE IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${TensorRT_LIBRARIES}"
        )
    endif()
endif()

mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIB
    TensorRT_NVINFER_PLUGIN_LIB
    TensorRT_NVONNXPARSER_LIB
)
