# Inference library (TRT engine wrapper, ONNX -> engine builder, YOLOX, RTMPose).
add_library(fitra_infer STATIC
    infer/trt_engine.cpp
    infer/trt_builder.cpp
    infer/yolox.cpp
    infer/rtmpose.cpp
)

target_include_directories(fitra_infer PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(fitra_infer PUBLIC
    CUDA::cudart
    TensorRT::TensorRT
    Threads::Threads
    opencv_core
    opencv_imgproc
)

# Top-level `main` binary (Phase 0 skeleton — will grow into the pipeline
# driver as later phases land).
add_executable(fitra_cam_main
    main.cpp
)

target_link_libraries(fitra_cam_main PRIVATE
    fitra_infer
)

set_target_properties(fitra_cam_main PROPERTIES
    OUTPUT_NAME "main"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
