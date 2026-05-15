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

# Camera capture + JPEG decode (Phase 2 baseline). Phase 4 will swap the
# decode path to GPU NVJPEG.
add_library(fitra_camera STATIC
    camera/v4l2_capture.cpp
    camera/jpeg_decoder.cpp
)

target_include_directories(fitra_camera PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(fitra_camera PUBLIC
    Threads::Threads
    opencv_core
    opencv_imgcodecs
)

# Pipeline glue: capture -> decode -> YOLOX -> RTMPose.
add_library(fitra_pipeline STATIC
    pipeline/pose_pipeline.cpp
)

target_link_libraries(fitra_pipeline PUBLIC
    fitra_infer
    fitra_camera
)

# Top-level `main` binary (Phase 0 probe; will be replaced by the
# full pipeline driver in a later phase).
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
