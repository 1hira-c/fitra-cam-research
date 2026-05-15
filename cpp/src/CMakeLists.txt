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

# Camera capture + JPEG decode + per-camera decode worker.
# Phase 4d parallelizes CPU decode across cameras via FrameSource.
# Full GPU NVJPEG / NvBuffer is a Phase 5 consideration.
add_library(fitra_camera STATIC
    camera/v4l2_capture.cpp
    camera/jpeg_decoder.cpp
    camera/frame_source.cpp
)

target_include_directories(fitra_camera PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(fitra_camera PUBLIC
    fitra_infer       # FrameSource owns a Yolox (per-camera context)
    Threads::Threads
    opencv_core
    opencv_imgcodecs
)

# Pipeline glue: single-camera helper, multi-camera driver, snapshot bus.
add_library(fitra_pipeline STATIC
    pipeline/pose_pipeline.cpp
    pipeline/multi_pipeline.cpp
    pipeline/snapshot.cpp
)

target_link_libraries(fitra_pipeline PUBLIC
    fitra_infer
    fitra_camera
)

# Crow HTTP+WebSocket server.
add_library(fitra_web STATIC
    web/crow_server.cpp
)

target_link_libraries(fitra_web PUBLIC
    fitra_pipeline
    Crow::Crow
)

# Top-level driver: N cameras -> shared TRT -> Crow web.
add_executable(fitra_cam_main
    main.cpp
)

target_link_libraries(fitra_cam_main PRIVATE
    fitra_pipeline
    fitra_web
)

set_target_properties(fitra_cam_main PROPERTIES
    OUTPUT_NAME "main"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
