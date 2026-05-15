# Standalone CLI tools that consume the fitra_infer library.

add_executable(fitra_build_engines
    build_engines.cpp
)

target_link_libraries(fitra_build_engines PRIVATE
    fitra_infer
)

set_target_properties(fitra_build_engines PROPERTIES
    OUTPUT_NAME "build_engines"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
)

# Phase 1 correctness driver: video -> JSONL of per-frame YOLOX/RTMPose results.
add_executable(fitra_dump_keypoints
    dump_keypoints.cpp
)

target_link_libraries(fitra_dump_keypoints PRIVATE
    fitra_infer
    opencv_videoio
    opencv_imgproc
    opencv_core
)

set_target_properties(fitra_dump_keypoints PROPERTIES
    OUTPUT_NAME "dump_keypoints"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
)
