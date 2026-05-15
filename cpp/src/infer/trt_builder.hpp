#pragma once
//
// ONNX → TensorRT engine build helper.
//
// Designed for two specific models in this repo:
//   - YOLOX (static batch=1, 1x3x416x416, NMS-in-graph)
//   - RTMPose (dynamic batch 1..3, Bx3x256x192)
//
// Generic enough that other static-shape ONNX models will build fine too;
// dynamic-shape models need to pass an OptimizationProfile.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

namespace fitra::infer {

struct DynamicProfile {
    std::string input_name;
    nvinfer1::Dims min_dims;
    nvinfer1::Dims opt_dims;
    nvinfer1::Dims max_dims;
};

struct BuildOptions {
    std::string onnx_path;
    std::string engine_path;
    bool        fp16           = false;
    bool        int8           = false;   // not used in Phase 1
    std::size_t workspace_mb   = 1024;
    // Optional dynamic-shape profile. Empty input_name => static shapes only.
    std::vector<DynamicProfile> profiles;
};

// Build an engine and write it to disk. Returns the engine size in bytes.
// Throws on failure.
std::size_t build_engine(const BuildOptions& opts, nvinfer1::ILogger& logger);

}  // namespace fitra::infer
