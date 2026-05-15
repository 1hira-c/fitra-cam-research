#pragma once
//
// RTMPose SimCC wrapper around a TrtEngine.
//
// Matches python/scripts/pose_pipeline.py::RtmposeOnnx including the
// 3-point affine warp, ImageNet BGR mean/std normalization, SimCC argmax,
// and inverse-affine remap. Phase 1 runs batch=1 (one bbox at a time);
// Phase 3 will batch up to 3 persons per call.
//
// Spec (body7-256x192):
//   input  : "input"   (B, 3, 256, 192) float32, BGR, ImageNet-normalized
//   output : "simcc_x" (B, K=17, 384)   float32
//   output : "simcc_y" (B, K=17, 512)   float32

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "infer/trt_engine.hpp"
#include "infer/types.hpp"

namespace fitra::infer {

class RtmPose {
public:
    struct Options {
        std::string input_name   = "input";
        std::string simcc_x_name = "simcc_x";
        std::string simcc_y_name = "simcc_y";
        int   input_w = 192;
        int   input_h = 256;
        float padding = 1.25f;
        float simcc_split = 2.0f;
    };

    explicit RtmPose(TrtEngine& engine);
    RtmPose(TrtEngine& engine, Options opts);

    // Run pose inference for each bbox (batch=1 per call internally for
    // Phase 1). Returns one Person per input bbox in input order.
    std::vector<Person> infer(const cv::Mat& frame_bgr,
                              const std::vector<Bbox>& bboxes);

private:
    void preprocess_one(const cv::Mat& frame_bgr,
                        const Bbox& bb,
                        cv::Mat& M_inv_out);

    TrtEngine& engine_;
    Options    opts_;

    // Reusable host buffers
    cv::Mat warp_;                       // 256x192x3 uint8
    cv::Mat warp_f_;                     // 256x192x3 float32 (normalized)
    std::vector<float> input_blob_;      // CHW float32
    std::vector<float> simcc_x_host_;
    std::vector<float> simcc_y_host_;
};

}  // namespace fitra::infer
