#pragma once
//
// RTMPose SimCC wrapper around a TrtEngine.
//
// Matches python/scripts/pose_pipeline.py::RtmposeOnnx including the
// 3-point affine warp, ImageNet BGR mean/std normalization, SimCC argmax,
// and inverse-affine remap.
//
// The wrapper batches up to `kMaxBatch` requests per TRT inference (the
// engine is built with profile min=1/opt=1/max=3, matching the project's
// 3-camera goal). Requests beyond that are split across multiple calls.
//
// Spec (body7-256x192):
//   input  : "input"   (B, 3, 256, 192) float32, BGR, ImageNet-normalized
//   output : "simcc_x" (B, K=17, 384)   float32
//   output : "simcc_y" (B, K=17, 512)   float32

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "infer/trt_engine.hpp"
#include "infer/types.hpp"

namespace fitra::infer {

class RtmPose {
public:
    // One pose request: a (frame, bbox) pair. The frame is borrowed; the
    // caller must keep it alive until the corresponding infer_batch() call
    // returns. Mixing frames across cameras is allowed.
    struct Request {
        const cv::Mat* frame = nullptr;
        Bbox           bbox{};
    };

    struct Options {
        std::string input_name   = "input";
        std::string simcc_x_name = "simcc_x";
        std::string simcc_y_name = "simcc_y";
        int   input_w   = 192;
        int   input_h   = 256;
        float padding   = 1.25f;
        float simcc_split = 2.0f;
        int   max_batch = 3;  // matches engine profile max
    };

    explicit RtmPose(TrtEngine& engine);
    RtmPose(TrtEngine& engine, Options opts);

    // Convenience: all bboxes share the same frame (single-camera path).
    std::vector<Person> infer(const cv::Mat& frame_bgr,
                              const std::vector<Bbox>& bboxes);

    // True batched inference. Persons are returned in request order. When
    // `reqs.size() > opts.max_batch`, the engine is invoked multiple times
    // with up to max_batch requests each.
    std::vector<Person> infer_batch(const std::vector<Request>& reqs);

private:
    void preprocess_one(const cv::Mat& frame_bgr,
                        const Bbox& bb,
                        float* dst_chw,        // points to b * 3*H*W in input_blob_
                        cv::Mat& M_inv_out);

    void run_one_batch(const Request* reqs,
                       std::size_t n,
                       std::vector<Person>& out);

    TrtEngine& engine_;
    Options    opts_;

    // Reusable host buffers; sized for max_batch.
    cv::Mat warp_;                       // 256x192x3 uint8
    cv::Mat warp_f_;                     // 256x192x3 float32 (normalized)
    std::vector<float> input_blob_;      // B*3*H*W CHW float32
    std::vector<float> simcc_x_host_;    // B*K*Wx
    std::vector<float> simcc_y_host_;    // B*K*Wy
};

}  // namespace fitra::infer
