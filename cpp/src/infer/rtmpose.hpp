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

    // One pre-baked pose request: caller already ran the affine warp +
    // normalize + HWC->CHW step on its own thread (typically the per-camera
    // FrameSource worker). RtmPose only batches the GPU inference + decode.
    struct PrebakedRequest {
        const float* chw     = nullptr;  // 3*H*W floats
        cv::Mat      M_inv;              // 2x3 CV_64F inverse affine
        Bbox         bbox{};             // for downstream Person.bbox
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

    // Free-standing preprocess: affine warp + BGR ImageNet normalize +
    // HWC->CHW, writing 3*input_h*input_w floats at `dst_chw`. Safe to call
    // from multiple threads concurrently (no shared scratch).
    static void preprocess_to_blob(const Options& opts,
                                   const cv::Mat& frame_bgr,
                                   const Bbox& bbox,
                                   float* dst_chw,
                                   cv::Mat& M_inv_out);

    static std::size_t blob_floats_per_item(const Options& opts) {
        return static_cast<std::size_t>(3) * opts.input_h * opts.input_w;
    }

    const Options& options() const { return opts_; }

    // Convenience: all bboxes share the same frame (single-camera path).
    // Internally preprocesses (parallel for n>=2) then runs infer_prebaked.
    std::vector<Person> infer(const cv::Mat& frame_bgr,
                              const std::vector<Bbox>& bboxes);

    // True batched inference with preprocessing done internally. For
    // multi-camera live pipelines, prefer infer_prebaked() so that
    // preprocess can run in parallel on the per-camera workers.
    std::vector<Person> infer_batch(const std::vector<Request>& reqs);

    // Batched inference with preprocessed inputs. Persons are returned in
    // request order; the engine is invoked once per opts.max_batch chunk.
    std::vector<Person> infer_prebaked(const std::vector<PrebakedRequest>& reqs);

private:
    void run_one_batch(const Request* reqs,
                       std::size_t n,
                       std::vector<Person>& out);
    void run_one_prebaked(const PrebakedRequest* reqs,
                          std::size_t n,
                          std::vector<Person>& out);

    TrtEngine& engine_;
    Options    opts_;

    // Reusable host buffers; sized for max_batch.
    std::vector<float> input_blob_;      // B*3*H*W CHW float32
    std::vector<float> simcc_x_host_;    // B*K*Wx
    std::vector<float> simcc_y_host_;    // B*K*Wy
};

}  // namespace fitra::infer
