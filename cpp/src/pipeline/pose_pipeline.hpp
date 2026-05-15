#pragma once
//
// Single-camera capture + decode + YOLOX + RTMPose, mirroring
// python/scripts/pose_pipeline.py::PoseEngine semantics:
//   - detector runs every `det_frequency` frames (cached bboxes between)
//   - single-person mode keeps the largest-area bbox
//   - pose runs on every retained bbox
//
// Designed to be driven from a single thread (capture happens on a
// separate worker inside V4l2Capture). Each call to step() polls the
// capture for the latest frame; if no new frame is available, returns
// std::nullopt so the caller can spin / sleep.

#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "camera/jpeg_decoder.hpp"
#include "camera/v4l2_capture.hpp"
#include "infer/rtmpose.hpp"
#include "infer/types.hpp"
#include "infer/yolox.hpp"

namespace fitra::pipeline {

struct StepResult {
    std::uint64_t seq{0};
    std::chrono::steady_clock::time_point captured_at{};
    std::chrono::steady_clock::time_point processed_at{};
    // BGR frame as decoded; usable for downstream overlay or saving.
    cv::Mat frame_bgr;
    std::vector<infer::Bbox>   bboxes;
    std::vector<infer::Person> persons;
};

struct PipelineStats {
    std::uint64_t processed_count = 0;
    double recent_pose_fps = 0.0;
    double avg_pose_fps    = 0.0;
    double last_stage_ms   = 0.0;
};

struct PipelineOptions {
    int  det_frequency = 10;
    bool single_person = true;
};

class PosePipeline {
public:
    PosePipeline(camera::V4l2Capture& capture,
                 infer::Yolox& yolox,
                 infer::RtmPose& rtmpose,
                 PipelineOptions opts);

    // Try to process the latest frame from the capture. Returns nullopt if
    // no new frame is available.
    std::optional<StepResult> step();

    const PipelineStats& stats() const { return stats_; }
    camera::V4l2Capture&  capture() { return capture_; }

private:
    void update_stats(std::chrono::steady_clock::time_point now,
                      std::chrono::steady_clock::time_point captured_at);

    camera::V4l2Capture& capture_;
    infer::Yolox&        yolox_;
    infer::RtmPose&      rtmpose_;
    PipelineOptions      opts_;

    camera::JpegDecoder  decoder_;
    cv::Mat              decode_scratch_;
    int                  frame_idx_ = 0;
    std::vector<infer::Bbox> cached_bboxes_;

    PipelineStats stats_;
    std::deque<std::chrono::steady_clock::time_point> recent_;
    std::chrono::steady_clock::time_point start_time_{};
};

}  // namespace fitra::pipeline
