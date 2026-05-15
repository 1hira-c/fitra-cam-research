#pragma once
//
// N-camera driver:
//   - N V4l2Capture instances (each runs its own capture thread)
//   - 1 inference thread that round-robins through cameras, sharing
//     a single Yolox and RtmPose (one TRT execution context each)
//   - SnapshotBus for the publisher to read from
//
// This deliberately mirrors the Python dual_rtmpose_web.py shape so the
// existing web/dual_rtmpose/ frontend can be served unchanged.

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

#include "camera/jpeg_decoder.hpp"
#include "camera/v4l2_capture.hpp"
#include "infer/rtmpose.hpp"
#include "infer/yolox.hpp"
#include "pipeline/pose_pipeline.hpp"
#include "pipeline/snapshot.hpp"

namespace fitra::pipeline {

class MultiCameraDriver {
public:
    struct Options {
        int  det_frequency = 10;
        bool single_person = true;
    };

    MultiCameraDriver(std::vector<std::unique_ptr<camera::V4l2Capture>> caps,
                      infer::Yolox& yolox,
                      infer::RtmPose& rtmpose,
                      SnapshotBus& bus,
                      Options opts);
    ~MultiCameraDriver();

    MultiCameraDriver(const MultiCameraDriver&) = delete;
    MultiCameraDriver& operator=(const MultiCameraDriver&) = delete;

    void start();
    void stop();

    std::size_t camera_count() const { return caps_.size(); }
    const PipelineStats& stats_for(std::size_t i) const { return per_cam_[i].stats; }

private:
    struct CamState {
        camera::JpegDecoder  decoder;
        cv::Mat              frame;
        int                  frame_idx = 0;
        std::vector<infer::Bbox> cached_bboxes;
        PipelineStats        stats;
        std::deque<std::chrono::steady_clock::time_point> recent;
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    };

    void loop();
    void process_one(std::size_t cam_idx);
    void update_stats(CamState& cs,
                      std::chrono::steady_clock::time_point now,
                      std::chrono::steady_clock::time_point captured_at);

    std::vector<std::unique_ptr<camera::V4l2Capture>> caps_;
    infer::Yolox&        yolox_;
    infer::RtmPose&      rtmpose_;
    SnapshotBus&         bus_;
    Options              opts_;

    std::vector<CamState> per_cam_;
    std::thread           worker_;
    std::atomic<bool>     stop_{false};
};

}  // namespace fitra::pipeline
