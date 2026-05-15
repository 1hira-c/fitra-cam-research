#pragma once
//
// N-camera driver.
//
// Each camera has its own FrameSource (own V4L2 thread + own decode/YOLOX
// thread with its own TRT execution context). This central driver only
// polls the per-camera ready slots, batches the resulting (frame, bbox)
// requests into one RTMPose call, and distributes the persons back to
// per-camera snapshots.
//
// The Yolox / Yolox-engine plumbing now lives in main.cpp (constructs one
// shared ICudaEngine for YOLOX and N per-camera Yolox/IExecutionContexts)
// and inside FrameSource, not here.

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

#include "camera/frame_source.hpp"
#include "infer/rtmpose.hpp"
#include "pipeline/pose_pipeline.hpp"
#include "pipeline/snapshot.hpp"

namespace fitra::pipeline {

class MultiCameraDriver {
public:
    MultiCameraDriver(std::vector<std::unique_ptr<camera::FrameSource>> sources,
                      infer::RtmPose& rtmpose,
                      SnapshotBus& bus);
    ~MultiCameraDriver();

    MultiCameraDriver(const MultiCameraDriver&) = delete;
    MultiCameraDriver& operator=(const MultiCameraDriver&) = delete;

    void start();
    void stop();

    std::size_t camera_count() const { return sources_.size(); }
    const PipelineStats& stats_for(std::size_t i) const { return per_cam_[i].stats; }

private:
    struct CamState {
        PipelineStats        stats;
        std::deque<std::chrono::steady_clock::time_point> recent;
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    };

    void loop();
    void update_stats(CamState& cs,
                      std::chrono::steady_clock::time_point now,
                      std::chrono::steady_clock::time_point captured_at);

    std::vector<std::unique_ptr<camera::FrameSource>> sources_;
    infer::RtmPose&      rtmpose_;
    SnapshotBus&         bus_;

    // Latest decoded frame + bboxes per camera, kept alive across the
    // RTMPose batched call so we can hand cv::Mat pointers into reqs.
    std::vector<camera::DecodedFrame> latest_per_cam_;
    std::vector<CamState>             per_cam_;
    std::thread                       worker_;
    std::atomic<bool>                 stop_{false};
};

}  // namespace fitra::pipeline
