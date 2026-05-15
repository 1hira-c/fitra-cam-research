#pragma once
//
// Snapshot bus: per-camera latest YOLOX/RTMPose result, atomically
// readable by the WebSocket publisher.
//
// The bundle JSON schema must match python/scripts/dual_rtmpose_web.py
// so that web/dual_rtmpose/app.js works unchanged.

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "infer/types.hpp"

namespace fitra::pipeline {

struct CameraSnapshot {
    int                                    id = 0;
    int                                    w  = 0;
    int                                    h  = 0;
    std::uint64_t                          seq = 0;
    std::chrono::steady_clock::time_point  captured_at{};
    std::chrono::system_clock::time_point  captured_wall{};  // wall-clock for ts_ms
    std::vector<infer::Person>             persons;
    std::vector<infer::Bbox>               bboxes;
    // Stats (mirror python dual_rtmpose_web.py "stats" object)
    double      recv_fps         = 0.0;
    double      recent_pose_fps  = 0.0;
    double      avg_pose_fps     = 0.0;
    std::uint64_t processed      = 0;
    std::uint64_t pending        = 0;
    double      stage_ms         = 0.0;
};

class SnapshotBus {
public:
    explicit SnapshotBus(std::size_t n_cameras);

    // Atomically replace the snapshot for cam_id.
    void update(const CameraSnapshot& s);

    // Build the JSON bundle as a single string. seq increments on each call.
    // Schema:
    //   {"seq":N,"ts_ms":int,"cameras":[{...},{...}]}
    std::string make_bundle_json();

    // Number of cameras the bus was sized for.
    std::size_t size() const { return snapshots_.size(); }

private:
    mutable std::mutex          mu_;
    std::vector<CameraSnapshot> snapshots_;
    std::uint64_t               bundle_seq_ = 0;
};

}  // namespace fitra::pipeline
