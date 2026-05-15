#include "pipeline/multi_pipeline.hpp"

#include <chrono>

#include "util/logging.hpp"

namespace fitra::pipeline {

MultiCameraDriver::MultiCameraDriver(
    std::vector<std::unique_ptr<camera::FrameSource>> sources,
    infer::RtmPose& rtmpose,
    SnapshotBus& bus)
    : sources_{std::move(sources)},
      rtmpose_{rtmpose},
      bus_{bus},
      latest_per_cam_(sources_.size()),
      per_cam_(sources_.size()) {}

MultiCameraDriver::~MultiCameraDriver() {
    try { stop(); } catch (...) {}
}

void MultiCameraDriver::start() {
    for (auto& s : sources_) s->start();
    stop_.store(false);
    worker_ = std::thread{&MultiCameraDriver::loop, this};
    FITRA_LOG_INFO("multi-camera driver started ({} cameras)", sources_.size());
}

void MultiCameraDriver::stop() {
    if (!worker_.joinable() && sources_.empty()) return;
    stop_.store(true);
    if (worker_.joinable()) worker_.join();
    for (auto& s : sources_) s->stop();
}

void MultiCameraDriver::loop() {
    struct PendingCam {
        std::size_t   idx;
        std::size_t   person_offset;
        std::size_t   person_count;
    };
    std::vector<PendingCam>              pending;
    std::vector<infer::RtmPose::Request> reqs;

    while (!stop_.load()) {
        pending.clear();
        reqs.clear();

        // Pass 1: pull the latest (frame, bboxes) from each FrameSource.
        // Decode + YOLOX already ran in the per-camera worker thread.
        for (std::size_t i = 0; i < sources_.size(); ++i) {
            if (stop_.load()) break;
            camera::DecodedFrame df;
            if (!sources_[i]->try_pop_latest_decoded(df)) continue;

            latest_per_cam_[i] = std::move(df);

            PendingCam pc;
            pc.idx           = i;
            pc.person_offset = reqs.size();
            pc.person_count  = latest_per_cam_[i].bboxes.size();
            for (const auto& bb : latest_per_cam_[i].bboxes) {
                reqs.push_back(infer::RtmPose::Request{
                    &latest_per_cam_[i].bgr, bb});
            }
            pending.push_back(pc);
        }

        if (pending.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // Pass 2: one batched RTMPose call across all cameras' bboxes.
        std::vector<infer::Person> all_persons;
        if (!reqs.empty()) {
            all_persons = rtmpose_.infer_batch(reqs);
        }

        // Pass 3: distribute + update snapshot bus.
        auto wall_now = std::chrono::system_clock::now();
        auto now      = std::chrono::steady_clock::now();
        for (const auto& pc : pending) {
            auto& cs       = per_cam_[pc.idx];
            const auto& df = latest_per_cam_[pc.idx];
            update_stats(cs, now, df.captured_at);

            CameraSnapshot snap;
            snap.id  = static_cast<int>(pc.idx);
            snap.w   = sources_[pc.idx]->options().width;
            snap.h   = sources_[pc.idx]->options().height;
            snap.seq = df.seq;
            snap.captured_at = df.captured_at;
            auto lag = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now - df.captured_at);
            snap.captured_wall = wall_now - lag;
            if (pc.person_count > 0) {
                snap.persons.assign(
                    all_persons.begin() + pc.person_offset,
                    all_persons.begin() + pc.person_offset + pc.person_count);
            }
            snap.bboxes          = df.bboxes;
            snap.recv_fps        = sources_[pc.idx]->recv_fps();
            snap.recent_pose_fps = cs.stats.recent_pose_fps;
            snap.avg_pose_fps    = cs.stats.avg_pose_fps;
            snap.processed       = cs.stats.processed_count;
            std::uint64_t recv = sources_[pc.idx]->total_received();
            snap.pending         = recv > snap.processed ? recv - snap.processed : 0;
            snap.stage_ms        = cs.stats.last_stage_ms;
            bus_.update(snap);
        }
    }
}

void MultiCameraDriver::update_stats(CamState& cs,
                                     std::chrono::steady_clock::time_point now,
                                     std::chrono::steady_clock::time_point captured_at) {
    ++cs.stats.processed_count;
    cs.recent.push_back(now);
    while (cs.recent.size() > 60) cs.recent.pop_front();
    if (cs.recent.size() >= 2) {
        auto span = std::chrono::duration<double>(cs.recent.back() - cs.recent.front()).count();
        if (span > 0) cs.stats.recent_pose_fps = (cs.recent.size() - 1) / span;
    }
    auto elapsed = std::chrono::duration<double>(now - cs.start_time).count();
    if (elapsed > 0) cs.stats.avg_pose_fps = cs.stats.processed_count / elapsed;
    cs.stats.last_stage_ms = std::chrono::duration<double, std::milli>(now - captured_at).count();
}

}  // namespace fitra::pipeline
