#include "pipeline/multi_pipeline.hpp"

#include <algorithm>
#include <chrono>

#include "util/logging.hpp"

namespace fitra::pipeline {

MultiCameraDriver::MultiCameraDriver(
    std::vector<std::unique_ptr<camera::V4l2Capture>> caps,
    infer::Yolox& yolox,
    infer::RtmPose& rtmpose,
    SnapshotBus& bus,
    Options opts)
    : caps_{std::move(caps)},
      yolox_{yolox},
      rtmpose_{rtmpose},
      bus_{bus},
      opts_{std::move(opts)},
      per_cam_(caps_.size()) {}

MultiCameraDriver::~MultiCameraDriver() {
    try { stop(); } catch (...) {}
}

void MultiCameraDriver::start() {
    for (auto& c : caps_) c->start();
    stop_.store(false);
    worker_ = std::thread{&MultiCameraDriver::loop, this};
    FITRA_LOG_INFO("multi-camera driver started ({} cameras)", caps_.size());
}

void MultiCameraDriver::stop() {
    if (!worker_.joinable() && caps_.empty()) return;
    stop_.store(true);
    if (worker_.joinable()) worker_.join();
    for (auto& c : caps_) c->stop();
}

void MultiCameraDriver::loop() {
    while (!stop_.load()) {
        bool any_work = false;
        for (std::size_t i = 0; i < caps_.size(); ++i) {
            if (stop_.load()) break;
            camera::Frame raw;
            if (!caps_[i]->try_pop_latest(raw)) continue;
            any_work = true;

            auto& cs = per_cam_[i];
            if (!cs.decoder.decode(raw.jpeg, cs.frame)) {
                FITRA_LOG_WARN("cam{}: jpeg decode failed for seq={}", i, raw.seq);
                continue;
            }

            bool do_detect = (cs.frame_idx % opts_.det_frequency == 0)
                          || cs.cached_bboxes.empty();
            if (do_detect) {
                auto dets = yolox_.infer(cs.frame);
                if (opts_.single_person && dets.size() > 1) {
                    auto largest = std::max_element(
                        dets.begin(), dets.end(),
                        [](const auto& a, const auto& b) {
                            float aa = (a.x2 - a.x1) * (a.y2 - a.y1);
                            float bb = (b.x2 - b.x1) * (b.y2 - b.y1);
                            return aa < bb;
                        });
                    infer::Bbox keep = *largest;
                    dets.clear();
                    dets.push_back(keep);
                }
                cs.cached_bboxes = std::move(dets);
            }
            auto persons = rtmpose_.infer(cs.frame, cs.cached_bboxes);

            auto now = std::chrono::steady_clock::now();
            ++cs.frame_idx;
            update_stats(cs, now, raw.captured_at);

            CameraSnapshot snap;
            snap.id = static_cast<int>(i);
            snap.w  = caps_[i]->options().width;
            snap.h  = caps_[i]->options().height;
            snap.seq = raw.seq;
            snap.captured_at = raw.captured_at;
            // Approximate wall time by mapping monotonic now->wall, then
            // subtracting the latency from raw.captured_at.
            auto wall_now = std::chrono::system_clock::now();
            auto lag = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now - raw.captured_at);
            snap.captured_wall = wall_now - lag;
            snap.persons       = std::move(persons);
            snap.bboxes        = cs.cached_bboxes;
            snap.recv_fps        = caps_[i]->recv_fps();
            snap.recent_pose_fps = cs.stats.recent_pose_fps;
            snap.avg_pose_fps    = cs.stats.avg_pose_fps;
            snap.processed       = cs.stats.processed_count;
            std::uint64_t recv = caps_[i]->total_received();
            snap.pending         = recv > snap.processed ? recv - snap.processed : 0;
            snap.stage_ms        = cs.stats.last_stage_ms;
            bus_.update(snap);
        }
        if (!any_work) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
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
