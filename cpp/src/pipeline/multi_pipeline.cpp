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
    : yolox_{yolox},
      rtmpose_{rtmpose},
      bus_{bus},
      opts_{std::move(opts)},
      per_cam_(caps.size()) {
    sources_.reserve(caps.size());
    for (auto& c : caps) {
        sources_.push_back(std::make_unique<camera::FrameSource>(std::move(c)));
    }
}

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
        std::size_t                          idx;
        std::uint64_t                        seq;
        std::chrono::steady_clock::time_point captured_at;
        std::size_t                          person_offset;
        std::size_t                          person_count;
    };
    std::vector<PendingCam> pending;
    std::vector<infer::RtmPose::Request> reqs;

    while (!stop_.load()) {
        pending.clear();
        reqs.clear();

        // Pass 1: poll each camera's decoded frame slot. Decoding itself
        // happens on the per-source worker threads, so this is just a
        // mutex-protected swap.
        for (std::size_t i = 0; i < sources_.size(); ++i) {
            if (stop_.load()) break;
            camera::DecodedFrame df;
            if (!sources_[i]->try_pop_latest_decoded(df)) continue;

            auto& cs = per_cam_[i];
            cs.frame = std::move(df.bgr);

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

            PendingCam pc;
            pc.idx           = i;
            pc.seq           = df.seq;
            pc.captured_at   = df.captured_at;
            pc.person_offset = reqs.size();
            pc.person_count  = cs.cached_bboxes.size();
            for (const auto& bb : cs.cached_bboxes) {
                reqs.push_back(infer::RtmPose::Request{&cs.frame, bb});
            }
            pending.push_back(std::move(pc));
        }

        if (pending.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // Pass 2: one batched RTMPose call.
        std::vector<infer::Person> all_persons;
        if (!reqs.empty()) {
            all_persons = rtmpose_.infer_batch(reqs);
        }

        // Pass 3: distribute persons + update stats + snapshot bus.
        auto wall_now = std::chrono::system_clock::now();
        auto now      = std::chrono::steady_clock::now();
        for (const auto& pc : pending) {
            auto& cs = per_cam_[pc.idx];
            ++cs.frame_idx;
            update_stats(cs, now, pc.captured_at);

            CameraSnapshot snap;
            snap.id  = static_cast<int>(pc.idx);
            snap.w   = sources_[pc.idx]->options().width;
            snap.h   = sources_[pc.idx]->options().height;
            snap.seq = pc.seq;
            snap.captured_at = pc.captured_at;
            auto lag = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now - pc.captured_at);
            snap.captured_wall = wall_now - lag;
            if (pc.person_count > 0) {
                snap.persons.assign(
                    all_persons.begin() + pc.person_offset,
                    all_persons.begin() + pc.person_offset + pc.person_count);
            }
            snap.bboxes          = cs.cached_bboxes;
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
