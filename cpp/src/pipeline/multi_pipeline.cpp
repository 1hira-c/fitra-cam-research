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
    std::vector<PendingCam>                       pending;
    std::vector<infer::RtmPose::PrebakedRequest>  reqs;

    // Rolling stage breakdown (debug aid; prints every ~3s of work).
    int    iter_count = 0;
    double sum_poll_ms = 0.0, sum_rtm_ms = 0.0, sum_snap_ms = 0.0;
    int    sum_reqs = 0;
    auto   stats_anchor = std::chrono::steady_clock::now();

    while (!stop_.load()) {
        pending.clear();
        reqs.clear();
        auto iter_start = std::chrono::steady_clock::now();

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
            // FrameSource has already pre-baked the RTMPose inputs.
            const auto& cam_df = latest_per_cam_[i];
            const std::size_t per_item =
                cam_df.chw_concat.empty()
                    ? 0
                    : cam_df.chw_concat.size() / std::max<std::size_t>(1, cam_df.bboxes.size());
            for (std::size_t bi = 0; bi < cam_df.bboxes.size(); ++bi) {
                infer::RtmPose::PrebakedRequest pr;
                pr.chw   = cam_df.chw_concat.data() + bi * per_item;
                pr.M_inv = cam_df.M_invs[bi];
                pr.bbox  = cam_df.bboxes[bi];
                reqs.push_back(pr);
            }
            pending.push_back(pc);
        }

        if (pending.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        auto t_after_poll = std::chrono::steady_clock::now();

        // Pass 2: one batched RTMPose call across all cameras' bboxes.
        // Preprocess already ran on per-camera worker threads — this is
        // just memcpy + GPU enqueue + sync + SimCC decode.
        std::vector<infer::Person> all_persons;
        if (!reqs.empty()) {
            all_persons = rtmpose_.infer_prebaked(reqs);
        }
        auto t_after_rtm = std::chrono::steady_clock::now();

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
        auto t_after_snap = std::chrono::steady_clock::now();

        ++iter_count;
        sum_poll_ms += std::chrono::duration<double, std::milli>(t_after_poll  - iter_start).count();
        sum_rtm_ms  += std::chrono::duration<double, std::milli>(t_after_rtm   - t_after_poll).count();
        sum_snap_ms += std::chrono::duration<double, std::milli>(t_after_snap  - t_after_rtm).count();
        sum_reqs    += static_cast<int>(reqs.size());

        auto elapsed = std::chrono::duration<double>(t_after_snap - stats_anchor).count();
        if (elapsed >= 3.0) {
            double iter_ms = (sum_poll_ms + sum_rtm_ms + sum_snap_ms) / iter_count;
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "breakdown iter_ms=%.2f poll=%.2f rtm=%.2f (per-cam avg %.2f reqs) snap=%.2f",
                          iter_ms,
                          sum_poll_ms / iter_count,
                          sum_rtm_ms  / iter_count,
                          static_cast<double>(sum_reqs) / iter_count,
                          sum_snap_ms / iter_count);
            FITRA_LOG_INFO("{}", buf);
            iter_count = 0;
            sum_poll_ms = sum_rtm_ms = sum_snap_ms = 0.0;
            sum_reqs = 0;
            stats_anchor = t_after_snap;
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
