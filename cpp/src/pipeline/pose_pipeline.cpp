#include "pipeline/pose_pipeline.hpp"

#include <algorithm>

#include "util/logging.hpp"

namespace fitra::pipeline {

PosePipeline::PosePipeline(camera::V4l2Capture& capture,
                           infer::Yolox& yolox,
                           infer::RtmPose& rtmpose,
                           PipelineOptions opts)
    : capture_{capture},
      yolox_{yolox},
      rtmpose_{rtmpose},
      opts_{std::move(opts)},
      start_time_{std::chrono::steady_clock::now()} {}

std::optional<StepResult> PosePipeline::step() {
    camera::Frame raw;
    if (!capture_.try_pop_latest(raw)) return std::nullopt;

    StepResult sr;
    sr.seq         = raw.seq;
    sr.captured_at = raw.captured_at;

    if (!decoder_.decode(raw.jpeg, decode_scratch_)) {
        FITRA_LOG_WARN("jpeg decode failed for seq={}", raw.seq);
        return std::nullopt;
    }
    sr.frame_bgr = decode_scratch_;  // shallow ref; safe to use until next step

    bool do_detect = (frame_idx_ % opts_.det_frequency == 0) || cached_bboxes_.empty();
    if (do_detect) {
        auto dets = yolox_.infer(sr.frame_bgr);
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
        cached_bboxes_ = std::move(dets);
    }
    sr.bboxes = cached_bboxes_;
    sr.persons = rtmpose_.infer(sr.frame_bgr, cached_bboxes_);
    sr.processed_at = std::chrono::steady_clock::now();
    ++frame_idx_;

    update_stats(sr.processed_at, sr.captured_at);
    return sr;
}

void PosePipeline::update_stats(std::chrono::steady_clock::time_point now,
                                std::chrono::steady_clock::time_point captured_at) {
    ++stats_.processed_count;
    recent_.push_back(now);
    while (recent_.size() > 60) recent_.pop_front();
    if (recent_.size() >= 2) {
        auto span = std::chrono::duration<double>(recent_.back() - recent_.front()).count();
        if (span > 0) stats_.recent_pose_fps = (recent_.size() - 1) / span;
    }
    auto elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed > 0) stats_.avg_pose_fps = stats_.processed_count / elapsed;
    stats_.last_stage_ms = std::chrono::duration<double, std::milli>(now - captured_at).count();
}

}  // namespace fitra::pipeline
