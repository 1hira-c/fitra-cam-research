#include "camera/frame_source.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "util/logging.hpp"

namespace fitra::camera {

FrameSource::FrameSource(std::unique_ptr<V4l2Capture> capture,
                         std::unique_ptr<infer::Yolox> yolox,
                         Options opts,
                         const infer::RtmPose::Options* rtmpose_opts)
    : capture_{std::move(capture)},
      yolox_{std::move(yolox)},
      opts_{std::move(opts)} {
    if (rtmpose_opts) {
        rtmpose_enabled_ = true;
        rtmpose_opts_    = *rtmpose_opts;
    }
}

FrameSource::~FrameSource() {
    try { stop(); } catch (...) {}
}

void FrameSource::start() {
    capture_->start();
    stop_.store(false);
    worker_ = std::thread{&FrameSource::decode_loop, this};
}

void FrameSource::stop() {
    if (!worker_.joinable() && !capture_) return;
    stop_.store(true);
    if (worker_.joinable()) worker_.join();
    if (capture_) capture_->stop();
}

void FrameSource::decode_loop() {
    std::uint64_t last_seq = 0;
    cv::Mat scratch;
    while (!stop_.load()) {
        Frame raw;
        if (!capture_->try_pop_latest(raw) || raw.seq == last_seq) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        last_seq = raw.seq;
        if (!decoder_.decode(raw.jpeg, scratch)) {
            FITRA_LOG_WARN("frame_source: jpeg decode failed for seq={}", raw.seq);
            continue;
        }

        // YOLOX runs on this thread (one IExecutionContext per FrameSource),
        // so all cameras detect in parallel.
        if (yolox_) {
            bool do_detect = (frame_idx_ % opts_.det_frequency == 0)
                          || cached_bboxes_.empty();
            if (do_detect) {
                auto dets = yolox_->infer(scratch);
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
        }

        if (opts_.fake_bbox_if_empty && cached_bboxes_.empty()) {
            infer::Bbox fake{};
            float w = static_cast<float>(scratch.cols);
            float h = static_cast<float>(scratch.rows);
            fake.x1 = 0.2f * w;
            fake.y1 = 0.2f * h;
            fake.x2 = 0.8f * w;
            fake.y2 = 0.8f * h;
            fake.score = 1.0f;
            cached_bboxes_.push_back(fake);
        }

        DecodedFrame df;
        df.seq         = raw.seq;
        df.captured_at = raw.captured_at;
        df.bboxes      = cached_bboxes_;  // copy of current cache

        if (rtmpose_enabled_ && !df.bboxes.empty()) {
            // Preprocess each (frame, bbox) into the contiguous CHW block
            // here on the per-camera worker thread. Phase 6b: shifts the
            // dominant CPU cost off the central inference thread.
            const std::size_t per_item =
                infer::RtmPose::blob_floats_per_item(rtmpose_opts_);
            df.chw_concat.resize(df.bboxes.size() * per_item);
            df.M_invs.resize(df.bboxes.size());
            for (std::size_t i = 0; i < df.bboxes.size(); ++i) {
                infer::RtmPose::preprocess_to_blob(
                    rtmpose_opts_, scratch, df.bboxes[i],
                    df.chw_concat.data() + i * per_item,
                    df.M_invs[i]);
            }
        } else if (!rtmpose_enabled_) {
            df.bgr = scratch.clone();
        }

        {
            std::lock_guard<std::mutex> lk{slot_mu_};
            latest_ = std::move(df);
        }
        ++frame_idx_;
    }
}

bool FrameSource::try_pop_latest_decoded(DecodedFrame& out) {
    std::lock_guard<std::mutex> lk{slot_mu_};
    if (!latest_) return false;
    if (latest_->seq == last_returned_seq_) return false;
    last_returned_seq_ = latest_->seq;
    out = *latest_;
    return true;
}

}  // namespace fitra::camera
