#include "camera/frame_source.hpp"

#include <chrono>

#include "util/logging.hpp"

namespace fitra::camera {

FrameSource::FrameSource(std::unique_ptr<V4l2Capture> capture)
    : capture_{std::move(capture)} {}

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
        DecodedFrame df;
        df.bgr         = scratch.clone();  // detach from scratch so we can reuse it
        df.seq         = raw.seq;
        df.captured_at = raw.captured_at;
        std::lock_guard<std::mutex> lk{slot_mu_};
        latest_ = std::move(df);
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
