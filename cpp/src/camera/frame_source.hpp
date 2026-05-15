#pragma once
//
// Per-camera decode worker.
//
// Wraps a V4l2Capture (MJPEG producer) and runs a dedicated decode thread
// that converts the latest JPEG into a BGR cv::Mat. The inference thread
// pulls decoded frames via try_pop_latest_decoded(), keeping CPU JPEG
// decode off the inference path so N cameras can decode in parallel.
//
// Replaces the inline cv::imdecode() in MultiCameraDriver::loop. With 2-3
// cameras on the same iteration, decode latency drops from N*~8ms (serial)
// to ~8ms (parallel) before RTMPose.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <opencv2/core.hpp>

#include "camera/jpeg_decoder.hpp"
#include "camera/v4l2_capture.hpp"

namespace fitra::camera {

struct DecodedFrame {
    cv::Mat                              bgr;
    std::uint64_t                        seq{0};
    std::chrono::steady_clock::time_point captured_at{};
};

class FrameSource {
public:
    explicit FrameSource(std::unique_ptr<V4l2Capture> capture);
    ~FrameSource();

    FrameSource(const FrameSource&) = delete;
    FrameSource& operator=(const FrameSource&) = delete;

    void start();   // Starts capture + decode threads.
    void stop();

    // Try to fetch the most-recent decoded frame. Returns false if no new
    // frame is ready since the last call. The returned cv::Mat shares the
    // FrameSource's internal buffer (deep-copied via cv::Mat::clone() into
    // `out.bgr`), so it's safe to use across iterations.
    bool try_pop_latest_decoded(DecodedFrame& out);

    V4l2Capture&  capture()             { return *capture_; }
    const V4l2Options& options() const  { return capture_->options(); }
    double recv_fps() const             { return capture_->recv_fps(); }
    std::uint64_t total_received() const{ return capture_->total_received(); }

private:
    void decode_loop();

    std::unique_ptr<V4l2Capture> capture_;
    JpegDecoder                  decoder_;
    std::thread                  worker_;
    std::atomic<bool>            stop_{false};

    mutable std::mutex          slot_mu_;
    std::optional<DecodedFrame> latest_;
    std::uint64_t               last_returned_seq_ = 0;
};

}  // namespace fitra::camera
