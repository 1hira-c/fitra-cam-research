#pragma once
//
// Raw V4L2 MJPEG capture for a single camera.
//
// Bypasses OpenCV's VideoCapture so we can choose buffer count, decode
// path, and synchronization explicitly. Phase 2 keeps things simple:
//   - 4 mmap buffers (matches the migration plan's ring size)
//   - blocking VIDIOC_DQBUF in a worker thread
//   - latest-frame-wins semantics (drop older frames if the consumer is
//     behind), mirroring python/scripts/pose_pipeline.py::CameraReader
//
// The Frame holds a COPY of the JPEG bytes so the V4L2 buffer can be
// re-queued immediately. Phase 4 will swap this for NvBuffer zero-copy.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fitra::camera {

struct Frame {
    std::vector<std::uint8_t> jpeg;
    std::uint64_t seq{0};
    std::chrono::steady_clock::time_point captured_at{};
};

struct V4l2Options {
    std::string device_path;
    int width  = 640;
    int height = 480;
    int fps    = 30;
    int n_buffers = 4;
};

class V4l2Capture {
public:
    explicit V4l2Capture(V4l2Options opts);
    ~V4l2Capture();

    V4l2Capture(const V4l2Capture&) = delete;
    V4l2Capture& operator=(const V4l2Capture&) = delete;

    // Open device, set format, request buffers, start streaming, start the
    // worker thread. Throws on failure.
    void start();

    // Stop streaming, join thread, close device.
    void stop();

    // If a new frame has arrived since the last call, fill `out` and return
    // true. Otherwise return false (no copy). Thread-safe.
    bool try_pop_latest(Frame& out);

    // Receive rate over the last ~60 frames (instantaneous-ish), Hz.
    double recv_fps() const { return recv_fps_.load(); }

    // Monotonically increasing capture count.
    std::uint64_t total_received() const { return total_received_.load(); }

    const V4l2Options& options() const { return opts_; }

private:
    struct MmapBuf {
        void*       ptr     = nullptr;
        std::size_t length  = 0;
    };

    void worker_loop();
    void update_recv_fps(std::chrono::steady_clock::time_point now);

    V4l2Options opts_;
    int fd_ = -1;
    std::vector<MmapBuf> bufs_;
    std::thread worker_;
    std::atomic<bool> stop_{false};

    // latest-frame slot
    mutable std::mutex slot_mu_;
    std::optional<Frame> latest_;
    std::uint64_t last_returned_seq_ = 0;
    std::atomic<std::uint64_t> total_received_{0};

    // fps EMA
    mutable std::mutex fps_mu_;
    std::deque<std::chrono::steady_clock::time_point> recv_times_;
    std::atomic<double> recv_fps_{0.0};
};

}  // namespace fitra::camera
