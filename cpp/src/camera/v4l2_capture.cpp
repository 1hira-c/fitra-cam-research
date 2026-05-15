#include "camera/v4l2_capture.hpp"

#include <cerrno>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "util/logging.hpp"

namespace fitra::camera {

namespace {

int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do {
        r = ::ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

[[noreturn]] void throw_errno(const std::string& what) {
    std::ostringstream oss;
    oss << what << ": errno=" << errno << " (" << std::strerror(errno) << ")";
    throw std::runtime_error(oss.str());
}

}  // namespace

V4l2Capture::V4l2Capture(V4l2Options opts) : opts_{std::move(opts)} {}

V4l2Capture::~V4l2Capture() {
    try { stop(); } catch (...) {}
}

void V4l2Capture::start() {
    fd_ = ::open(opts_.device_path.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd_ < 0) throw_errno("open(" + opts_.device_path + ")");

    // Set MJPEG format.
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = opts_.width;
    fmt.fmt.pix.height      = opts_.height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) throw_errno("VIDIOC_S_FMT");
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
        throw std::runtime_error("driver did not accept MJPEG format");
    }
    if (static_cast<int>(fmt.fmt.pix.width)  != opts_.width
     || static_cast<int>(fmt.fmt.pix.height) != opts_.height) {
        FITRA_LOG_WARN("driver returned {}x{} (requested {}x{})",
                       fmt.fmt.pix.width, fmt.fmt.pix.height,
                       opts_.width, opts_.height);
        opts_.width  = static_cast<int>(fmt.fmt.pix.width);
        opts_.height = static_cast<int>(fmt.fmt.pix.height);
    }

    // Frame interval (best-effort).
    v4l2_streamparm sp{};
    sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    sp.parm.capture.timeperframe.numerator   = 1;
    sp.parm.capture.timeperframe.denominator = opts_.fps;
    if (xioctl(fd_, VIDIOC_S_PARM, &sp) < 0) {
        FITRA_LOG_WARN("VIDIOC_S_PARM failed; continuing with driver default fps");
    }

    // Request N mmap buffers.
    v4l2_requestbuffers req{};
    req.count  = static_cast<__u32>(opts_.n_buffers);
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) throw_errno("VIDIOC_REQBUFS");
    if (req.count < 2) {
        throw std::runtime_error("driver granted <2 buffers; cannot pipeline");
    }
    bufs_.resize(req.count);

    // Query and mmap each buffer.
    for (std::size_t i = 0; i < bufs_.size(); ++i) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = static_cast<__u32>(i);
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) throw_errno("VIDIOC_QUERYBUF");
        void* p = ::mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                         fd_, buf.m.offset);
        if (p == MAP_FAILED) throw_errno("mmap");
        bufs_[i].ptr    = p;
        bufs_[i].length = buf.length;
    }

    // Queue all buffers.
    for (std::size_t i = 0; i < bufs_.size(); ++i) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = static_cast<__u32>(i);
        if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) throw_errno("VIDIOC_QBUF (initial)");
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) throw_errno("VIDIOC_STREAMON");

    FITRA_LOG_INFO("v4l2: {} opened ({}x{}, MJPG, {} buffers, requested {} fps)",
                   opts_.device_path, opts_.width, opts_.height,
                   bufs_.size(), opts_.fps);

    stop_.store(false);
    worker_ = std::thread{&V4l2Capture::worker_loop, this};
}

void V4l2Capture::stop() {
    if (fd_ < 0 && !worker_.joinable()) return;
    stop_.store(true);
    if (worker_.joinable()) worker_.join();

    if (fd_ >= 0) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    for (auto& b : bufs_) {
        if (b.ptr && b.length) ::munmap(b.ptr, b.length);
        b.ptr = nullptr;
        b.length = 0;
    }
    bufs_.clear();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void V4l2Capture::worker_loop() {
    while (!stop_.load()) {
        pollfd pfd{};
        pfd.fd     = fd_;
        pfd.events = POLLIN;
        int pr = ::poll(&pfd, 1, 200);  // 200ms tick to let stop_ propagate
        if (pr < 0) {
            if (errno == EINTR) continue;
            FITRA_LOG_ERROR("v4l2 poll: {}", std::strerror(errno));
            return;
        }
        if (pr == 0) continue;
        if (!(pfd.revents & POLLIN)) continue;

        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) continue;
            FITRA_LOG_ERROR("v4l2 VIDIOC_DQBUF: {}", std::strerror(errno));
            return;
        }
        auto now = std::chrono::steady_clock::now();
        std::uint64_t seq = total_received_.fetch_add(1) + 1;

        // Copy out so we can re-queue the V4L2 buffer immediately.
        Frame f;
        f.jpeg.assign(static_cast<std::uint8_t*>(bufs_[buf.index].ptr),
                      static_cast<std::uint8_t*>(bufs_[buf.index].ptr) + buf.bytesused);
        f.seq         = seq;
        f.captured_at = now;

        // Re-queue
        v4l2_buffer qb{};
        qb.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        qb.memory = V4L2_MEMORY_MMAP;
        qb.index  = buf.index;
        if (xioctl(fd_, VIDIOC_QBUF, &qb) < 0) {
            FITRA_LOG_ERROR("v4l2 VIDIOC_QBUF: {}", std::strerror(errno));
            return;
        }

        {
            std::lock_guard<std::mutex> lk{slot_mu_};
            latest_ = std::move(f);
        }
        update_recv_fps(now);
    }
}

void V4l2Capture::update_recv_fps(std::chrono::steady_clock::time_point now) {
    std::lock_guard<std::mutex> lk{fps_mu_};
    recv_times_.push_back(now);
    while (recv_times_.size() > 60) recv_times_.pop_front();
    if (recv_times_.size() >= 2) {
        auto span = std::chrono::duration<double>(recv_times_.back() - recv_times_.front()).count();
        if (span > 0) {
            recv_fps_.store((recv_times_.size() - 1) / span);
        }
    }
}

bool V4l2Capture::try_pop_latest(Frame& out) {
    std::lock_guard<std::mutex> lk{slot_mu_};
    if (!latest_) return false;
    if (latest_->seq == last_returned_seq_) return false;
    last_returned_seq_ = latest_->seq;
    out = *latest_;
    return true;
}

}  // namespace fitra::camera
