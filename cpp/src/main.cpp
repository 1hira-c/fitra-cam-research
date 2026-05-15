// fitra-cam main — Phase 3 driver.
//
// Runs N V4L2 USB cameras through the shared YOLOX + RTMPose TRT
// pipeline and exposes the result via Crow (HTTP + WebSocket).
//
// Usage (one line, no shell continuation):
//   fitra-cam --cam0 PATH [--cam1 PATH] [--cam2 PATH] --det-engine PATH --pose-engine PATH
//             [--port 8000] [--host 0.0.0.0] [--static DIR] [--no-web]
//             [--width 640] [--height 480] [--fps 30]
//             [--det-frequency 10] [--multi-person] [--probe]
//
// `--probe` keeps the Phase 0 diagnostic (CUDA device + TRT runtime sanity check).

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferVersion.h>

#include "camera/v4l2_capture.hpp"
#include "infer/rtmpose.hpp"
#include "infer/trt_engine.hpp"
#include "infer/yolox.hpp"
#include "pipeline/multi_pipeline.hpp"
#include "pipeline/snapshot.hpp"
#include "util/cuda_check.hpp"
#include "util/logging.hpp"
#include "web/crow_server.hpp"

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > Severity::kWARNING) return;
        using S = Severity;
        switch (severity) {
            case S::kINTERNAL_ERROR: FITRA_LOG_ERROR("[trt] INTERNAL: {}", msg); return;
            case S::kERROR:          FITRA_LOG_ERROR("[trt] {}",          msg); return;
            case S::kWARNING:        FITRA_LOG_WARN ("[trt] {}",          msg); return;
            case S::kINFO:           FITRA_LOG_INFO ("[trt] {}",          msg); return;
            case S::kVERBOSE:        FITRA_LOG_TRACE("[trt] {}",          msg); return;
        }
    }
};

void print_help() {
    std::puts(
        "fitra-cam (C++) — Phase 3 driver\n"
        "\n"
        "Required:\n"
        "  --cam0 PATH               first V4L2 device (e.g. /dev/v4l/by-path/...index0)\n"
        "  --det-engine PATH         YOLOX .engine\n"
        "  --pose-engine PATH        RTMPose .engine\n"
        "\n"
        "Additional cameras:\n"
        "  --cam1 PATH               second camera\n"
        "  --cam2 PATH               third camera\n"
        "\n"
        "Optional:\n"
        "  --port N                  HTTP/WS port (default 8000)\n"
        "  --host ADDR               bind address (default 0.0.0.0)\n"
        "  --static DIR              web frontend dir (default <repo>/web/dual_rtmpose)\n"
        "  --no-web                  do not start Crow (driver only, for bench)\n"
        "  --width N / --height N    capture size per camera (default 640x480)\n"
        "  --fps N                   requested capture fps (default 30)\n"
        "  --det-frequency N         run YOLOX every N frames (default 10)\n"
        "  --multi-person            process all bboxes per camera (default: largest only)\n"
        "  --det-score F             detection score threshold (default 0.5)\n"
        "  --log-every-s F           stats interval in seconds (default 2.0)\n"
        "  --probe                   Phase 0 sanity check and exit\n"
        "  --help                    show this help\n");
}

int probe() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    FITRA_LOG_INFO("CUDA device count = {}", device_count);
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        FITRA_LOG_INFO("  [{}] {} (sm_{}{}, {} MB)",
                       i, prop.name, prop.major, prop.minor,
                       static_cast<unsigned long long>(prop.totalGlobalMem) / (1024ULL * 1024ULL));
    }
    FITRA_LOG_INFO("TensorRT headers: {}.{}.{}",
                   NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
    TrtLogger trt_logger;
    std::unique_ptr<nvinfer1::IRuntime> rt{nvinfer1::createInferRuntime(trt_logger)};
    TRT_CHECK(rt != nullptr);
    FITRA_LOG_INFO("nvinfer1::IRuntime ok (lib build: {})", getInferLibVersion());
    return EXIT_SUCCESS;
}

std::filesystem::path guess_static_dir() {
    auto exe = std::filesystem::canonical("/proc/self/exe");
    // build/main lives at <repo>/cpp/build/main; we want <repo>/web/dual_rtmpose
    auto repo = exe.parent_path().parent_path().parent_path();
    return repo / "web" / "dual_rtmpose";
}

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> cam_paths;
    cam_paths.resize(3);  // slots for cam0..cam2
    std::string det_engine_path;
    std::string pose_engine_path;
    int   port = 8000;
    std::string host = "0.0.0.0";
    std::string static_dir;
    bool  no_web = false;
    int   width = 640, height = 480, fps = 30;
    int   det_frequency = 10;
    bool  multi_person = false;
    bool  bench_fake_bbox = false;
    float det_score = 0.5f;
    double log_every_s = 2.0;
    bool  want_probe = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing argument for %s\n", flag);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if      (a == "--help" || a == "-h") { print_help(); return EXIT_SUCCESS; }
        else if (a == "--probe")             { want_probe = true; }
        else if (a == "--cam0")              { cam_paths[0] = need("--cam0"); }
        else if (a == "--cam1")              { cam_paths[1] = need("--cam1"); }
        else if (a == "--cam2")              { cam_paths[2] = need("--cam2"); }
        else if (a == "--det-engine")        { det_engine_path  = need("--det-engine"); }
        else if (a == "--pose-engine")       { pose_engine_path = need("--pose-engine"); }
        else if (a == "--port")              { port = std::atoi(need("--port")); }
        else if (a == "--host")              { host = need("--host"); }
        else if (a == "--static")            { static_dir = need("--static"); }
        else if (a == "--no-web")            { no_web = true; }
        else if (a == "--width")             { width  = std::atoi(need("--width")); }
        else if (a == "--height")            { height = std::atoi(need("--height")); }
        else if (a == "--fps")               { fps    = std::atoi(need("--fps")); }
        else if (a == "--det-frequency")     { det_frequency = std::atoi(need("--det-frequency")); }
        else if (a == "--multi-person")      { multi_person  = true; }
        else if (a == "--bench-fake-bbox")   { bench_fake_bbox = true; }
        else if (a == "--det-score")         { det_score = std::stof(need("--det-score")); }
        else if (a == "--log-every-s")       { log_every_s = std::stod(need("--log-every-s")); }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            print_help();
            return EXIT_FAILURE;
        }
    }

    try {
        if (want_probe) return probe();
        if (cam_paths[0].empty() || det_engine_path.empty() || pose_engine_path.empty()) {
            print_help();
            return EXIT_FAILURE;
        }
        std::signal(SIGINT, on_signal);

        TrtLogger tlog;
        std::unique_ptr<nvinfer1::IRuntime> rt{nvinfer1::createInferRuntime(tlog)};
        TRT_CHECK(rt != nullptr);

        FITRA_LOG_INFO("loading YOLOX engine (shared): {}", det_engine_path);
        auto yolox_shared = fitra::infer::TrtEngine::load_shared(*rt, det_engine_path);
        FITRA_LOG_INFO("loading RTMPose engine: {}", pose_engine_path);
        auto rtmpose_eng  = fitra::infer::TrtEngine::from_file(*rt, pose_engine_path, tlog);
        // RTMPose stays as a single shared instance — batching across cameras
        // requires one execution context fed serially from the main thread.
        fitra::infer::RtmPose rtmpose{*rtmpose_eng};

        // One V4l2Capture + one Yolox (per-camera IExecutionContext) per cam.
        // Engines wrap into FrameSource which runs its own decode + YOLOX
        // thread, so all N cameras run capture/decode/YOLOX in parallel.
        std::vector<std::unique_ptr<fitra::infer::TrtEngine>> yolox_engines;
        std::vector<std::unique_ptr<fitra::camera::FrameSource>> sources;
        for (auto& path : cam_paths) {
            if (path.empty()) continue;
            fitra::camera::V4l2Options o;
            o.device_path = path;
            o.width  = width;
            o.height = height;
            o.fps    = fps;
            auto cap = std::make_unique<fitra::camera::V4l2Capture>(o);

            auto yolox_eng = fitra::infer::TrtEngine::from_shared(yolox_shared);
            fitra::infer::Yolox::Options yolo_opts;
            yolo_opts.score_thr = det_score;
            auto yolox = std::make_unique<fitra::infer::Yolox>(*yolox_eng, yolo_opts);
            yolox_engines.push_back(std::move(yolox_eng));

            fitra::camera::FrameSource::Options src_opts;
            src_opts.det_frequency = det_frequency;
            src_opts.single_person = !multi_person;
            src_opts.fake_bbox_if_empty = bench_fake_bbox;
            // Have the per-camera worker pre-bake the RTMPose input so the
            // central inference thread only does memcpy + GPU + decode.
            const auto& rtmpose_opts = rtmpose.options();
            sources.push_back(std::make_unique<fitra::camera::FrameSource>(
                std::move(cap), std::move(yolox), src_opts, &rtmpose_opts));
        }
        std::size_t n_cams = sources.size();

        fitra::pipeline::SnapshotBus bus{n_cams};
        fitra::pipeline::MultiCameraDriver driver{
            std::move(sources), rtmpose, bus};
        driver.start();

        std::unique_ptr<fitra::web::CrowServer> server;
        if (!no_web) {
            fitra::web::ServerOptions sopts;
            sopts.host = host;
            sopts.port = port;
            sopts.static_dir = static_dir.empty()
                                ? guess_static_dir().string()
                                : static_dir;
            server = std::make_unique<fitra::web::CrowServer>(bus, sopts);
            server->start();
        }

        auto last_log = std::chrono::steady_clock::now();
        while (!g_stop.load()) {
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - last_log).count();
            if (dt >= log_every_s) {
                for (std::size_t i = 0; i < n_cams; ++i) {
                    const auto& s = driver.stats_for(i);
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                                  "cam%zu: recv=%5.2f avg_pose=%5.2f recent_pose=%5.2f "
                                  "stage_ms=%6.1f processed=%llu pending=%llu",
                                  i, driver.recv_fps_for(i),
                                  s.avg_pose_fps, s.recent_pose_fps,
                                  s.last_stage_ms,
                                  static_cast<unsigned long long>(s.processed_count),
                                  static_cast<unsigned long long>(driver.pending_for(i)));
                    FITRA_LOG_INFO("{}", buf);
                }
                last_log = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (server) server->stop();
        driver.stop();
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        FITRA_LOG_ERROR("fatal: {}", e.what());
        return EXIT_FAILURE;
    }
}
