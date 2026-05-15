// pose_bench — Phase 2 single-camera end-to-end benchmark.
//
// Opens one V4L2 camera, runs the YOLOX + RTMPose pipeline, prints
// rolling fps / stage latency until --max-frames or Ctrl-C.
//
// Compare against the Python single-camera baseline at the same image
// size / det-frequency (one line, line continuations omitted):
//   python python/scripts/dual_rtmpose_cameras.py --device cuda --max-frames 200

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#include <NvInfer.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "camera/v4l2_capture.hpp"
#include "infer/rtmpose.hpp"
#include "infer/trt_engine.hpp"
#include "infer/yolox.hpp"
#include "pipeline/pose_pipeline.hpp"
#include "util/cuda_check.hpp"
#include "util/logging.hpp"

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
        "pose_bench — Phase 2 single-camera benchmark\n"
        "\n"
        "Required:\n"
        "  --cam0 PATH               V4L2 device (e.g. /dev/v4l/by-path/...index0)\n"
        "  --det-engine PATH         YOLOX .engine\n"
        "  --pose-engine PATH        RTMPose .engine\n"
        "\n"
        "Optional:\n"
        "  --width N / --height N    capture size (default 640x480)\n"
        "  --fps N                   requested capture fps (default 30)\n"
        "  --det-frequency N         run YOLOX every N frames (default 10)\n"
        "  --multi-person            process all bboxes (default: largest only)\n"
        "  --det-score F             detection score threshold (default 0.5)\n"
        "  --max-frames N            stop after N processed frames (0 = run until Ctrl-C)\n"
        "  --log-every-s F           stats interval in seconds (default 1.0)\n"
        "  --save-every N            save annotated JPEG every N processed frames (0 = off)\n"
        "  --output-dir DIR          where to save JPEGs (default outputs/pose_bench)\n");
}

void draw_overlay(cv::Mat& frame, const std::vector<fitra::infer::Person>& persons) {
    static const std::pair<int, int> kSkel[] = {
        {0,1},{0,2},{1,3},{2,4},
        {5,7},{7,9},{6,8},{8,10},
        {5,6},{5,11},{6,12},{11,12},
        {11,13},{13,15},{12,14},{14,16},
    };
    cv::Scalar color(0, 220, 0);
    for (const auto& p : persons) {
        for (auto [a, b] : kSkel) {
            const auto& ka = p.kpts[static_cast<std::size_t>(a)];
            const auto& kb = p.kpts[static_cast<std::size_t>(b)];
            if (ka.score < 0.3f || kb.score < 0.3f) continue;
            cv::line(frame,
                     {static_cast<int>(ka.x), static_cast<int>(ka.y)},
                     {static_cast<int>(kb.x), static_cast<int>(kb.y)},
                     color, 2, cv::LINE_AA);
        }
        for (const auto& kp : p.kpts) {
            if (kp.score < 0.3f) continue;
            cv::circle(frame, {static_cast<int>(kp.x), static_cast<int>(kp.y)},
                       3, color, -1, cv::LINE_AA);
        }
        cv::rectangle(frame,
                      {static_cast<int>(p.bbox.x1), static_cast<int>(p.bbox.y1)},
                      {static_cast<int>(p.bbox.x2), static_cast<int>(p.bbox.y2)},
                      color, 1, cv::LINE_AA);
    }
}

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

}  // namespace

int main(int argc, char** argv) {
    fitra::camera::V4l2Options cam_opts;
    cam_opts.device_path = "";
    cam_opts.width = 640;
    cam_opts.height = 480;
    cam_opts.fps = 30;
    cam_opts.n_buffers = 4;

    std::string det_engine_path;
    std::string pose_engine_path;
    fitra::pipeline::PipelineOptions pipe_opts;
    float det_score = 0.5f;
    int   max_frames = 0;
    double log_every_s = 1.0;
    int   save_every  = 0;
    std::string output_dir = "outputs/pose_bench";

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
        else if (a == "--cam0")            cam_opts.device_path = need("--cam0");
        else if (a == "--width")           cam_opts.width  = std::atoi(need("--width"));
        else if (a == "--height")          cam_opts.height = std::atoi(need("--height"));
        else if (a == "--fps")             cam_opts.fps    = std::atoi(need("--fps"));
        else if (a == "--det-engine")      det_engine_path  = need("--det-engine");
        else if (a == "--pose-engine")     pose_engine_path = need("--pose-engine");
        else if (a == "--det-frequency")   pipe_opts.det_frequency = std::atoi(need("--det-frequency"));
        else if (a == "--multi-person")    pipe_opts.single_person = false;
        else if (a == "--det-score")       det_score = std::stof(need("--det-score"));
        else if (a == "--max-frames")      max_frames = std::atoi(need("--max-frames"));
        else if (a == "--log-every-s")     log_every_s = std::stod(need("--log-every-s"));
        else if (a == "--save-every")      save_every = std::atoi(need("--save-every"));
        else if (a == "--output-dir")      output_dir = need("--output-dir");
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            print_help();
            return EXIT_FAILURE;
        }
    }
    if (cam_opts.device_path.empty() || det_engine_path.empty() || pose_engine_path.empty()) {
        print_help();
        return EXIT_FAILURE;
    }
    std::signal(SIGINT, on_signal);

    try {
        TrtLogger tlog;
        std::unique_ptr<nvinfer1::IRuntime> rt{nvinfer1::createInferRuntime(tlog)};
        TRT_CHECK(rt != nullptr);

        FITRA_LOG_INFO("loading YOLOX engine: {}", det_engine_path);
        auto yolox_eng   = fitra::infer::TrtEngine::from_file(*rt, det_engine_path, tlog);
        FITRA_LOG_INFO("loading RTMPose engine: {}", pose_engine_path);
        auto rtmpose_eng = fitra::infer::TrtEngine::from_file(*rt, pose_engine_path, tlog);

        fitra::infer::Yolox::Options yolo_opts;
        yolo_opts.score_thr = det_score;
        fitra::infer::Yolox   yolox{*yolox_eng, yolo_opts};
        fitra::infer::RtmPose rtmpose{*rtmpose_eng};

        fitra::camera::V4l2Capture cap{cam_opts};
        cap.start();

        fitra::pipeline::PosePipeline pipe{cap, yolox, rtmpose, pipe_opts};

        std::filesystem::path save_dir;
        if (save_every > 0) {
            auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            char ts[32];
            std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
            save_dir = std::filesystem::path{output_dir} / ts;
            std::filesystem::create_directories(save_dir);
            FITRA_LOG_INFO("saving annotated JPEGs to {}", save_dir.string());
        }

        auto last_log = std::chrono::steady_clock::now();
        int saved = 0;
        while (!g_stop.load()) {
            auto sr = pipe.step();
            if (!sr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } else {
                if (save_every > 0
                    && static_cast<int>(pipe.stats().processed_count) % save_every == 0) {
                    draw_overlay(sr->frame_bgr, sr->persons);
                    char fname[64];
                    std::snprintf(fname, sizeof(fname), "frame_%06d.jpg",
                                  static_cast<int>(pipe.stats().processed_count));
                    cv::imwrite((save_dir / fname).string(), sr->frame_bgr);
                    ++saved;
                }
            }
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - last_log).count();
            if (dt >= log_every_s) {
                const auto& s = pipe.stats();
                std::uint64_t recv_total = cap.total_received();
                std::uint64_t pending = recv_total > s.processed_count
                                        ? recv_total - s.processed_count : 0;
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "recv=%5.2f avg_pose=%5.2f recent_pose=%5.2f "
                              "stage_ms=%6.1f processed=%llu pending=%llu",
                              cap.recv_fps(), s.avg_pose_fps, s.recent_pose_fps,
                              s.last_stage_ms,
                              static_cast<unsigned long long>(s.processed_count),
                              static_cast<unsigned long long>(pending));
                FITRA_LOG_INFO("{}", buf);
                last_log = now;
            }
            if (max_frames > 0
                && static_cast<int>(pipe.stats().processed_count) >= max_frames) {
                break;
            }
        }

        cap.stop();
        const auto& s = pipe.stats();
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "final: processed=%llu avg_pose=%.2f recent_pose=%.2f saved_jpegs=%d",
                      static_cast<unsigned long long>(s.processed_count),
                      s.avg_pose_fps, s.recent_pose_fps, saved);
        FITRA_LOG_INFO("{}", buf);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        FITRA_LOG_ERROR("fatal: {}", e.what());
        return EXIT_FAILURE;
    }
}
