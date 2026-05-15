// dump_keypoints — C++/TensorRT side of the correctness check.
//
// Runs YOLOX + RTMPose on every frame of an input MP4 and emits the same
// JSON Lines schema as python/scripts/dump_reference_keypoints.py:
//   {"frame":N,"persons":[{"bbox":[x1,y1,x2,y2,score],
//                          "kpts":[[x,y,score], ...17]}]}
//
// Pair with python/scripts/compare_keypoints.py to compute bbox IoU and
// keypoint L2 statistics against the Python reference.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <NvInfer.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "infer/rtmpose.hpp"
#include "infer/trt_engine.hpp"
#include "infer/yolox.hpp"
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
        "dump_keypoints — TRT pipeline dump for correctness check\n"
        "\n"
        "Required:\n"
        "  --video PATH              input MP4 (e.g. outputs/recorded_rtmpose/.../raw_cam0.mp4)\n"
        "  --det-engine PATH         YOLOX .engine\n"
        "  --pose-engine PATH        RTMPose .engine\n"
        "  --output PATH             output JSONL\n"
        "\n"
        "Optional:\n"
        "  --max-frames N            stop after N frames (default 0 = whole video)\n"
        "  --det-score F             detection score threshold (default 0.5)\n"
        "  --multi-person            run pose on all bboxes (default: largest only)\n"
        "  --help                    show this help\n");
}

std::string fmt_float(float v) {
    // Format with enough digits to round-trip without padding zeros.
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", static_cast<double>(v));
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    std::string video;
    std::string det_engine;
    std::string pose_engine;
    std::string output;
    int   max_frames = 0;
    float det_score  = 0.5f;
    bool  multi_person = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        auto need_arg = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing argument for %s\n", flag);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { print_help(); return EXIT_SUCCESS; }
        else if (a == "--video")        { video       = need_arg("--video"); }
        else if (a == "--det-engine")   { det_engine  = need_arg("--det-engine"); }
        else if (a == "--pose-engine")  { pose_engine = need_arg("--pose-engine"); }
        else if (a == "--output")       { output      = need_arg("--output"); }
        else if (a == "--max-frames")   { max_frames  = std::atoi(need_arg("--max-frames")); }
        else if (a == "--det-score")    { det_score   = std::stof(need_arg("--det-score")); }
        else if (a == "--multi-person") { multi_person = true; }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            print_help();
            return EXIT_FAILURE;
        }
    }
    if (video.empty() || det_engine.empty() || pose_engine.empty() || output.empty()) {
        print_help();
        return EXIT_FAILURE;
    }

    try {
        TrtLogger tlog;
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(tlog)};
        TRT_CHECK(runtime != nullptr);

        FITRA_LOG_INFO("loading YOLOX engine: {}", det_engine);
        auto yolox_engine = fitra::infer::TrtEngine::from_file(*runtime, det_engine, tlog);
        FITRA_LOG_INFO("loading RTMPose engine: {}", pose_engine);
        auto rtmpose_engine = fitra::infer::TrtEngine::from_file(*runtime, pose_engine, tlog);

        fitra::infer::Yolox::Options yolo_opts;
        yolo_opts.score_thr = det_score;
        fitra::infer::Yolox yolox{*yolox_engine, yolo_opts};
        fitra::infer::RtmPose rtmpose{*rtmpose_engine};

        cv::VideoCapture cap{video};
        if (!cap.isOpened()) {
            FITRA_LOG_ERROR("failed to open {}", video);
            return EXIT_FAILURE;
        }
        int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        FITRA_LOG_INFO("video: {} frames", total);

        std::filesystem::path outp{output};
        if (outp.has_parent_path()) {
            std::filesystem::create_directories(outp.parent_path());
        }
        std::ofstream fout{output, std::ios::trunc};
        if (!fout.is_open()) {
            FITRA_LOG_ERROR("failed to open {}", output);
            return EXIT_FAILURE;
        }

        cv::Mat frame;
        int     written = 0;
        auto    t0 = std::chrono::steady_clock::now();
        for (int i = 0; ; ++i) {
            if (!cap.read(frame) || frame.empty()) break;

            auto bboxes = yolox.infer(frame);
            if (!multi_person && bboxes.size() > 1) {
                auto largest = std::max_element(
                    bboxes.begin(), bboxes.end(),
                    [](const auto& a, const auto& b) {
                        float aa = (a.x2 - a.x1) * (a.y2 - a.y1);
                        float bb = (b.x2 - b.x1) * (b.y2 - b.y1);
                        return aa < bb;
                    });
                fitra::infer::Bbox keep = *largest;
                bboxes.clear();
                bboxes.push_back(keep);
            }
            auto persons = rtmpose.infer(frame, bboxes);

            // emit JSON line
            std::ostringstream line;
            line << "{\"frame\":" << i << ",\"persons\":[";
            for (std::size_t pi = 0; pi < persons.size(); ++pi) {
                if (pi) line << ",";
                const auto& p = persons[pi];
                line << "{\"bbox\":[" << fmt_float(p.bbox.x1) << "," << fmt_float(p.bbox.y1)
                     << "," << fmt_float(p.bbox.x2) << "," << fmt_float(p.bbox.y2)
                     << "," << fmt_float(p.bbox.score) << "],\"kpts\":[";
                for (std::size_t k = 0; k < p.kpts.size(); ++k) {
                    if (k) line << ",";
                    line << "[" << fmt_float(p.kpts[k].x) << "," << fmt_float(p.kpts[k].y)
                         << "," << fmt_float(p.kpts[k].score) << "]";
                }
                line << "]}";
            }
            line << "]}";
            fout << line.str() << "\n";

            ++written;
            if (max_frames && written >= max_frames) break;
            if (written % 50 == 0) {
                FITRA_LOG_INFO("  {}/{} frames", written, total);
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.2fs (%.2f fps)",
                      secs, written / std::max(secs, 1e-9));
        FITRA_LOG_INFO("done: {} frames in {} -> {}", written, buf, output);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        FITRA_LOG_ERROR("fatal: {}", e.what());
        return EXIT_FAILURE;
    }
}
