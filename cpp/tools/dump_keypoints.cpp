// dump_keypoints — C++/TensorRT side of the correctness check.
//
// Runs YOLOX + RTMPose on every frame of an input MP4 and emits the same
// JSON Lines schema as python/scripts/dump_reference_keypoints.py:
//   {"frame":N,"persons":[{"bbox":[x1,y1,x2,y2,score],
//                          "kpts":[[x,y,score], ...17]}]}
//
// Pair with python/scripts/compare_keypoints.py to compute bbox IoU and
// keypoint L2 statistics against the Python reference.
//
// Two RTMPose code paths:
//   - default: rtmpose.infer(frame, bboxes)  (internal preprocess)
//   - --prebaked: preprocess_to_blob + infer_prebaked(), mirrors the
//     Phase 6b live-pipeline path. Use this to verify the fast path is
//     numerically equivalent on a recorded MP4.
//
// --overlay PATH also writes an MP4 with skeleton drawn on each frame so
// a human can sanity-check the result visually (the Phase 6 PR demands
// this; raw fps numbers say nothing about whether bboxes/kpts make sense).

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
#include <utility>
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
        "  --prebaked                use Phase 6b infer_prebaked() path\n"
        "                             (preprocess_to_blob + infer_prebaked)\n"
        "  --overlay PATH            also write an overlay MP4 with skeleton drawn\n"
        "  --help                    show this help\n");
}

std::string fmt_float(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", static_cast<double>(v));
    return buf;
}

// COCO 17-keypoint skeleton (same edges as python/scripts/pose_pipeline.py).
constexpr std::pair<int, int> kSkel[] = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {5, 6}, {5, 11}, {6, 12}, {11, 12},
    {11, 13}, {13, 15}, {12, 14}, {14, 16},
};

void draw_overlay(cv::Mat& frame,
                  const std::vector<fitra::infer::Person>& persons,
                  float kp_thr = 0.3f) {
    const cv::Scalar color(0, 220, 0);
    for (const auto& p : persons) {
        cv::rectangle(frame,
                      {static_cast<int>(p.bbox.x1), static_cast<int>(p.bbox.y1)},
                      {static_cast<int>(p.bbox.x2), static_cast<int>(p.bbox.y2)},
                      cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
        for (auto [a, b] : kSkel) {
            const auto& ka = p.kpts[static_cast<std::size_t>(a)];
            const auto& kb = p.kpts[static_cast<std::size_t>(b)];
            if (ka.score < kp_thr || kb.score < kp_thr) continue;
            cv::line(frame,
                     {static_cast<int>(ka.x), static_cast<int>(ka.y)},
                     {static_cast<int>(kb.x), static_cast<int>(kb.y)},
                     color, 2, cv::LINE_AA);
        }
        for (const auto& kp : p.kpts) {
            if (kp.score < kp_thr) continue;
            cv::circle(frame, {static_cast<int>(kp.x), static_cast<int>(kp.y)},
                       3, color, -1, cv::LINE_AA);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string video;
    std::string det_engine;
    std::string pose_engine;
    std::string output;
    std::string overlay_path;
    int   max_frames = 0;
    float det_score  = 0.5f;
    bool  multi_person = false;
    bool  use_prebaked = false;

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
        else if (a == "--prebaked")     { use_prebaked = true; }
        else if (a == "--overlay")      { overlay_path = need_arg("--overlay"); }
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
        const auto& rtmpose_opts = rtmpose.options();
        const std::size_t per_item = fitra::infer::RtmPose::blob_floats_per_item(rtmpose_opts);

        cv::VideoCapture cap{video};
        if (!cap.isOpened()) {
            FITRA_LOG_ERROR("failed to open {}", video);
            return EXIT_FAILURE;
        }
        int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double src_fps = cap.get(cv::CAP_PROP_FPS);
        if (src_fps <= 0) src_fps = 30.0;
        int src_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int src_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        char src_meta[80];
        std::snprintf(src_meta, sizeof(src_meta),
                      "%d frames %dx%d @ %.2f fps", total, src_w, src_h, src_fps);
        FITRA_LOG_INFO("video: {}", src_meta);

        std::filesystem::path outp{output};
        if (outp.has_parent_path()) {
            std::filesystem::create_directories(outp.parent_path());
        }
        std::ofstream fout{output, std::ios::trunc};
        if (!fout.is_open()) {
            FITRA_LOG_ERROR("failed to open {}", output);
            return EXIT_FAILURE;
        }

        std::unique_ptr<cv::VideoWriter> overlay_writer;
        if (!overlay_path.empty()) {
            std::filesystem::path op{overlay_path};
            if (op.has_parent_path()) {
                std::filesystem::create_directories(op.parent_path());
            }
            overlay_writer = std::make_unique<cv::VideoWriter>(
                overlay_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                src_fps, cv::Size(src_w, src_h));
            if (!overlay_writer->isOpened()) {
                FITRA_LOG_ERROR("failed to open overlay writer at {}", overlay_path);
                return EXIT_FAILURE;
            }
            FITRA_LOG_INFO("overlay output: {} ({} mode)",
                           overlay_path, use_prebaked ? "prebaked" : "internal");
        }

        // Scratch buffers reused across frames for the prebaked path.
        std::vector<std::vector<float>>                chw_bufs;
        std::vector<cv::Mat>                            M_invs;
        std::vector<fitra::infer::RtmPose::PrebakedRequest> prereqs;

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

            std::vector<fitra::infer::Person> persons;
            if (use_prebaked) {
                // Mirrors the Phase 6b live path: preprocess_to_blob on the
                // caller side, then infer_prebaked.
                chw_bufs.resize(bboxes.size());
                M_invs.resize(bboxes.size());
                prereqs.clear();
                prereqs.reserve(bboxes.size());
                for (std::size_t k = 0; k < bboxes.size(); ++k) {
                    chw_bufs[k].resize(per_item);
                    fitra::infer::RtmPose::preprocess_to_blob(
                        rtmpose_opts, frame, bboxes[k],
                        chw_bufs[k].data(), M_invs[k]);
                    fitra::infer::RtmPose::PrebakedRequest pr;
                    pr.chw   = chw_bufs[k].data();
                    pr.M_inv = M_invs[k];
                    pr.bbox  = bboxes[k];
                    prereqs.push_back(pr);
                }
                persons = rtmpose.infer_prebaked(prereqs);
            } else {
                persons = rtmpose.infer(frame, bboxes);
            }

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

            if (overlay_writer) {
                draw_overlay(frame, persons);
                overlay_writer->write(frame);
            }

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
        if (overlay_writer) {
            overlay_writer->release();
            FITRA_LOG_INFO("overlay -> {}", overlay_path);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        FITRA_LOG_ERROR("fatal: {}", e.what());
        return EXIT_FAILURE;
    }
}
