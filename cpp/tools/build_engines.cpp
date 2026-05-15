// build_engines — CLI: ONNX -> TensorRT engine.
//
// Usage:
//   build_engines --onnx <path> --output <path> [--fp16]
//                 [--workspace-mb N]
//                 [--profile name:minD:optD:maxD]   (repeatable)
//
// Profile dims are comma-separated, e.g. "input:1x3x256x192:1x3x256x192:3x3x256x192".
//
// Convenience presets:
//   --preset yolox        => static, no profile (engine consumes parser-provided shape)
//   --preset rtmpose      => dynamic batch profile on input "input" with
//                            min=1,opt=1,max=3 over (B,3,256,192)
//
// Phase 1 use (one command per model; line continuations omitted on purpose):
//   build_engines --preset yolox   --onnx <yolox_tiny....onnx>   --output models/yolox_tiny.fp16.engine   --fp16
//   build_engines --preset rtmpose --onnx <rtmpose-s....onnx>    --output models/rtmpose_s.fp16.engine    --fp16

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <NvInfer.h>

#include "infer/trt_builder.hpp"
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

nvinfer1::Dims parse_dims(std::string_view s) {
    nvinfer1::Dims d;
    d.nbDims = 0;
    std::string buf{s};
    std::stringstream ss{buf};
    std::string tok;
    while (std::getline(ss, tok, 'x')) {
        if (tok.empty()) continue;
        if (d.nbDims >= nvinfer1::Dims::MAX_DIMS) {
            throw std::runtime_error("too many dims: " + buf);
        }
        d.d[d.nbDims++] = std::stoi(tok);
    }
    if (d.nbDims == 0) {
        throw std::runtime_error("empty dims: " + buf);
    }
    return d;
}

fitra::infer::DynamicProfile parse_profile(std::string_view s) {
    // name:minD:optD:maxD
    std::vector<std::string> parts;
    std::string buf{s};
    std::size_t pos = 0;
    while (pos < buf.size()) {
        auto next = buf.find(':', pos);
        if (next == std::string::npos) {
            parts.emplace_back(buf.substr(pos));
            break;
        }
        parts.emplace_back(buf.substr(pos, next - pos));
        pos = next + 1;
    }
    if (parts.size() != 4) {
        throw std::runtime_error(
            std::string{"--profile expects name:minD:optD:maxD, got: "} + buf);
    }
    fitra::infer::DynamicProfile p;
    p.input_name = parts[0];
    p.min_dims   = parse_dims(parts[1]);
    p.opt_dims   = parse_dims(parts[2]);
    p.max_dims   = parse_dims(parts[3]);
    return p;
}

void print_help() {
    std::puts(
        "build_engines — ONNX -> TensorRT engine (TRT 10)\n"
        "\n"
        "Required:\n"
        "  --onnx PATH           input ONNX file\n"
        "  --output PATH         output .engine file\n"
        "\n"
        "Optional:\n"
        "  --fp16                enable FP16 (Jetson Orin Nano Super: recommended)\n"
        "  --int8                enable INT8 (Phase 4; no calibrator wired yet)\n"
        "  --workspace-mb N      builder workspace (default 1024)\n"
        "  --profile NAME:MIN:OPT:MAX  dynamic-shape profile (repeatable)\n"
        "                              e.g. input:1x3x256x192:1x3x256x192:3x3x256x192\n"
        "  --preset yolox        no profile (static batch=1)\n"
        "  --preset rtmpose      profile input min=1,opt=1,max=3 over Bx3x256x192\n"
        "  --help                show this help\n");
}

}  // namespace

int main(int argc, char** argv) {
    fitra::infer::BuildOptions opts;
    std::string preset;

    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        auto need_arg = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing argument for %s\n", flag);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            print_help();
            return EXIT_SUCCESS;
        } else if (a == "--onnx") {
            opts.onnx_path = need_arg("--onnx");
        } else if (a == "--output" || a == "-o") {
            opts.engine_path = need_arg("--output");
        } else if (a == "--fp16") {
            opts.fp16 = true;
        } else if (a == "--int8") {
            opts.int8 = true;
        } else if (a == "--workspace-mb") {
            opts.workspace_mb = static_cast<std::size_t>(std::stoul(need_arg("--workspace-mb")));
        } else if (a == "--profile") {
            opts.profiles.push_back(parse_profile(need_arg("--profile")));
        } else if (a == "--preset") {
            preset = need_arg("--preset");
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            print_help();
            return EXIT_FAILURE;
        }
    }

    if (opts.onnx_path.empty() || opts.engine_path.empty()) {
        std::fprintf(stderr, "both --onnx and --output are required\n");
        return EXIT_FAILURE;
    }

    if (preset == "rtmpose" && opts.profiles.empty()) {
        fitra::infer::DynamicProfile p;
        p.input_name = "input";
        p.min_dims   = parse_dims("1x3x256x192");
        // opt=3 matches the 3-camera target (one bbox per cam in single-
        // person mode). TRT picks kernels best for `opt`, so opt=1 leaves
        // multi-cam throughput on the table.
        p.opt_dims   = parse_dims("3x3x256x192");
        p.max_dims   = parse_dims("3x3x256x192");
        opts.profiles.push_back(std::move(p));
    } else if (preset == "yolox") {
        // no profile; static shape from ONNX
    } else if (!preset.empty()) {
        std::fprintf(stderr, "unknown preset: %s\n", preset.c_str());
        return EXIT_FAILURE;
    }

    try {
        TrtLogger logger;
        auto bytes = fitra::infer::build_engine(opts, logger);
        FITRA_LOG_INFO("done: {} bytes", bytes);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        FITRA_LOG_ERROR("fatal: {}", e.what());
        return EXIT_FAILURE;
    }
}
