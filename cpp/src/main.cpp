// fitra-cam main entry — Phase 0 skeleton.
//
// Goal at this phase: confirm the build links, CUDA reports a device, and
// nvinfer1::createInferRuntime() returns a non-null pointer.
//
// Phases 1-4 will replace this main with the real pipeline (V4L2 capture →
// NVJPEG decode → TRT YOLOX/RTMPose → Crow WebSocket publish). The CLI
// surface here is intentionally tiny — CLI11 lands when Phase 1 arrives.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferVersion.h>

#include "util/cuda_check.hpp"
#include "util/logging.hpp"

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > Severity::kWARNING) return;  // drop INFO/VERBOSE noise
        using S = Severity;
        const char* tag = "trt";
        switch (severity) {
            case S::kINTERNAL_ERROR: FITRA_LOG_ERROR("[{}] INTERNAL: {}", tag, msg); return;
            case S::kERROR:          FITRA_LOG_ERROR("[{}] {}",          tag, msg); return;
            case S::kWARNING:        FITRA_LOG_WARN ("[{}] {}",          tag, msg); return;
            case S::kINFO:           FITRA_LOG_INFO ("[{}] {}",          tag, msg); return;
            case S::kVERBOSE:        FITRA_LOG_TRACE("[{}] {}",          tag, msg); return;
        }
    }
};

void print_help() {
    std::printf(
        "fitra-cam (C++) — Phase 0 skeleton\n"
        "\n"
        "Usage: fitra-cam [--help] [--probe]\n"
        "\n"
        "  --help    show this help and exit\n"
        "  --probe   print CUDA + TensorRT diagnostics and exit (default)\n"
        "\n"
        "Phases 1-4 will add: build_engines / correctness_check / pose pipeline / WebSocket server.\n"
        "See docs/cpp-migration-plan.md\n");
}

int probe() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    FITRA_LOG_INFO("CUDA device count = {}", device_count);
    if (device_count == 0) {
        FITRA_LOG_ERROR("no CUDA devices available");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        FITRA_LOG_INFO("  [{}] {} (sm_{}{}, {} MB global mem)",
                       i, prop.name, prop.major, prop.minor,
                       static_cast<unsigned long long>(prop.totalGlobalMem) / (1024ULL * 1024ULL));
    }

    int driver_v = 0;
    int runtime_v = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_v));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_v));
    FITRA_LOG_INFO("CUDA driver {}.{} / runtime {}.{}",
                   driver_v / 1000, (driver_v % 100) / 10,
                   runtime_v / 1000, (runtime_v % 100) / 10);

    FITRA_LOG_INFO("TensorRT headers: {}.{}.{}",
                   NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

    TrtLogger trt_logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(trt_logger)};
    TRT_CHECK(runtime != nullptr);
    FITRA_LOG_INFO("TensorRT IRuntime created (lib build: {})",
                   getInferLibVersion());
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    bool want_help = false;
    bool want_probe = (argc <= 1);  // default action

    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        if (a == "--help" || a == "-h") {
            want_help = true;
        } else if (a == "--probe") {
            want_probe = true;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            print_help();
            return EXIT_FAILURE;
        }
    }

    if (want_help) {
        print_help();
        return EXIT_SUCCESS;
    }

    try {
        if (want_probe) {
            return probe();
        }
    } catch (const std::exception& e) {
        FITRA_LOG_ERROR("fatal: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
