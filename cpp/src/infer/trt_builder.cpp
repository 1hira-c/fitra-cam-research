#include "infer/trt_builder.hpp"

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <NvOnnxParser.h>

#include "util/cuda_check.hpp"
#include "util/logging.hpp"

namespace fitra::infer {

static std::string dims_str(const nvinfer1::Dims& d) {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) oss << ",";
        oss << d.d[i];
    }
    oss << ")";
    return oss.str();
}

std::size_t build_engine(const BuildOptions& opts, nvinfer1::ILogger& logger) {
    FITRA_LOG_INFO("build_engine: onnx={} -> engine={} (fp16={}, ws={}MB)",
                   opts.onnx_path, opts.engine_path,
                   opts.fp16 ? "yes" : "no", opts.workspace_mb);

    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    TRT_CHECK(builder != nullptr);

    // TRT 10 networks are explicit-batch by default.
    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0)};
    TRT_CHECK(network != nullptr);

    std::unique_ptr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(*network, logger)};
    TRT_CHECK(parser != nullptr);

    if (!parser->parseFromFile(opts.onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            const auto* e = parser->getError(i);
            FITRA_LOG_ERROR("onnx parse [{}]: {}", i, e->desc());
        }
        throw std::runtime_error("ONNX parse failed: " + opts.onnx_path);
    }

    // Log inputs/outputs we parsed
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto* t = network->getInput(i);
        FITRA_LOG_INFO("  input  [{}] {} {}", i, t->getName(),
                       dims_str(t->getDimensions()));
    }
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        auto* t = network->getOutput(i);
        FITRA_LOG_INFO("  output [{}] {} {}", i, t->getName(),
                       dims_str(t->getDimensions()));
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    TRT_CHECK(config != nullptr);

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               static_cast<std::size_t>(opts.workspace_mb) * (1ull << 20));

    if (opts.fp16) {
        if (!builder->platformHasFastFp16()) {
            FITRA_LOG_WARN("platform does not advertise fast FP16; building anyway");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (opts.int8) {
        if (!builder->platformHasFastInt8()) {
            FITRA_LOG_WARN("platform does not advertise fast INT8; building anyway");
        }
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

    if (!opts.profiles.empty()) {
        auto* profile = builder->createOptimizationProfile();
        for (const auto& p : opts.profiles) {
            using nvinfer1::OptProfileSelector;
            profile->setDimensions(p.input_name.c_str(), OptProfileSelector::kMIN, p.min_dims);
            profile->setDimensions(p.input_name.c_str(), OptProfileSelector::kOPT, p.opt_dims);
            profile->setDimensions(p.input_name.c_str(), OptProfileSelector::kMAX, p.max_dims);
            FITRA_LOG_INFO("  profile: {} min={} opt={} max={}",
                           p.input_name,
                           dims_str(p.min_dims),
                           dims_str(p.opt_dims),
                           dims_str(p.max_dims));
        }
        config->addOptimizationProfile(profile);
    }

    FITRA_LOG_INFO("building serialized network (this may take minutes)...");
    std::unique_ptr<nvinfer1::IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    TRT_CHECK(plan != nullptr);
    FITRA_LOG_INFO("serialized network size = {} bytes", plan->size());

    {
        std::ofstream out{opts.engine_path, std::ios::binary | std::ios::trunc};
        if (!out.is_open()) {
            throw std::runtime_error("failed to open engine output: " + opts.engine_path);
        }
        out.write(reinterpret_cast<const char*>(plan->data()),
                  static_cast<std::streamsize>(plan->size()));
    }
    FITRA_LOG_INFO("wrote {}", opts.engine_path);
    return plan->size();
}

}  // namespace fitra::infer
