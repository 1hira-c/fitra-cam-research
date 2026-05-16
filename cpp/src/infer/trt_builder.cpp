#include "infer/trt_builder.hpp"

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <NvOnnxParser.h>

#include "infer/int8_calibrator.hpp"
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

static std::size_t dtype_size(nvinfer1::DataType dtype) {
    using nvinfer1::DataType;
    switch (dtype) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL:  return 1;
        case DataType::kUINT8: return 1;
        case DataType::kFP8:   return 1;
        case DataType::kBF16:  return 2;
        case DataType::kINT64: return 8;
        case DataType::kINT4:  return 1;
    }
    throw std::runtime_error("unsupported TensorRT data type");
}

static std::size_t volume(const nvinfer1::Dims& d) {
    std::size_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] <= 0) {
            throw std::runtime_error("cannot compute volume for dynamic dims "
                                     + dims_str(d));
        }
        n *= static_cast<std::size_t>(d.d[i]);
    }
    return n;
}

static bool has_dynamic_dim(const nvinfer1::Dims& d) {
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) return true;
    }
    return false;
}

static nvinfer1::Dims calibration_dims_for(const nvinfer1::Dims& network_dims,
                                           const DynamicProfile* profile,
                                           int batch_size) {
    nvinfer1::Dims d = network_dims;
    if (profile) {
        d = profile->opt_dims;
    }
    if (d.nbDims > 0 && batch_size > 0) {
        d.d[0] = batch_size;
    }
    return d;
}

static const DynamicProfile* find_profile_for_input(const BuildOptions& opts,
                                                    const std::string& input_name) {
    for (const auto& p : opts.profiles) {
        if (p.input_name == input_name) return &p;
    }
    return nullptr;
}

std::size_t build_engine(const BuildOptions& opts, nvinfer1::ILogger& logger) {
    FITRA_LOG_INFO("build_engine: onnx={} -> engine={} (fp16={}, int8={}, ws={}MB)",
                   opts.onnx_path, opts.engine_path,
                   opts.fp16 ? "yes" : "no",
                   opts.int8 ? "yes" : "no",
                   opts.workspace_mb);

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
        if (opts.int8_blob_path.empty()) {
            throw std::runtime_error("--int8 requires --int8-blobs PATH");
        }
        if (network->getNbInputs() != 1) {
            throw std::runtime_error("INT8 calibration currently expects exactly one network input");
        }
        if (!builder->platformHasFastInt8()) {
            FITRA_LOG_WARN("platform does not advertise fast INT8; building anyway");
        }
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

    nvinfer1::IOptimizationProfile* first_profile = nullptr;
    if (!opts.profiles.empty()) {
        auto* profile = builder->createOptimizationProfile();
        first_profile = profile;
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

    std::unique_ptr<Int8EntropyCalibrator2> int8_calibrator;
    if (opts.int8) {
        auto* input = network->getInput(0);
        std::string input_name = input->getName();
        if (input->getType() != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("INT8 calibration blobs currently expect a float32 model input");
        }
        const auto* dyn_profile = find_profile_for_input(opts, input_name);
        auto calib_dims = calibration_dims_for(input->getDimensions(),
                                               dyn_profile,
                                               opts.int8_batch_size);
        if (has_dynamic_dim(calib_dims)) {
            throw std::runtime_error("INT8 calibration dims remain dynamic for input "
                                     + input_name + ": " + dims_str(calib_dims));
        }
        std::size_t batch_elems = volume(calib_dims);
        std::size_t batch_size = static_cast<std::size_t>(opts.int8_batch_size);
        std::size_t sample_elems = batch_elems / batch_size;
        std::size_t sample_bytes = sample_elems * dtype_size(input->getType());
        std::string cache_path = opts.int8_cache_path.empty()
            ? (opts.engine_path + ".calib.cache")
            : opts.int8_cache_path;

        if (first_profile) {
            auto* calib_profile = builder->createOptimizationProfile();
            for (const auto& p : opts.profiles) {
                using nvinfer1::OptProfileSelector;
                nvinfer1::Dims d = calibration_dims_for(p.opt_dims, &p,
                                                         opts.int8_batch_size);
                calib_profile->setDimensions(p.input_name.c_str(), OptProfileSelector::kMIN, d);
                calib_profile->setDimensions(p.input_name.c_str(), OptProfileSelector::kOPT, d);
                calib_profile->setDimensions(p.input_name.c_str(), OptProfileSelector::kMAX, d);
                FITRA_LOG_INFO("  calibration profile: {} shape={}",
                               p.input_name, dims_str(d));
            }
            TRT_CHECK(config->setCalibrationProfile(calib_profile));
        }

        int8_calibrator = std::make_unique<Int8EntropyCalibrator2>(
            opts.int8_blob_path,
            sample_bytes,
            opts.int8_batch_size,
            input_name,
            cache_path);
        config->setInt8Calibrator(int8_calibrator.get());
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
