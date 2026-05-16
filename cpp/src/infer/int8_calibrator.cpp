#include "infer/int8_calibrator.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <cuda_runtime_api.h>

#include "util/cuda_check.hpp"
#include "util/logging.hpp"

namespace fitra::infer {

namespace {

std::vector<std::uint8_t> read_binary_file(const std::string& path) {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    if (!in.is_open()) {
        throw std::runtime_error("failed to open INT8 calibration blobs: " + path);
    }
    auto size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to stat INT8 calibration blobs: " + path);
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
    }
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("failed to read INT8 calibration blobs: " + path);
    }
    return data;
}

}  // namespace

Int8EntropyCalibrator2::Int8EntropyCalibrator2(
    std::string blob_path,
    std::size_t sample_bytes,
    int batch_size,
    std::string input_name,
    std::string cache_path)
    : blob_path_{std::move(blob_path)},
      input_name_{std::move(input_name)},
      cache_path_{std::move(cache_path)},
      sample_bytes_{sample_bytes},
      batch_size_{batch_size} {
    if (sample_bytes_ == 0) {
        throw std::runtime_error("INT8 calibration sample size is zero");
    }
    if (batch_size_ <= 0) {
        throw std::runtime_error("INT8 calibration batch size must be positive");
    }
    blob_ = read_binary_file(blob_path_);
    if (blob_.empty()) {
        throw std::runtime_error("INT8 calibration blob file is empty: " + blob_path_);
    }
    if ((blob_.size() % sample_bytes_) != 0) {
        throw std::runtime_error(
            "INT8 calibration blob size is not a multiple of one input sample");
    }
    sample_count_ = blob_.size() / sample_bytes_;
    if (sample_count_ < static_cast<std::size_t>(batch_size_)) {
        throw std::runtime_error("not enough INT8 calibration samples for one batch");
    }
    std::size_t batch_bytes = sample_bytes_ * static_cast<std::size_t>(batch_size_);
    CUDA_CHECK(cudaMalloc(&device_input_, batch_bytes));
    FITRA_LOG_INFO(
        "INT8 calibrator: blobs={} samples={} sample_bytes={} batch_size={} usable_batches={} cache={}",
        blob_path_, sample_count_, sample_bytes_, batch_size_, batch_count(),
        cache_path_.empty() ? "<disabled>" : cache_path_);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2() noexcept {
    if (device_input_) {
        cudaFree(device_input_);
        device_input_ = nullptr;
    }
}

int32_t Int8EntropyCalibrator2::getBatchSize() const noexcept {
    return batch_size_;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[],
                                      char const* names[],
                                      int32_t nbBindings) noexcept {
    int binding_index = -1;
    for (int32_t i = 0; i < nbBindings; ++i) {
        if (names[i] && input_name_ == names[i]) {
            binding_index = i;
            break;
        }
    }
    if (binding_index < 0) {
        std::fprintf(stderr, "[fitra] INT8 calibrator input binding not found: %s\n",
                     input_name_.c_str());
        return false;
    }

    std::size_t next = cursor_ + static_cast<std::size_t>(batch_size_);
    if (next > sample_count_) {
        return false;
    }
    std::size_t offset = cursor_ * sample_bytes_;
    std::size_t bytes = sample_bytes_ * static_cast<std::size_t>(batch_size_);
    cudaError_t err = cudaMemcpy(device_input_,
                                 blob_.data() + offset,
                                 bytes,
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[fitra] INT8 calibration cudaMemcpy failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    bindings[binding_index] = device_input_;
    cursor_ = next;
    return true;
}

void const* Int8EntropyCalibrator2::readCalibrationCache(std::size_t& length) noexcept {
    length = 0;
    cache_.clear();
    if (cache_path_.empty()) {
        return nullptr;
    }
    try {
        if (!std::filesystem::exists(cache_path_)) {
            return nullptr;
        }
        cache_ = read_binary_file(cache_path_);
        length = cache_.size();
        FITRA_LOG_INFO("INT8 calibration cache hit: {} ({} bytes)",
                       cache_path_, length);
        return cache_.empty() ? nullptr : cache_.data();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[fitra] failed to read INT8 calibration cache: %s\n",
                     e.what());
        cache_.clear();
        length = 0;
        return nullptr;
    }
}

void Int8EntropyCalibrator2::writeCalibrationCache(void const* ptr,
                                                   std::size_t length) noexcept {
    if (cache_path_.empty() || ptr == nullptr || length == 0) {
        return;
    }
    try {
        std::filesystem::path p{cache_path_};
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream out{cache_path_, std::ios::binary | std::ios::trunc};
        if (!out.is_open()) {
            std::fprintf(stderr, "[fitra] failed to open INT8 calibration cache for write: %s\n",
                         cache_path_.c_str());
            return;
        }
        out.write(reinterpret_cast<char const*>(ptr),
                  static_cast<std::streamsize>(length));
        FITRA_LOG_INFO("wrote INT8 calibration cache: {} ({} bytes)",
                       cache_path_, length);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[fitra] failed to write INT8 calibration cache: %s\n",
                     e.what());
    }
}

}  // namespace fitra::infer
