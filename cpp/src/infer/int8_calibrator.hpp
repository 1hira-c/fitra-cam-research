#pragma once
//
// TensorRT PTQ calibrator for raw float32 input blobs.
//
// The blob file is headerless and stores N contiguous samples for one network
// input. Each sample must already match the model preprocessing, excluding the
// explicit batch dimension.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <NvInfer.h>

namespace fitra::infer {

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator2(std::string blob_path,
                           std::size_t sample_bytes,
                           int batch_size,
                           std::string input_name,
                           std::string cache_path);
    ~Int8EntropyCalibrator2() noexcept override;

    int32_t getBatchSize() const noexcept override;
    bool getBatch(void* bindings[],
                  char const* names[],
                  int32_t nbBindings) noexcept override;
    void const* readCalibrationCache(std::size_t& length) noexcept override;
    void writeCalibrationCache(void const* ptr,
                               std::size_t length) noexcept override;

    std::size_t sample_count() const { return sample_count_; }
    std::size_t batch_count() const { return sample_count_ / batch_size_; }

private:
    std::string blob_path_;
    std::string input_name_;
    std::string cache_path_;
    std::size_t sample_bytes_ = 0;
    int batch_size_ = 1;
    std::size_t sample_count_ = 0;
    std::size_t cursor_ = 0;

    std::vector<std::uint8_t> blob_;
    std::vector<std::uint8_t> cache_;
    void* device_input_ = nullptr;
};

}  // namespace fitra::infer
