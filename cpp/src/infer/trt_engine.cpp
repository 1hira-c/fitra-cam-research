#include "infer/trt_engine.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "util/cuda_check.hpp"
#include "util/logging.hpp"

namespace fitra::infer {

std::size_t dtype_bytes(nvinfer1::DataType dt) {
    using nvinfer1::DataType;
    switch (dt) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL:  return 1;
        case DataType::kUINT8: return 1;
        case DataType::kFP8:   return 1;
        case DataType::kBF16:  return 2;
        case DataType::kINT64: return 8;
        case DataType::kINT4:  return 1;  // packed; callers must adjust
    }
    return 0;
}

std::size_t volume(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) return 0;
    std::size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) return 0;  // dynamic dim -> not yet sized
        v *= static_cast<std::size_t>(dims.d[i]);
    }
    return v;
}

bool all_dims_resolved(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) return false;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) return false;
    }
    return true;
}

static std::string dims_to_string(const nvinfer1::Dims& d) {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) oss << ",";
        oss << d.d[i];
    }
    oss << ")";
    return oss.str();
}

// --- IOutputAllocator that writes back into a TensorBinding -------------

class TrtEngine::BindingOutputAllocator : public nvinfer1::IOutputAllocator {
public:
    explicit BindingOutputAllocator(TensorBinding& binding) : b_{binding} {}

    void* reallocateOutputAsync(const char* /*tensorName*/,
                                void* /*currentMemory*/,
                                std::uint64_t size,
                                std::uint64_t /*alignment*/,
                                cudaStream_t /*stream*/) noexcept override {
        // Round size up to keep allocation churn down. cudaMalloc itself is
        // 256-byte aligned, so alignment is satisfied without extra work.
        if (size == 0) {
            return b_.device_ptr;
        }
        if (size <= b_.bytes && b_.device_ptr) {
            return b_.device_ptr;
        }
        if (b_.device_ptr) {
            cudaFree(b_.device_ptr);
            b_.device_ptr = nullptr;
            b_.bytes = 0;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, size) != cudaSuccess) {
            return nullptr;
        }
        b_.device_ptr = p;
        b_.bytes      = static_cast<std::size_t>(size);
        return p;
    }

    void notifyShape(const char* /*tensorName*/,
                     const nvinfer1::Dims& dims) noexcept override {
        b_.dims = dims;
    }

private:
    TensorBinding& b_;
};

// --- ctor / dtor ---------------------------------------------------------

TrtEngine::TrtEngine(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                     std::unique_ptr<nvinfer1::IExecutionContext> context,
                     cudaStream_t stream)
    : engine_{std::move(engine)},
      context_{std::move(context)},
      stream_{stream} {
    reset_bindings_from_engine();
    bind_addresses();
}

TrtEngine::~TrtEngine() {
    for (auto& b : bindings_) {
        if (b.device_ptr) {
            cudaFree(b.device_ptr);
            b.device_ptr = nullptr;
        }
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    // The IExecutionContext is destroyed by its unique_ptr deleter, and
    // the shared engine is released when the last TrtEngine using it goes
    // away (engine destroy via TRT's operator delete).
}

// --- factory -------------------------------------------------------------

std::shared_ptr<nvinfer1::ICudaEngine> TrtEngine::load_shared(
    nvinfer1::IRuntime& runtime, const std::string& engine_path) {
    std::ifstream f{engine_path, std::ios::binary | std::ios::ate};
    if (!f.is_open()) {
        throw std::runtime_error("failed to open engine file: " + engine_path);
    }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<std::size_t>(sz));
    if (!f.read(blob.data(), sz)) {
        throw std::runtime_error("failed to read engine file: " + engine_path);
    }

    nvinfer1::ICudaEngine* raw =
        runtime.deserializeCudaEngine(blob.data(), blob.size());
    TRT_CHECK(raw != nullptr);
    return std::shared_ptr<nvinfer1::ICudaEngine>{raw};
}

std::unique_ptr<TrtEngine> TrtEngine::from_shared(
    std::shared_ptr<nvinfer1::ICudaEngine> engine) {
    TRT_CHECK(engine != nullptr);
    std::unique_ptr<nvinfer1::IExecutionContext> ctx{
        engine->createExecutionContext()};
    TRT_CHECK(ctx != nullptr);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    return std::unique_ptr<TrtEngine>(
        new TrtEngine(std::move(engine), std::move(ctx), stream));
}

std::unique_ptr<TrtEngine> TrtEngine::from_file(nvinfer1::IRuntime& runtime,
                                                const std::string& engine_path,
                                                nvinfer1::ILogger& /*logger*/) {
    return from_shared(load_shared(runtime, engine_path));
}

// --- bindings ------------------------------------------------------------

bool TrtEngine::has_dynamic_dims(const nvinfer1::Dims& dims) const {
    if (dims.nbDims < 0) return true;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) return true;
    }
    return false;
}

void TrtEngine::install_output_allocator(TensorBinding& b) {
    // Locate the allocator slot that corresponds to this binding.
    auto idx = static_cast<std::size_t>(&b - &bindings_[0]);
    if (idx >= allocators_.size()) {
        throw std::runtime_error("binding/allocator index out of range");
    }
    allocators_[idx] = std::make_unique<BindingOutputAllocator>(b);
    TRT_CHECK(context_->setOutputAllocator(b.name.c_str(), allocators_[idx].get()));
}

void TrtEngine::reset_bindings_from_engine() {
    bindings_.clear();
    allocators_.clear();
    int n = engine_->getNbIOTensors();
    bindings_.reserve(static_cast<std::size_t>(n));
    allocators_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        TensorBinding b{};
        b.name      = name;
        b.dtype     = engine_->getTensorDataType(name);
        b.is_input  = (engine_->getTensorIOMode(name)
                       == nvinfer1::TensorIOMode::kINPUT);
        b.dims      = engine_->getTensorShape(name);
        b.bytes     = 0;
        b.device_ptr= nullptr;
        if (b.is_input && all_dims_resolved(b.dims)) {
            // Static input: tell the context and pre-allocate the device buffer.
            TRT_CHECK(context_->setInputShape(b.name.c_str(), b.dims));
            resize_tensor_buffer(b, b.dims);
        }
        bindings_.push_back(std::move(b));
        allocators_.emplace_back();  // nullptr placeholder
    }

    // All outputs use an IOutputAllocator. This unifies static and
    // data-dependent shapes: TRT calls reallocateOutputAsync with the
    // actual size at enqueue time, and notifyShape writes the resolved
    // dims into the binding so current_shape() is authoritative.
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        auto& b = bindings_[i];
        if (b.is_input) continue;
        install_output_allocator(b);
        // If the output shape is already statically resolved (engine has
        // only static inputs), pre-size the buffer so the first enqueue
        // doesn't pay an allocation.
        auto shape = context_->getTensorShape(b.name.c_str());
        if (!has_dynamic_dims(shape)) {
            resize_tensor_buffer(b, shape);
        }
    }
}

void TrtEngine::resize_tensor_buffer(TensorBinding& b, const nvinfer1::Dims& dims) {
    std::size_t want = volume(dims) * dtype_bytes(b.dtype);
    if (want == 0) return;
    if (want == b.bytes && b.device_ptr) return;
    if (b.device_ptr) {
        CUDA_CHECK(cudaFree(b.device_ptr));
        b.device_ptr = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&b.device_ptr, want));
    b.bytes = want;
    b.dims  = dims;
}

void TrtEngine::bind_addresses() {
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        const auto& b = bindings_[i];
        // Outputs are exclusively allocator-managed (setOutputAllocator
        // is incompatible with setTensorAddress).
        if (!b.is_input) continue;
        if (b.device_ptr) {
            TRT_CHECK(context_->setTensorAddress(b.name.c_str(), b.device_ptr));
        }
    }
}

const TensorBinding& TrtEngine::binding(const std::string& name) const {
    auto it = std::find_if(bindings_.begin(), bindings_.end(),
                           [&](const auto& b) { return b.name == name; });
    if (it == bindings_.end()) {
        throw std::runtime_error("no binding named '" + name + "'");
    }
    return *it;
}

void TrtEngine::set_input_shape(const std::string& name, const nvinfer1::Dims& dims) {
    auto it = std::find_if(bindings_.begin(), bindings_.end(),
                           [&](const auto& b) { return b.name == name && b.is_input; });
    if (it == bindings_.end()) {
        throw std::runtime_error("no input binding named '" + name + "'");
    }
    TRT_CHECK(context_->setInputShape(name.c_str(), dims));
    resize_tensor_buffer(*it, dims);
    TRT_CHECK(context_->setTensorAddress(it->name.c_str(), it->device_ptr));

    // Outputs are allocator-managed; TRT will resize them at enqueue time.
    // Pre-size statically resolvable outputs to avoid an allocation on
    // first inference.
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        auto& b = bindings_[i];
        if (b.is_input) continue;
        auto shape = context_->getTensorShape(b.name.c_str());
        if (all_dims_resolved(shape)) {
            resize_tensor_buffer(b, shape);
        }
    }
}

// --- transfers -----------------------------------------------------------

static TensorBinding& mut_binding_by_name(std::vector<TensorBinding>& bs,
                                          const std::string& name) {
    auto it = std::find_if(bs.begin(), bs.end(),
                           [&](const auto& b) { return b.name == name; });
    if (it == bs.end()) {
        throw std::runtime_error("no binding named '" + name + "'");
    }
    return *it;
}

void TrtEngine::copy_input_from_host(const std::string& name,
                                     const void* host, std::size_t bytes) {
    auto& b = mut_binding_by_name(bindings_, name);
    if (!b.is_input) {
        throw std::runtime_error("binding '" + name + "' is not an input");
    }
    if (bytes != b.bytes) {
        std::ostringstream oss;
        oss << "input '" << name << "' size mismatch: host=" << bytes
            << " device=" << b.bytes << " dims=" << dims_to_string(b.dims);
        throw std::runtime_error(oss.str());
    }
    CUDA_CHECK(cudaMemcpyAsync(b.device_ptr, host, bytes,
                               cudaMemcpyHostToDevice, stream_));
}

void TrtEngine::copy_output_to_host(const std::string& name,
                                    void* host, std::size_t bytes) {
    auto& b = mut_binding_by_name(bindings_, name);
    if (b.is_input) {
        throw std::runtime_error("binding '" + name + "' is an input");
    }
    if (bytes > b.bytes) {
        std::ostringstream oss;
        oss << "output '" << name << "' read overruns device buffer: host="
            << bytes << " device(max)=" << b.bytes;
        throw std::runtime_error(oss.str());
    }
    CUDA_CHECK(cudaMemcpyAsync(host, b.device_ptr, bytes,
                               cudaMemcpyDeviceToHost, stream_));
}

nvinfer1::Dims TrtEngine::current_shape(const std::string& name) const {
    // Outputs are allocator-managed: notifyShape updates binding.dims with
    // the resolved shape after each enqueue. Inputs: binding.dims tracks
    // whatever the caller set most recently (or the engine's static shape).
    for (const auto& b : bindings_) {
        if (b.name == name) return b.dims;
    }
    return context_->getTensorShape(name.c_str());
}

// --- enqueue -------------------------------------------------------------

void TrtEngine::enqueue() {
    TRT_CHECK(context_->enqueueV3(stream_));
}

void TrtEngine::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace fitra::infer
