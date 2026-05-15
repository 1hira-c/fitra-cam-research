#pragma once
//
// Thin wrapper around an nvinfer1::ICudaEngine + IExecutionContext.
//
// Owns:
//   - one execution context (single-threaded use)
//   - device buffers for every IO tensor
//   - a CUDA stream
//
// Shares (via shared_ptr):
//   - the deserialized ICudaEngine. Multiple TrtEngine instances can wrap
//     the same engine, each with their own context — required for
//     per-camera parallel inference (TRT contexts are not thread-safe).
//
// Does NOT own:
//   - the IRuntime (callers pass one in; runtimes are cheap to share)
//   - the TRT logger
//
// The shape policy is "static unless told otherwise". For dynamic-shape
// engines, call set_input_shape() before the first enqueue.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

namespace fitra::infer {

struct TensorBinding {
    std::string  name;
    nvinfer1::DataType dtype;
    nvinfer1::Dims     dims;        // -1 entries mean "dynamic, set at runtime"
    bool         is_input;
    std::size_t  bytes;             // current allocation size (set after shape known)
    void*        device_ptr;        // owned by TrtEngine
};

class TrtEngine {
public:
    // Build an engine from a serialized .plan/.engine file. Returns a
    // TrtEngine that owns its own copy of the deserialized network.
    static std::unique_ptr<TrtEngine> from_file(nvinfer1::IRuntime& runtime,
                                                const std::string& engine_path,
                                                nvinfer1::ILogger& logger);

    // Load just the deserialized network so several TrtEngines can share
    // it. Each TrtEngine still holds its own IExecutionContext, stream and
    // device buffers — only the (large, read-only) engine is shared.
    static std::shared_ptr<nvinfer1::ICudaEngine> load_shared(
        nvinfer1::IRuntime& runtime,
        const std::string& engine_path);

    // Build a TrtEngine around a pre-loaded ICudaEngine. The engine is
    // shared via shared_ptr so callers can construct N TrtEngines (one per
    // thread) sharing one ICudaEngine.
    static std::unique_ptr<TrtEngine> from_shared(
        std::shared_ptr<nvinfer1::ICudaEngine> engine);

    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // --- introspection ---
    nvinfer1::ICudaEngine&       engine()        { return *engine_; }
    nvinfer1::IExecutionContext& context()       { return *context_; }
    cudaStream_t                 stream() const  { return stream_; }
    const std::vector<TensorBinding>& bindings() const { return bindings_; }
    const TensorBinding& binding(const std::string& name) const;

    // --- dynamic shape ---
    // Set the runtime shape for a dynamic input. Reallocates the device
    // buffer for that tensor + any output buffers whose shape becomes
    // resolvable after this call.
    void set_input_shape(const std::string& name, const nvinfer1::Dims& dims);

    // --- IO transfer helpers ---
    void copy_input_from_host(const std::string& name,
                              const void* host, std::size_t bytes);
    void copy_output_to_host(const std::string& name,
                             void* host, std::size_t bytes);

    // The context's view of the named tensor's shape *right now*. For
    // dynamic-shape outputs the dims become fully resolved only after
    // enqueue + synchronize.
    nvinfer1::Dims current_shape(const std::string& name) const;

    // --- enqueue + synchronize ---
    // Returns when work is enqueued. Caller can submit further work to the
    // same stream, or call synchronize() to wait.
    void enqueue();
    void synchronize();

private:
    // IOutputAllocator that writes the resolved pointer + shape back into a
    // TensorBinding. Used for outputs whose engine-declared shape contains
    // -1 dims, where TRT only reports the runtime size via notifyShape.
    class BindingOutputAllocator;

    TrtEngine(std::shared_ptr<nvinfer1::ICudaEngine> engine,
              std::unique_ptr<nvinfer1::IExecutionContext> context,
              cudaStream_t stream);

    void reset_bindings_from_engine();
    void resize_tensor_buffer(TensorBinding& b, const nvinfer1::Dims& dims);
    bool has_dynamic_dims(const nvinfer1::Dims& dims) const;
    void install_output_allocator(TensorBinding& b);
    void bind_addresses();

    std::shared_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t                                 stream_;
    std::vector<TensorBinding>                   bindings_;
    // Parallel to bindings_; nullptr for inputs and static-shape outputs.
    std::vector<std::unique_ptr<BindingOutputAllocator>> allocators_;
};

// Helpers
std::size_t dtype_bytes(nvinfer1::DataType dt);
std::size_t volume(const nvinfer1::Dims& dims);
bool        all_dims_resolved(const nvinfer1::Dims& dims);

}  // namespace fitra::infer
