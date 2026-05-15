#pragma once
//
// CUDA / TensorRT error handling helpers.
//
// CUDA_CHECK(expr)   - throws std::runtime_error on non-cudaSuccess
// TRT_CHECK(expr)    - throws std::runtime_error if expression evaluates falsy
//                      (TensorRT 10 APIs return nullptr / false on failure)
//

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace fitra::util {

[[noreturn]] inline void throw_cuda_error(cudaError_t err, const char* expr,
                                          const char* file, int line) {
    std::ostringstream oss;
    oss << "CUDA error " << static_cast<int>(err) << " ("
        << cudaGetErrorString(err) << ") at " << file << ":" << line
        << " in `" << expr << "`";
    throw std::runtime_error(oss.str());
}

[[noreturn]] inline void throw_trt_error(const char* expr, const char* file,
                                         int line) {
    std::ostringstream oss;
    oss << "TensorRT call returned failure at " << file << ":" << line
        << " in `" << expr << "`";
    throw std::runtime_error(oss.str());
}

}  // namespace fitra::util

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            ::fitra::util::throw_cuda_error(_e, #expr, __FILE__, __LINE__);    \
        }                                                                      \
    } while (0)

#define TRT_CHECK(expr)                                                        \
    do {                                                                       \
        if (!(expr)) {                                                         \
            ::fitra::util::throw_trt_error(#expr, __FILE__, __LINE__);         \
        }                                                                      \
    } while (0)
