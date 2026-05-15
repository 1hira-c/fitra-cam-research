#pragma once
//
// Minimal logging facade.
//
// Phase 0 implementation: writes "[level] message" to std::cerr.
// Phase 1+ may swap the backend to spdlog without changing call sites.
//
// Usage:
//   FITRA_LOG_INFO("opened camera {}", cam_id);
//   FITRA_LOG_WARN("plain string also works");
//

#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace fitra::log {

enum class Level { trace, debug, info, warn, error };

inline const char* level_tag(Level lv) {
    switch (lv) {
        case Level::trace: return "trace";
        case Level::debug: return "debug";
        case Level::info:  return "info";
        case Level::warn:  return "warn";
        case Level::error: return "error";
    }
    return "?";
}

namespace detail {

inline std::mutex& sink_mutex() {
    static std::mutex m;
    return m;
}

inline void emit_one(std::ostringstream&) {}

template <typename T, typename... Rest>
void emit_one(std::ostringstream& oss, T&& head, Rest&&... rest) {
    oss << std::forward<T>(head);
    emit_one(oss, std::forward<Rest>(rest)...);
}

// Very small "{}" substitution. Not printf, not std::format. Enough to keep
// call sites readable in Phase 0 without depending on libfmt/spdlog yet.
inline std::string render(std::string_view fmt) {
    return std::string{fmt};
}

template <typename T, typename... Rest>
std::string render(std::string_view fmt, T&& arg, Rest&&... rest) {
    auto pos = fmt.find("{}");
    if (pos == std::string_view::npos) {
        return std::string{fmt};
    }
    std::ostringstream oss;
    oss << fmt.substr(0, pos);
    oss << std::forward<T>(arg);
    std::string tail{fmt.substr(pos + 2)};
    return oss.str() + render(tail, std::forward<Rest>(rest)...);
}

template <typename... Args>
void log_impl(Level lv, std::string_view fmt, Args&&... args) {
    std::string body = render(fmt, std::forward<Args>(args)...);
    std::lock_guard<std::mutex> lock{sink_mutex()};
    std::cerr << "[" << level_tag(lv) << "] " << body << '\n';
}

}  // namespace detail
}  // namespace fitra::log

#define FITRA_LOG_TRACE(...) ::fitra::log::detail::log_impl(::fitra::log::Level::trace, __VA_ARGS__)
#define FITRA_LOG_DEBUG(...) ::fitra::log::detail::log_impl(::fitra::log::Level::debug, __VA_ARGS__)
#define FITRA_LOG_INFO(...)  ::fitra::log::detail::log_impl(::fitra::log::Level::info,  __VA_ARGS__)
#define FITRA_LOG_WARN(...)  ::fitra::log::detail::log_impl(::fitra::log::Level::warn,  __VA_ARGS__)
#define FITRA_LOG_ERROR(...) ::fitra::log::detail::log_impl(::fitra::log::Level::error, __VA_ARGS__)
