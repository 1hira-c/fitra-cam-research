#pragma once
//
// Crow HTTP + WebSocket server.
//
// Mirrors python/scripts/dual_rtmpose_web.py:
//   - GET /            serves web/dual_rtmpose/index.html
//   - GET /<path>      serves files under web/dual_rtmpose/
//   - GET /stats       returns the current bundle as JSON
//   - WS  /ws          broadcasts the bundle at ≤30 Hz to every client
//
// The publisher loop runs on its own thread; Crow's worker pool handles
// the HTTP request and WS plumbing.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "pipeline/snapshot.hpp"

namespace fitra::web {

struct ServerOptions {
    std::string host = "0.0.0.0";
    int         port = 8000;
    std::string static_dir;        // absolute path to web/dual_rtmpose/
    double      publish_hz = 30.0;
    int         crow_threads = 2;
};

class CrowServer {
public:
    CrowServer(pipeline::SnapshotBus& bus, ServerOptions opts);
    ~CrowServer();

    CrowServer(const CrowServer&) = delete;
    CrowServer& operator=(const CrowServer&) = delete;

    // Start listening + broadcasting on a background thread. Returns when
    // the server is bound and ready (best-effort; Crow's run() blocks).
    void start();
    void stop();

private:
    void publisher_loop();

    pipeline::SnapshotBus& bus_;
    ServerOptions          opts_;
    std::thread            server_thread_;
    std::thread            publisher_thread_;
    std::atomic<bool>      stop_{false};

    struct Impl;
    std::unique_ptr<Impl>  impl_;
};

}  // namespace fitra::web
