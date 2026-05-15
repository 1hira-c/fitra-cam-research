#include "web/crow_server.hpp"

#include <chrono>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>

#define CROW_MAIN
#include <crow.h>

#include "util/logging.hpp"

namespace fitra::web {

namespace {

struct WsClients {
    std::mutex                       mu;
    std::set<crow::websocket::connection*> conns;
};

std::string read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) return {};
    std::ostringstream oss;
    oss << f.rdbuf();
    return oss.str();
}

std::string guess_content_type(const std::filesystem::path& p) {
    auto ext = p.extension().string();
    if (ext == ".html") return "text/html; charset=utf-8";
    if (ext == ".js")   return "application/javascript; charset=utf-8";
    if (ext == ".css")  return "text/css; charset=utf-8";
    if (ext == ".json") return "application/json; charset=utf-8";
    if (ext == ".png")  return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    return "application/octet-stream";
}

}  // namespace

struct CrowServer::Impl {
    crow::SimpleApp app;
    WsClients       clients;
};

CrowServer::CrowServer(pipeline::SnapshotBus& bus, ServerOptions opts)
    : bus_{bus}, opts_{std::move(opts)}, impl_{std::make_unique<Impl>()} {}

CrowServer::~CrowServer() {
    try { stop(); } catch (...) {}
}

void CrowServer::start() {
    auto& app     = impl_->app;
    auto& clients = impl_->clients;

    // WS /ws — register first so the catch-all HTTP route below does not
    // shadow upgrade requests (Crow's BaseRule::handle_upgrade returns
    // 404 without writing it, which the client sees as a closed socket).
    CROW_WEBSOCKET_ROUTE(app, "/ws")
    .onopen([&clients](crow::websocket::connection& c) {
        std::lock_guard<std::mutex> lk{clients.mu};
        clients.conns.insert(&c);
    })
    .onclose([&clients](crow::websocket::connection& c,
                       const std::string& /*reason*/,
                       uint16_t /*code*/) {
        std::lock_guard<std::mutex> lk{clients.mu};
        clients.conns.erase(&c);
    })
    .onmessage([](crow::websocket::connection& /*c*/,
                  const std::string& /*data*/,
                  bool /*is_binary*/) {
        // ignore client messages (ping etc.)
    });

    // GET /stats — current bundle as JSON
    CROW_ROUTE(app, "/stats")
    ([this]() {
        crow::response resp{bus_.make_bundle_json()};
        resp.set_header("Content-Type", "application/json; charset=utf-8");
        return resp;
    });

    // Static files under opts_.static_dir
    std::filesystem::path static_root{opts_.static_dir};
    CROW_ROUTE(app, "/")
    ([static_root]() {
        auto body = read_file(static_root / "index.html");
        if (body.empty()) {
            return crow::response{404, "index.html not found"};
        }
        crow::response resp{body};
        resp.set_header("Content-Type", "text/html; charset=utf-8");
        return resp;
    });

    CROW_ROUTE(app, "/<path>")
    ([static_root](const std::string& sub) {
        // Reject anything that escapes the static root.
        std::filesystem::path req = static_root / sub;
        auto canon_req  = std::filesystem::weakly_canonical(req);
        auto canon_root = std::filesystem::weakly_canonical(static_root);
        auto root_str = canon_root.string();
        if (canon_req.string().rfind(root_str, 0) != 0) {
            return crow::response{403, "forbidden"};
        }
        if (!std::filesystem::is_regular_file(canon_req)) {
            return crow::response{404, "not found"};
        }
        auto body = read_file(canon_req);
        crow::response resp{body};
        resp.set_header("Content-Type", guess_content_type(canon_req));
        return resp;
    });

    server_thread_ = std::thread{[this]() {
        impl_->app
            // Crow installs SIGINT/SIGTERM by default and stops itself when
            // they fire. We want our own main() handler to drive the whole
            // shutdown (driver -> server), so clear Crow's handlers.
            .signal_clear()
            .loglevel(crow::LogLevel::Warning)
            .concurrency(static_cast<std::uint16_t>(opts_.crow_threads))
            .port(static_cast<std::uint16_t>(opts_.port))
            .bindaddr(opts_.host)
            .run();
    }};

    stop_.store(false);
    publisher_thread_ = std::thread{&CrowServer::publisher_loop, this};

    FITRA_LOG_INFO("crow listening on http://{}:{}/", opts_.host, opts_.port);
}

void CrowServer::stop() {
    if (!server_thread_.joinable() && !publisher_thread_.joinable()) return;
    stop_.store(true);
    if (impl_) impl_->app.stop();
    if (publisher_thread_.joinable()) publisher_thread_.join();
    if (server_thread_.joinable())    server_thread_.join();
}

void CrowServer::publisher_loop() {
    using clock = std::chrono::steady_clock;
    auto period = std::chrono::duration<double>(1.0 / std::max(opts_.publish_hz, 1.0));
    auto next = clock::now();
    while (!stop_.load()) {
        next += std::chrono::duration_cast<clock::duration>(period);
        std::this_thread::sleep_until(next);
        if (stop_.load()) break;

        auto msg = bus_.make_bundle_json();
        std::lock_guard<std::mutex> lk{impl_->clients.mu};
        for (auto* c : impl_->clients.conns) {
            try {
                c->send_text(msg);
            } catch (...) {
                // best-effort; client will be reaped on close
            }
        }
    }
}

}  // namespace fitra::web
