#include "pipeline/snapshot.hpp"

#include <cstdio>
#include <ctime>
#include <sstream>

namespace fitra::pipeline {

namespace {

void append_float(std::string& out, double v, int precision = 6) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    out += buf;
}

}  // namespace

SnapshotBus::SnapshotBus(std::size_t n_cameras) : snapshots_(n_cameras) {
    for (std::size_t i = 0; i < n_cameras; ++i) {
        snapshots_[i].id = static_cast<int>(i);
    }
}

void SnapshotBus::update(const CameraSnapshot& s) {
    if (static_cast<std::size_t>(s.id) >= snapshots_.size()) return;
    std::lock_guard<std::mutex> lk{mu_};
    snapshots_[static_cast<std::size_t>(s.id)] = s;
}

std::string SnapshotBus::make_bundle_json() {
    using clock = std::chrono::system_clock;
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lk{mu_};
    ++bundle_seq_;

    std::string out;
    out.reserve(2048);
    out += "{\"seq\":";
    out += std::to_string(bundle_seq_);
    out += ",\"ts_ms\":";
    out += std::to_string(static_cast<long long>(now_ms));
    out += ",\"cameras\":[";
    for (std::size_t i = 0; i < snapshots_.size(); ++i) {
        if (i) out += ",";
        const auto& s = snapshots_[i];
        out += "{\"id\":";
        out += std::to_string(s.id);
        out += ",\"w\":";
        out += std::to_string(s.w);
        out += ",\"h\":";
        out += std::to_string(s.h);

        out += ",\"persons\":[";
        for (std::size_t pi = 0; pi < s.persons.size(); ++pi) {
            if (pi) out += ",";
            const auto& p = s.persons[pi];
            out += "{\"kpts\":[";
            for (std::size_t k = 0; k < p.kpts.size(); ++k) {
                if (k) out += ",";
                out += "[";
                append_float(out, p.kpts[k].x);     out += ",";
                append_float(out, p.kpts[k].y);     out += ",";
                append_float(out, p.kpts[k].score);
                out += "]";
            }
            out += "]";
            // bbox aligned with infer::Person::bbox (the Python publisher
            // includes the bbox alongside kpts when available).
            if (pi < s.bboxes.size()) {
                const auto& bb = s.bboxes[pi];
                out += ",\"bbox\":[";
                append_float(out, bb.x1);    out += ",";
                append_float(out, bb.y1);    out += ",";
                append_float(out, bb.x2);    out += ",";
                append_float(out, bb.y2);    out += ",";
                append_float(out, bb.score); out += "]";
            }
            out += "}";
        }
        out += "]";

        out += ",\"stats\":{";
        out += "\"recv_fps\":";       append_float(out, s.recv_fps, 4);
        out += ",\"recent_pose_fps\":"; append_float(out, s.recent_pose_fps, 4);
        out += ",\"avg_pose_fps\":";    append_float(out, s.avg_pose_fps, 4);
        out += ",\"pending\":";         out += std::to_string(static_cast<long long>(s.pending));
        out += ",\"stage_ms\":";        append_float(out, s.stage_ms, 4);
        out += ",\"processed\":";       out += std::to_string(static_cast<long long>(s.processed));
        out += ",\"captured_at_ms\":";
        auto cap_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          s.captured_wall.time_since_epoch()).count();
        out += std::to_string(static_cast<long long>(cap_ms));
        out += "}";
        out += "}";
    }
    out += "]}";
    return out;
}

}  // namespace fitra::pipeline
