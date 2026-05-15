#include "infer/yolox.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "util/cuda_check.hpp"
#include "util/logging.hpp"

namespace fitra::infer {

namespace {

// Letterbox the input BGR image into `dst` (input_size x input_size x 3 uint8),
// returning the scale factor r so that out_x = in_x * r.
//
// Mirrors python/scripts/pose_pipeline.py::_yolox_letterbox exactly:
//   r  = min(target/h, target/w)
//   nh = round(h * r), nw = round(w * r)
//   resized = cv2.resize(img, (nw, nh), INTER_LINEAR)
//   padded[0:nh, 0:nw] = resized   (rest = 114)
float letterbox(const cv::Mat& src_bgr, int target, cv::Mat& dst) {
    int h = src_bgr.rows;
    int w = src_bgr.cols;
    float rh = static_cast<float>(target) / static_cast<float>(h);
    float rw = static_cast<float>(target) / static_cast<float>(w);
    float r  = std::min(rh, rw);
    int nh = static_cast<int>(std::round(h * r));
    int nw = static_cast<int>(std::round(w * r));

    cv::Mat resized;
    cv::resize(src_bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    dst.create(target, target, CV_8UC3);
    dst.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(dst(cv::Rect(0, 0, nw, nh)));
    return r;
}

// HWC uint8 -> CHW float32 (no normalization, BGR order preserved).
void hwc_uint8_to_chw_float(const cv::Mat& hwc, std::vector<float>& chw) {
    CV_Assert(hwc.type() == CV_8UC3);
    int H = hwc.rows;
    int W = hwc.cols;
    int C = 3;
    chw.resize(static_cast<std::size_t>(C) * H * W);
    // dst layout: [c0_all_pixels, c1_all_pixels, c2_all_pixels]
    const std::uint8_t* src = hwc.ptr<std::uint8_t>();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::uint8_t* px = src + (y * W + x) * C;
            chw[0 * H * W + y * W + x] = static_cast<float>(px[0]);
            chw[1 * H * W + y * W + x] = static_cast<float>(px[1]);
            chw[2 * H * W + y * W + x] = static_cast<float>(px[2]);
        }
    }
}

}  // namespace

Yolox::Yolox(TrtEngine& engine) : Yolox(engine, Options{}) {}

Yolox::Yolox(TrtEngine& engine, Options opts)
    : engine_{engine}, opts_{std::move(opts)} {
    // Sanity: input shape should be (1, 3, input_size, input_size).
    const auto& bin = engine_.binding(opts_.input_name);
    if (bin.dims.nbDims != 4
        || bin.dims.d[0] != 1
        || bin.dims.d[1] != 3
        || bin.dims.d[2] != opts_.input_size
        || bin.dims.d[3] != opts_.input_size) {
        FITRA_LOG_WARN("YOLOX input '{}' shape unexpected; engine declares dims with nbDims={}",
                       opts_.input_name, bin.dims.nbDims);
    }
}

std::vector<Bbox> Yolox::infer(const cv::Mat& frame_bgr) {
    float r = letterbox(frame_bgr, opts_.input_size, letterbox_);
    hwc_uint8_to_chw_float(letterbox_, input_blob_);

    const std::size_t bytes = input_blob_.size() * sizeof(float);
    engine_.copy_input_from_host(opts_.input_name, input_blob_.data(), bytes);
    engine_.enqueue();
    engine_.synchronize();

    auto dets_shape   = engine_.current_shape(opts_.dets_name);    // (1, N, 5)
    auto labels_shape = engine_.current_shape(opts_.labels_name);  // (1, N)
    if (dets_shape.nbDims != 3 || labels_shape.nbDims != 2) {
        throw std::runtime_error("YOLOX outputs have unexpected rank");
    }
    int N = dets_shape.d[1];
    if (labels_shape.d[1] != N) {
        throw std::runtime_error("YOLOX dets/labels N mismatch");
    }
    if (N <= 0) {
        return {};
    }

    dets_host_.resize(static_cast<std::size_t>(N) * 5);
    labels_host_.resize(static_cast<std::size_t>(N));
    engine_.copy_output_to_host(opts_.dets_name,
                                dets_host_.data(),
                                dets_host_.size() * sizeof(float));
    engine_.copy_output_to_host(opts_.labels_name,
                                labels_host_.data(),
                                labels_host_.size() * sizeof(std::int64_t));
    engine_.synchronize();

    std::vector<Bbox> kept;
    kept.reserve(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        float score = dets_host_[i * 5 + 4];
        std::int64_t lbl = labels_host_[i];
        if (lbl != opts_.person_label) continue;
        if (score < opts_.score_thr) continue;
        Bbox b;
        b.x1    = dets_host_[i * 5 + 0] / r;
        b.y1    = dets_host_[i * 5 + 1] / r;
        b.x2    = dets_host_[i * 5 + 2] / r;
        b.y2    = dets_host_[i * 5 + 3] / r;
        b.score = score;
        kept.push_back(b);
    }
    return kept;
}

}  // namespace fitra::infer
