#include "infer/rtmpose.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <opencv2/imgproc.hpp>

#include "util/logging.hpp"

namespace fitra::infer {

namespace {

// BGR ImageNet mean/std used by the rtmlib-exported RTMPose ONNX (the same
// constants live in python/scripts/pose_pipeline.py).
constexpr float kMeanB = 103.53f;
constexpr float kMeanG = 116.28f;
constexpr float kMeanR = 123.675f;
constexpr float kStdB  = 57.375f;
constexpr float kStdG  = 57.12f;
constexpr float kStdR  = 58.395f;

// 90-degree rotation of (a - b) about b, returning b + R90(a-b).
// Matches python _third_point.
cv::Point2f third_point(const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f d = a - b;
    return b + cv::Point2f(-d.y, d.x);
}

// _bbox_to_cs in python: pad the bbox by `padding`, then expand the shorter
// side so the bbox aspect matches input_w/input_h.
void bbox_to_cs(const Bbox& bb,
                float padding,
                float aspect_w_over_h,
                cv::Point2f& center,
                cv::Point2f& scale) {
    center.x = 0.5f * (bb.x1 + bb.x2);
    center.y = 0.5f * (bb.y1 + bb.y2);
    float w = (bb.x2 - bb.x1) * padding;
    float h = (bb.y2 - bb.y1) * padding;
    if (w > h * aspect_w_over_h) {
        h = w / aspect_w_over_h;
    } else {
        w = h * aspect_w_over_h;
    }
    scale.x = w;
    scale.y = h;
}

// 3-point affine matrix (2x3 float64), matching python _warp_matrix.
// Forward maps `center` to (out_w/2, out_h/2) and the rotated src_dir to dst_dir.
cv::Mat warp_matrix(const cv::Point2f& center,
                    const cv::Point2f& scale,
                    int out_w,
                    int out_h,
                    bool inv) {
    float src_w  = scale.x;
    cv::Point2f src_dir(0.0f, -0.5f * src_w);  // rotation angle = 0
    cv::Point2f dst_dir(0.0f, -0.5f * static_cast<float>(out_w));

    cv::Point2f src[3];
    cv::Point2f dst[3];
    src[0] = center;
    src[1] = center + src_dir;
    src[2] = third_point(src[0], src[1]);
    dst[0] = cv::Point2f(0.5f * out_w, 0.5f * out_h);
    dst[1] = dst[0] + dst_dir;
    dst[2] = third_point(dst[0], dst[1]);

    return inv ? cv::getAffineTransform(dst, src)
               : cv::getAffineTransform(src, dst);
}

void apply_affine_inplace(const cv::Mat& M_inv_2x3,
                          std::array<Keypoint, kNumKeypoints>& kpts) {
    CV_Assert(M_inv_2x3.rows == 2 && M_inv_2x3.cols == 3
              && M_inv_2x3.type() == CV_64F);
    const double* m = M_inv_2x3.ptr<double>();
    double a = m[0], b = m[1], c = m[2];
    double d = m[3], e = m[4], f = m[5];
    for (auto& kp : kpts) {
        double x = a * kp.x + b * kp.y + c;
        double y = d * kp.x + e * kp.y + f;
        kp.x = static_cast<float>(x);
        kp.y = static_cast<float>(y);
    }
}

}  // namespace

RtmPose::RtmPose(TrtEngine& engine) : RtmPose(engine, Options{}) {}

RtmPose::RtmPose(TrtEngine& engine, Options opts)
    : engine_{engine}, opts_{std::move(opts)} {}

// Free-standing preprocess: thread-safe, no shared state. Writes a normalized
// BGR CHW tensor for one (frame, bbox) into `dst_chw` (3 * input_h * input_w
// floats), and returns the inverse affine in `M_inv_out`.
void RtmPose::preprocess_to_blob(const Options& opts,
                                 const cv::Mat& frame_bgr,
                                 const Bbox& bb,
                                 float* dst_chw,
                                 cv::Mat& M_inv_out) {
    cv::Point2f center, scale;
    float aspect = static_cast<float>(opts.input_w)
                 / static_cast<float>(opts.input_h);
    bbox_to_cs(bb, opts.padding, aspect, center, scale);

    cv::Mat warp;
    cv::Mat M = warp_matrix(center, scale, opts.input_w, opts.input_h, false);
    cv::warpAffine(frame_bgr, warp, M,
                   cv::Size(opts.input_w, opts.input_h),
                   cv::INTER_LINEAR);

    // Fuse uint8 -> float32 + per-channel normalize + HWC -> CHW into one
    // pass over the warped image.
    const int H   = warp.rows;
    const int W   = warp.cols;
    const int npx = H * W;
    const std::uint8_t* src = warp.ptr<std::uint8_t>();
    float* ch_b = dst_chw + 0 * npx;
    float* ch_g = dst_chw + 1 * npx;
    float* ch_r = dst_chw + 2 * npx;
    const float inv_std_b = 1.0f / kStdB;
    const float inv_std_g = 1.0f / kStdG;
    const float inv_std_r = 1.0f / kStdR;
    for (int i = 0; i < npx; ++i) {
        ch_b[i] = (static_cast<float>(src[i * 3 + 0]) - kMeanB) * inv_std_b;
        ch_g[i] = (static_cast<float>(src[i * 3 + 1]) - kMeanG) * inv_std_g;
        ch_r[i] = (static_cast<float>(src[i * 3 + 2]) - kMeanR) * inv_std_r;
    }

    M_inv_out = warp_matrix(center, scale, opts.input_w, opts.input_h, true);
}

void RtmPose::prepare_batch_buffers(std::size_t n,
                                    int& simcc_x_width,
                                    int& simcc_y_width,
                                    std::size_t& per_item) {
    // Engine-declared output shapes: (-1, K, Wx) and (-1, K, Wy).
    auto eng_sx = engine_.engine().getTensorShape(opts_.simcc_x_name.c_str());
    auto eng_sy = engine_.engine().getTensorShape(opts_.simcc_y_name.c_str());
    if (eng_sx.nbDims != 3 || eng_sy.nbDims != 3
        || eng_sx.d[1] != static_cast<int>(kNumKeypoints)
        || eng_sy.d[1] != static_cast<int>(kNumKeypoints)) {
        throw std::runtime_error("RTMPose engine output rank/K unexpected");
    }
    simcc_x_width = eng_sx.d[2];
    simcc_y_width = eng_sy.d[2];

    int H = opts_.input_h;
    int W = opts_.input_w;
    per_item = static_cast<std::size_t>(3) * H * W;
    input_blob_.resize(per_item * n);
    simcc_x_host_.resize(static_cast<std::size_t>(n) * kNumKeypoints * simcc_x_width);
    simcc_y_host_.resize(static_cast<std::size_t>(n) * kNumKeypoints * simcc_y_width);
}

void RtmPose::enqueue_current_input(std::size_t n) {
    int H = opts_.input_h;
    int W = opts_.input_w;
    nvinfer1::Dims in_dims;
    in_dims.nbDims = 4;
    in_dims.d[0] = static_cast<int>(n);
    in_dims.d[1] = 3;
    in_dims.d[2] = H;
    in_dims.d[3] = W;
    engine_.set_input_shape(opts_.input_name, in_dims);

    engine_.copy_input_from_host(opts_.input_name,
                                 input_blob_.data(),
                                 input_blob_.size() * sizeof(float));
    engine_.enqueue();
    engine_.copy_output_to_host(opts_.simcc_x_name,
                                simcc_x_host_.data(),
                                simcc_x_host_.size() * sizeof(float));
    engine_.copy_output_to_host(opts_.simcc_y_name,
                                simcc_y_host_.data(),
                                simcc_y_host_.size() * sizeof(float));
    engine_.synchronize();
}

void RtmPose::decode_current_outputs(std::size_t n,
                                     int simcc_x_width,
                                     int simcc_y_width,
                                     const std::vector<Bbox>& bboxes,
                                     const std::vector<cv::Mat>& M_invs,
                                     std::vector<Person>& out) {
    const std::size_t stride_x =
        static_cast<std::size_t>(kNumKeypoints) * simcc_x_width;
    const std::size_t stride_y =
        static_cast<std::size_t>(kNumKeypoints) * simcc_y_width;
    for (std::size_t i = 0; i < n; ++i) {
        Person p{};
        p.bbox = bboxes[i];
        const float* base_x = simcc_x_host_.data() + i * stride_x;
        const float* base_y = simcc_y_host_.data() + i * stride_y;
        for (std::size_t k = 0; k < kNumKeypoints; ++k) {
            const float* row_x = base_x + k * simcc_x_width;
            const float* row_y = base_y + k * simcc_y_width;
            int   x_arg = 0;
            float x_max = row_x[0];
            for (int j = 1; j < simcc_x_width; ++j) {
                if (row_x[j] > x_max) { x_max = row_x[j]; x_arg = j; }
            }
            int   y_arg = 0;
            float y_max = row_y[0];
            for (int j = 1; j < simcc_y_width; ++j) {
                if (row_y[j] > y_max) { y_max = row_y[j]; y_arg = j; }
            }
            float score = std::min(x_max, y_max);
            if (score < 0.0f) score = 0.0f;
            p.kpts[k].x = static_cast<float>(x_arg) / opts_.simcc_split;
            p.kpts[k].y = static_cast<float>(y_arg) / opts_.simcc_split;
            p.kpts[k].score = score;
        }
        apply_affine_inplace(M_invs[i], p.kpts);
        out.push_back(p);
    }
}

void RtmPose::run_one_batch(const Request* reqs,
                            std::size_t n,
                            std::vector<Person>& out) {
    if (n == 0) return;

    int Wx = 0;
    int Wy = 0;
    std::size_t per_item = 0;
    prepare_batch_buffers(n, Wx, Wy, per_item);

    // Preprocess all items into the batched blob. Runs in parallel for
    // n >= 2 — preprocess_to_blob is thread-safe (no shared scratch).
    std::vector<cv::Mat> M_invs(n);
    std::vector<Bbox> bboxes(n);
    for (std::size_t i = 0; i < n; ++i) {
        bboxes[i] = reqs[i].bbox;
    }
    if (n >= 2) {
        std::vector<std::thread> threads;
        threads.reserve(n - 1);
        for (std::size_t i = 1; i < n; ++i) {
            threads.emplace_back([&, i]() {
                preprocess_to_blob(opts_, *reqs[i].frame, reqs[i].bbox,
                                   input_blob_.data() + i * per_item,
                                   M_invs[i]);
            });
        }
        // Use the caller thread for item 0 to avoid an extra thread.
        preprocess_to_blob(opts_, *reqs[0].frame, reqs[0].bbox,
                           input_blob_.data(),
                           M_invs[0]);
        for (auto& t : threads) t.join();
    } else {
        preprocess_to_blob(opts_, *reqs[0].frame, reqs[0].bbox,
                           input_blob_.data(),
                           M_invs[0]);
    }

    enqueue_current_input(n);
    decode_current_outputs(n, Wx, Wy, bboxes, M_invs, out);
}

void RtmPose::run_one_prebaked(const PrebakedRequest* reqs,
                               std::size_t n,
                               std::vector<Person>& out) {
    if (n == 0) return;

    int Wx = 0;
    int Wy = 0;
    std::size_t per_item = 0;
    prepare_batch_buffers(n, Wx, Wy, per_item);

    // Pack the prebaked per-item CHW blobs into the contiguous batch buffer.
    // (Cheap memcpy; preprocess itself already ran on the per-camera threads.)
    std::vector<cv::Mat> M_invs(n);
    std::vector<Bbox> bboxes(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!reqs[i].chw) {
            throw std::runtime_error("RTMPose prebaked request has null CHW buffer");
        }
        std::memcpy(input_blob_.data() + i * per_item,
                    reqs[i].chw,
                    per_item * sizeof(float));
        M_invs[i] = reqs[i].M_inv;
        bboxes[i] = reqs[i].bbox;
    }

    enqueue_current_input(n);
    decode_current_outputs(n, Wx, Wy, bboxes, M_invs, out);
}

std::vector<Person> RtmPose::infer_prebaked(const std::vector<PrebakedRequest>& reqs) {
    std::vector<Person> out;
    out.reserve(reqs.size());
    if (reqs.empty()) return out;
    const std::size_t max_b = static_cast<std::size_t>(std::max(1, opts_.max_batch));
    for (std::size_t i = 0; i < reqs.size(); i += max_b) {
        std::size_t n = std::min(max_b, reqs.size() - i);
        run_one_prebaked(reqs.data() + i, n, out);
    }
    return out;
}

std::vector<Person> RtmPose::infer_batch(const std::vector<Request>& reqs) {
    std::vector<Person> out;
    out.reserve(reqs.size());
    if (reqs.empty()) return out;

    const std::size_t max_b = static_cast<std::size_t>(std::max(1, opts_.max_batch));
    for (std::size_t i = 0; i < reqs.size(); i += max_b) {
        std::size_t n = std::min(max_b, reqs.size() - i);
        run_one_batch(reqs.data() + i, n, out);
    }
    return out;
}

std::vector<Person> RtmPose::infer(const cv::Mat& frame_bgr,
                                   const std::vector<Bbox>& bboxes) {
    if (bboxes.empty()) return {};
    std::vector<Request> reqs;
    reqs.reserve(bboxes.size());
    for (const auto& b : bboxes) {
        reqs.push_back(Request{&frame_bgr, b});
    }
    return infer_batch(reqs);
}

}  // namespace fitra::infer
