#include "infer/rtmpose.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

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

// Writes a normalized BGR CHW tensor for one (frame, bbox) into the
// destination region of input_blob_, and returns the corresponding inverse
// affine so caller can remap the decoded keypoints back to frame coords.
void RtmPose::preprocess_one(const cv::Mat& frame_bgr,
                             const Bbox& bb,
                             float* dst_chw,
                             cv::Mat& M_inv_out) {
    cv::Point2f center, scale;
    float aspect = static_cast<float>(opts_.input_w)
                 / static_cast<float>(opts_.input_h);
    bbox_to_cs(bb, opts_.padding, aspect, center, scale);

    cv::Mat M = warp_matrix(center, scale, opts_.input_w, opts_.input_h, false);
    cv::warpAffine(frame_bgr, warp_, M,
                   cv::Size(opts_.input_w, opts_.input_h),
                   cv::INTER_LINEAR);
    warp_.convertTo(warp_f_, CV_32FC3);

    // Per-pixel normalize and split into CHW BGR (channel-major).
    int H = warp_f_.rows;
    int W = warp_f_.cols;
    int npx = H * W;
    const float* src = warp_f_.ptr<float>();
    float* ch_b = dst_chw + 0 * npx;
    float* ch_g = dst_chw + 1 * npx;
    float* ch_r = dst_chw + 2 * npx;
    for (int i = 0; i < npx; ++i) {
        float b = src[i * 3 + 0];
        float g = src[i * 3 + 1];
        float r = src[i * 3 + 2];
        ch_b[i] = (b - kMeanB) / kStdB;
        ch_g[i] = (g - kMeanG) / kStdG;
        ch_r[i] = (r - kMeanR) / kStdR;
    }

    M_inv_out = warp_matrix(center, scale, opts_.input_w, opts_.input_h, true);
}

void RtmPose::run_one_batch(const Request* reqs,
                            std::size_t n,
                            std::vector<Person>& out) {
    if (n == 0) return;

    // Engine-declared output shapes: (-1, K, Wx) and (-1, K, Wy).
    auto eng_sx = engine_.engine().getTensorShape(opts_.simcc_x_name.c_str());
    auto eng_sy = engine_.engine().getTensorShape(opts_.simcc_y_name.c_str());
    if (eng_sx.nbDims != 3 || eng_sy.nbDims != 3
        || eng_sx.d[1] != static_cast<int>(kNumKeypoints)
        || eng_sy.d[1] != static_cast<int>(kNumKeypoints)) {
        throw std::runtime_error("RTMPose engine output rank/K unexpected");
    }
    int Wx = eng_sx.d[2];
    int Wy = eng_sy.d[2];

    int H = opts_.input_h;
    int W = opts_.input_w;
    std::size_t per_item = static_cast<std::size_t>(3) * H * W;
    input_blob_.resize(per_item * n);
    simcc_x_host_.resize(static_cast<std::size_t>(n) * kNumKeypoints * Wx);
    simcc_y_host_.resize(static_cast<std::size_t>(n) * kNumKeypoints * Wy);

    // Preprocess all items into the batched blob.
    std::vector<cv::Mat> M_invs(n);
    for (std::size_t i = 0; i < n; ++i) {
        preprocess_one(*reqs[i].frame, reqs[i].bbox,
                       input_blob_.data() + i * per_item,
                       M_invs[i]);
    }

    // Set dynamic batch dim and run.
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

    // Decode each batch item.
    const std::size_t stride_x = static_cast<std::size_t>(kNumKeypoints) * Wx;
    const std::size_t stride_y = static_cast<std::size_t>(kNumKeypoints) * Wy;
    for (std::size_t i = 0; i < n; ++i) {
        Person p{};
        p.bbox = reqs[i].bbox;
        const float* base_x = simcc_x_host_.data() + i * stride_x;
        const float* base_y = simcc_y_host_.data() + i * stride_y;
        for (std::size_t k = 0; k < kNumKeypoints; ++k) {
            const float* row_x = base_x + k * Wx;
            const float* row_y = base_y + k * Wy;
            int   x_arg = 0;
            float x_max = row_x[0];
            for (int j = 1; j < Wx; ++j) {
                if (row_x[j] > x_max) { x_max = row_x[j]; x_arg = j; }
            }
            int   y_arg = 0;
            float y_max = row_y[0];
            for (int j = 1; j < Wy; ++j) {
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
