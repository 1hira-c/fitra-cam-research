#include "infer/rtmpose.hpp"

#include <algorithm>
#include <cmath>
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

// HWC float32 BGR -> CHW float32 BGR (preserve channel order).
void hwc_float_to_chw(const cv::Mat& hwc_bgr_f, std::vector<float>& chw) {
    CV_Assert(hwc_bgr_f.type() == CV_32FC3);
    int H = hwc_bgr_f.rows;
    int W = hwc_bgr_f.cols;
    int C = 3;
    chw.resize(static_cast<std::size_t>(C) * H * W);
    for (int y = 0; y < H; ++y) {
        const float* row = hwc_bgr_f.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            float b = row[x * 3 + 0];
            float g = row[x * 3 + 1];
            float r = row[x * 3 + 2];
            chw[0 * H * W + y * W + x] = b;
            chw[1 * H * W + y * W + x] = g;
            chw[2 * H * W + y * W + x] = r;
        }
    }
}

void apply_affine_inplace(const cv::Mat& M_inv_2x3,
                          std::array<Keypoint, kNumKeypoints>& kpts) {
    // M_inv from cv::getAffineTransform is CV_64F.
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

void RtmPose::preprocess_one(const cv::Mat& frame_bgr,
                             const Bbox& bb,
                             cv::Mat& M_inv_out) {
    cv::Point2f center, scale;
    float aspect = static_cast<float>(opts_.input_w)
                 / static_cast<float>(opts_.input_h);
    bbox_to_cs(bb, opts_.padding, aspect, center, scale);

    cv::Mat M = warp_matrix(center, scale, opts_.input_w, opts_.input_h, false);
    cv::warpAffine(frame_bgr, warp_, M,
                   cv::Size(opts_.input_w, opts_.input_h),
                   cv::INTER_LINEAR);

    // BGR ImageNet normalization with per-channel mean/std, done inline so
    // we don't depend on OpenCV's per-channel broadcast quirks.
    warp_.convertTo(warp_f_, CV_32FC3);
    int H = warp_f_.rows;
    int W = warp_f_.cols;
    int npx = H * W;
    float* p = warp_f_.ptr<float>();
    for (int i = 0; i < npx; ++i) {
        p[i * 3 + 0] = (p[i * 3 + 0] - kMeanB) / kStdB;
        p[i * 3 + 1] = (p[i * 3 + 1] - kMeanG) / kStdG;
        p[i * 3 + 2] = (p[i * 3 + 2] - kMeanR) / kStdR;
    }

    hwc_float_to_chw(warp_f_, input_blob_);
    M_inv_out = warp_matrix(center, scale, opts_.input_w, opts_.input_h, true);
}

std::vector<Person> RtmPose::infer(const cv::Mat& frame_bgr,
                                   const std::vector<Bbox>& bboxes) {
    std::vector<Person> out;
    out.reserve(bboxes.size());
    if (bboxes.empty()) return out;

    // Phase 1: batch=1 per bbox. Phase 3 batches.
    nvinfer1::Dims in_dims;
    in_dims.nbDims = 4;
    in_dims.d[0] = 1;
    in_dims.d[1] = 3;
    in_dims.d[2] = opts_.input_h;
    in_dims.d[3] = opts_.input_w;
    engine_.set_input_shape(opts_.input_name, in_dims);

    // Engine-declared output shapes: (-1, 17, Wx) and (-1, 17, Wy). The K
    // and W dims are static, batch is dynamic and now resolves to 1.
    auto eng_sx = engine_.engine().getTensorShape(opts_.simcc_x_name.c_str());
    auto eng_sy = engine_.engine().getTensorShape(opts_.simcc_y_name.c_str());
    if (eng_sx.nbDims != 3 || eng_sy.nbDims != 3
        || eng_sx.d[1] != static_cast<int>(kNumKeypoints)
        || eng_sy.d[1] != static_cast<int>(kNumKeypoints)) {
        throw std::runtime_error("RTMPose engine output rank/K unexpected");
    }
    int Wx = eng_sx.d[2];
    int Wy = eng_sy.d[2];
    simcc_x_host_.resize(static_cast<std::size_t>(kNumKeypoints) * Wx);
    simcc_y_host_.resize(static_cast<std::size_t>(kNumKeypoints) * Wy);

    for (const auto& bb : bboxes) {
        cv::Mat M_inv;
        preprocess_one(frame_bgr, bb, M_inv);

        const std::size_t in_bytes = input_blob_.size() * sizeof(float);
        engine_.copy_input_from_host(opts_.input_name, input_blob_.data(), in_bytes);
        engine_.enqueue();
        engine_.copy_output_to_host(opts_.simcc_x_name,
                                    simcc_x_host_.data(),
                                    simcc_x_host_.size() * sizeof(float));
        engine_.copy_output_to_host(opts_.simcc_y_name,
                                    simcc_y_host_.data(),
                                    simcc_y_host_.size() * sizeof(float));
        engine_.synchronize();

        Person p{};
        p.bbox = bb;
        for (std::size_t k = 0; k < kNumKeypoints; ++k) {
            const float* row_x = simcc_x_host_.data() + k * Wx;
            const float* row_y = simcc_y_host_.data() + k * Wy;

            // argmax + max for x and y independently
            int   x_arg = 0;
            float x_max = row_x[0];
            for (int i = 1; i < Wx; ++i) {
                if (row_x[i] > x_max) { x_max = row_x[i]; x_arg = i; }
            }
            int   y_arg = 0;
            float y_max = row_y[0];
            for (int i = 1; i < Wy; ++i) {
                if (row_y[i] > y_max) { y_max = row_y[i]; y_arg = i; }
            }

            float score = std::min(x_max, y_max);
            if (score < 0.0f) score = 0.0f;

            // (x, y) in 256x192 input frame coords (divide by simcc_split=2)
            p.kpts[k].x = static_cast<float>(x_arg) / opts_.simcc_split;
            p.kpts[k].y = static_cast<float>(y_arg) / opts_.simcc_split;
            p.kpts[k].score = score;
        }

        apply_affine_inplace(M_inv, p.kpts);
        out.push_back(p);
    }

    return out;
}

}  // namespace fitra::infer
