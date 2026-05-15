#include "camera/jpeg_decoder.hpp"

#include <opencv2/imgcodecs.hpp>

namespace fitra::camera {

bool JpegDecoder::decode(const std::uint8_t* jpeg,
                         std::size_t bytes,
                         cv::Mat& out_bgr) {
    if (!jpeg || bytes == 0) return false;
    cv::Mat input(1, static_cast<int>(bytes), CV_8UC1,
                  const_cast<std::uint8_t*>(jpeg));
    // cv::imdecode writes into out_bgr; passing a non-empty existing Mat is
    // accepted but the contents may be reallocated.
    out_bgr = cv::imdecode(input, cv::IMREAD_COLOR);
    return !out_bgr.empty();
}

}  // namespace fitra::camera
