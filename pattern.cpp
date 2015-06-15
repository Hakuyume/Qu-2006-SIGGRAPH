#include "pattern.hpp"
#include <opencv2/imgproc.hpp>

pattern::Feature pattern::getFeature(const cv::Mat &src, const cv::Point &pos)
{
  pattern::Feature feature;

  const auto roi = src(cv::Rect(
      pos.x - window / 2,
      pos.y - window / 2,
      window, window));

  for (size_t m = 0; m < scales; m++) {
    const double alpha_m{pow(alpha, m)};

    for (size_t n = 0; n < orientations; n++) {
      const auto gabor = cv::getGaborKernel(
                             cv::Size(window, window),
                             alpha_m * sigma_u,
                             n * CV_PI / orientations,
                             2 * CV_PI * alpha_m * sigma_u * sigma_u / window,
                             sigma_u / sigma_v,
                             0) /
                         alpha_m;
      cv::Mat wavelet;
      cv::filter2D(roi, wavelet, CV_64F, gabor);

      cv::Scalar mean, deviation;
      cv::meanStdDev(cv::abs(wavelet), mean, deviation);

      feature((orientations * m + n) * 2 + 0) = mean.val[0];
      feature((orientations * m + n) * 2 + 1) = deviation.val[0];
    }
  }

  return feature;
}
