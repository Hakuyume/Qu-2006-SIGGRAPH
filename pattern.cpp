#include <opencv2/imgproc.hpp>

#include "pattern.hpp"

pattern::Feature pattern::getFeature(const cv::Mat &src, const cv::Point &pos)
{
  pattern::Feature feature;

  const auto roi = src(cv::Rect(
      pos.x - window / 2,
      pos.y - window / 2,
      window, window));

  for (int m = 0; m < scales; m++)
    for (int n = 0; n < orientations; n++) {
      const auto gabor = cv::getGaborKernel(
                             cv::Size(window, window),
                             pow(alpha, m) * sigma_u,
                             n * CV_PI / orientations,
                             2 * CV_PI * pow(alpha, m) * sigma_u * sigma_u / window,
                             sigma_u / sigma_v,
                             0) *
                         pow(alpha, -m);
      cv::Mat wavelet;
      cv::filter2D(roi, wavelet, CV_64F, gabor);

      cv::Scalar mean, deviation;
      cv::meanStdDev(cv::abs(wavelet), mean, deviation);

      feature((orientations * m + n) * 2 + 0) = mean.val[0];
      feature((orientations * m + n) * 2 + 1) = deviation.val[0];
    }

  return feature;
}
