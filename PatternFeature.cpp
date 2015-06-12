#include "PatternFeature.hpp"

PatternFeature::PatternFeature(const cv::Mat &src)
{
  for (int m = 0; m < scales; m++)
    for (int n = 0; n < orientations; n++) {
      const auto gabor = cv::getGaborKernel(
                             cv::Size(ksize, ksize),
                             pow(alpha, m) * sigma_u,
                             n * CV_PI / orientations,
                             lambda,
                             sigma_u / sigma_v,
                             0) *
                         pow(alpha, -m) *
                         exp(2 * CV_PI * CV_PI * pow(alpha, 2 * m) * sigma_u * sigma_u / (lambda * lambda));
      cv::Mat wavelet;
      cv::filter2D(src, wavelet, CV_64F, gabor);

      cv::Scalar mean, deviation;
      cv::meanStdDev(cv::abs(wavelet), mean, deviation);

      means.at(orientations * m + n) = mean.val[0];
      deviations.at(orientations * m + n) = deviation.val[0];
    }
}

std::vector<double> PatternFeature::featureVector(void) const
{
  std::vector<double> vec;

  for (int i = 0; i < scales * orientations; i++) {
    vec.push_back(means.at(i));
    vec.push_back(deviations.at(i));
  }

  return vec;
}

std::ostream &operator<<(std::ostream &os, const PatternFeature &patFeat)
{
  for (auto &x : patFeat.featureVector())
    os << x << std::endl;

  return os;
}
