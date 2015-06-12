#include "PatternFeature.hpp"

PatternFeature::PatternFeature(const cv::Mat &src, const cv::Point &pos)
{
  const auto roi = src(cv::Rect(
      pos.x - ksize / 2,
      pos.y - ksize / 2,
      ksize, ksize));

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
      cv::filter2D(roi, wavelet, CV_64F, gabor);

      cv::Scalar mean, deviation;
      cv::meanStdDev(cv::abs(wavelet), mean, deviation);

      means.at(orientations * m + n) = mean.val[0];
      deviations.at(orientations * m + n) = deviation.val[0];
    }
}

double PatternFeature::distance(const PatternFeature &patFeat) const
{
  double sum = 0;

  for (int i = 0; i < scales * orientations; i++) {
    sum += (patFeat.means.at(i) - means.at(i)) * (patFeat.means.at(i) - means.at(i));
    sum += (patFeat.deviations.at(i) - deviations.at(i)) * (patFeat.deviations.at(i) - deviations.at(i));
  }

  return sum;
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
