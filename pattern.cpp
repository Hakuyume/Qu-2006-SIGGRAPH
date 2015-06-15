#include "pattern.hpp"
#include <vector>
#include <opencv2/imgproc.hpp>

void pattern::getFeature(const cv::Mat &src, cv::Mat &dst)
{
  std::vector<cv::Mat> channels;

  const auto averageKernel = cv::Mat::ones(cv::Size(window, window), CV_64F) / (window * window);

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
      cv::filter2D(src, wavelet, CV_64F, gabor);
      wavelet = cv::abs(wavelet);

      cv::Mat mean;
      cv::filter2D(wavelet, mean, CV_64F, averageKernel);
      cv::Mat squaredMean;
      cv::pow(mean, 2, squaredMean);

      cv::Mat squaredWavelet;
      cv::pow(wavelet, 2, squaredWavelet);
      cv::Mat meanSquared;
      cv::filter2D(squaredWavelet, meanSquared, CV_64F, averageKernel);

      cv::Mat deviation;
      cv::sqrt(meanSquared - squaredMean, deviation);

      channels.push_back(mean);
      channels.push_back(deviation);
    }
  }
  cv::merge(channels, dst);
}

void pattern::getDiff(const cv::Mat &src, cv::Mat &dst, const pattern::featureVec &feature)
{
  dst = cv::Mat::zeros(src.size(), CV_64F);

  std::vector<cv::Mat> channels;
  cv::split(src, channels);

  for (size_t i = 0; i < featureChannels; i++) {
    cv::Mat diff;
    cv::pow(channels.at(i) - feature(i), 2, diff);
    dst += diff;
  }
}
