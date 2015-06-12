#include <iostream>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

constexpr int scales = 4;
constexpr int orientations = 6;

using Feature = std::array<double, scales * orientations * 2>;

Feature patternFeature(const cv::Mat &input)
{
  Feature feature;

  for (int m = 0; m < scales; m++)
    for (int n = 0; n < orientations; n++) {
      const auto gabor = cv::getGaborKernel(cv::Size(25, 25), 2, CV_PI / 2, 5, 1);
      cv::Mat wavelet;
      cv::filter2D(input, wavelet, -1, gabor);

      cv::Scalar mean, deviation;
      cv::meanStdDev(wavelet, mean, deviation);

      feature.at((orientations * m + n) * 2 + 0) = mean.val[0];
      feature.at((orientations * m + n) * 2 + 1) = deviation.val[0];
    }

  return feature;
}

int main(int argc, char *argv[])
{
  if (argc < 2)
    return 1;
  auto input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  if (input.empty())
    return 1;
  cv::namedWindow("input", cv::WINDOW_AUTOSIZE);
  cv::imshow("input", input);

  cv::Mat select = cv::Mat::zeros(input.size(), CV_8U);
  cv::circle(select, cv::Point(150, 120), 20, cv::Scalar::all(0xff), -1);
  cv::namedWindow("select", cv::WINDOW_AUTOSIZE);
  cv::imshow("select", input & select);

  for (auto &t : patternFeature(input & select))
    std::cout << t << std::endl;

  cv::waitKey(0);

  return 0;
}
