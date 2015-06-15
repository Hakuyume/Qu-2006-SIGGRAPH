#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "pattern.hpp"

void expand(const cv::Mat &src, cv::Mat &dst, const pattern::Feature &feature, const cv::Point &pos)
{
  if (dst.at<unsigned char>(pos))
    return;

  if ((pattern::getFeature(src, pos) - feature).squaredNorm() > 10000)
    return;

  dst.at<unsigned char>(pos) = 0xff;

  expand(src, dst, feature, pos + cv::Point(-1, 0));
  expand(src, dst, feature, pos + cv::Point(+1, 0));
  expand(src, dst, feature, pos + cv::Point(0, -1));
  expand(src, dst, feature, pos + cv::Point(0, +1));
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

  std::vector<cv::Point> points{
      cv::Point(150, 120),
      cv::Point(175, 105),
      cv::Point(200, 90)};

  pattern::Feature sum = pattern::Feature::Zero();
  for (auto &p : points)
    sum += pattern::getFeature(input, p);
  const auto feature = sum / points.size();

  cv::Mat distance = cv::Mat::zeros(input.size(), CV_8U);

#pragma omp parallel for
  for (size_t i = pattern::window / 2; i < input.rows - pattern::window / 2; i++)
    for (size_t j = pattern::window / 2; j < input.cols - pattern::window / 2; j++)
      distance.at<unsigned char>(i, j) = 0xff / (1 + (pattern::getFeature(input, cv::Point(j, i)) - feature).squaredNorm() / 100000);

  cv::namedWindow("distance", cv::WINDOW_AUTOSIZE);
  cv::imshow("distance", distance);

  std::vector<cv::Mat> channels{input, input, distance};
  cv::Mat result;
  cv::merge(channels, result);
  cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
  cv::imshow("result", result);

  cv::waitKey(0);

  return 0;
}
