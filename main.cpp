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
  const auto userFeature = sum / points.size();

  cv::Mat result = cv::Mat::zeros(input.size(), CV_8U);

  expand(input, result, userFeature, points.at(0));

  cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
  cv::imshow("result", input & result);

  cv::waitKey(0);

  return 0;
}
