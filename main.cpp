#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "PatternFeature.hpp"

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

  PatternFeature user{input & select};
  std::cout << user << std::endl;

  cv::waitKey(0);

  return 0;
}
