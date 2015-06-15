#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "pattern.hpp"

void gradient(const cv::Mat &src, cv::Mat &dst)
{
  cv::Mat grad_x, grad_y;

  cv::Sobel(src, grad_x, CV_64F, 1, 0);
  cv::Sobel(src, grad_y, CV_64F, 0, 1);

  cv::pow(grad_x, 2, grad_x);
  cv::pow(grad_y, 2, grad_y);

  cv::sqrt(grad_x + grad_y, dst);
}

void propagate(const cv::Mat &src, cv::Mat &phi, const cv::Mat &diff)
{
  cv::Mat grad;
  gradient(src, grad);

  phi -= grad.mul(1 / (1 + diff));
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

  const cv::Point center{175, 105};
  const auto feature = pattern::getFeature(input, center);

  cv::Mat diff{input.size(), CV_64F};
#pragma omp parallel for
  for (size_t i = pattern::window / 2; i < input.rows - pattern::window / 2; i++)
    for (size_t j = pattern::window / 2; j < input.cols - pattern::window / 2; j++)
      diff.at<double>(i, j) = (pattern::getFeature(input, cv::Point(j, i)) - feature).squaredNorm();

  cv::Mat phi{input.size(), CV_64F};
#pragma omp parallel for
  for (size_t i = 0; i < input.rows; i++)
    for (size_t j = 0; j < input.cols; j++)
      phi.at<double>(i, j) = sqrt((j - center.x) * (j - center.x) + (i - center.y) * (i - center.y));

  cv::namedWindow("phi", cv::WINDOW_AUTOSIZE);
  cv::imshow("phi", phi);

  while (true) {
    propagate(input, phi, diff);
    cv::imshow("phi", phi);
    if (cv::waitKey(0) == 'q')
      break;
  }

  return 0;
}
