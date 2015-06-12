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

  std::vector<cv::Point> points{
      cv::Point(150, 120),
      cv::Point(175, 105),
      cv::Point(200, 90),

      cv::Point(110, 240),
      cv::Point(130, 230),
      cv::Point(150, 220),

      cv::Point(150, 300),
      cv::Point(165, 275),
      cv::Point(180, 250),

      cv::Point(45, 175),
      cv::Point(50, 200),
      cv::Point(55, 225)};

  auto result = input.clone();
  std::vector<PatternFeature> features;

  for (auto &p : points) {
    cv::circle(result, p, 5, cv::Scalar(0), -1);

    features.push_back(PatternFeature(input, p));
  }

  cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
  cv::imshow("result", result);

  for (int i = 0; i < features.size(); i++)
    for (int j = i + 1; j < features.size(); j++)
      std::cout << i << "-" << j << "\t" << features.at(i).distance(features.at(j)) << std::endl;

  cv::waitKey(0);

  return 0;
}
