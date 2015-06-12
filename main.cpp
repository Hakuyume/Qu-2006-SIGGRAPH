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

  int hsize = 4;
  std::vector<cv::Point> centers{
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

  auto rois = input.clone();
  std::vector<PatternFeature> features;

  for (auto &center : centers) {
    cv::Rect roi{
        center.x - hsize,
        center.y - hsize,
        hsize * 2 + 1,
        hsize * 2 + 1};

    cv::rectangle(rois, roi, cv::Scalar(255));

    features.push_back(PatternFeature(input(roi)));
  }

  cv::namedWindow("rois", cv::WINDOW_AUTOSIZE);
  cv::imshow("rois", rois);

  for (int i = 0; i < features.size(); i++)
    for (int j = i + 1; j < features.size(); j++)
      std::cout << i << "-" << j << "\t" << features.at(i).distance(features.at(j)) << std::endl;

  cv::waitKey(0);

  return 0;
}
