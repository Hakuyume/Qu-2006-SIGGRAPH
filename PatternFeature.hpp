#pragma once

#include <iostream>
#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class PatternFeature
{
private:
  static constexpr int scales = 4;
  static constexpr int orientations = 6;

  static constexpr int ksize = 9;
  static constexpr double alpha = 1.1;
  static constexpr double lambda = 2;
  static constexpr double sigma_u = 0.25;
  static constexpr double sigma_v = 0.25;

  std::array<double, scales * orientations> means, deviations;

public:
  PatternFeature(const cv::Mat &src);
  double distance(const PatternFeature &patFeat) const;
  std::vector<double> featureVector(void) const;
};

std::ostream &operator<<(std::ostream &os, const PatternFeature &patFeat);
