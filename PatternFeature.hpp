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

  std::array<double, scales * orientations> means, deviations;

public:
  PatternFeature(const cv::Mat &src);
  std::vector<double> featureVector(void) const;
};

std::ostream &operator<<(std::ostream &os, const PatternFeature &patFeat);
