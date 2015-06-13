#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace pattern
{
static constexpr size_t scales = 4;
static constexpr size_t orientations = 6;

static constexpr size_t window = 16;
static constexpr double alpha = 1.1;
static constexpr double lambda = 2;
static constexpr double sigma_u = 0.25;
static constexpr double sigma_v = 0.25;

static constexpr size_t featureSize = scales * orientations * 2;
using Feature = Eigen::Matrix<double, featureSize, 1>;

Feature getFeature(const cv::Mat &src, const cv::Point &pos);
};
