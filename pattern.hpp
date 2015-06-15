#pragma once

#include <opencv2/core.hpp>

namespace pattern
{
static constexpr size_t scales{4};
static constexpr size_t orientations{6};

static constexpr size_t window{16};
static constexpr double alpha{1.1};
static constexpr double sigma_u{0.25};
static constexpr double sigma_v{0.25};

static constexpr size_t featureChannels{scales * orientations * 2};
using featureVec = cv::Vec<double, featureChannels>;

void getFeature(const cv::Mat &src, cv::Mat &dst);
void getDiff(const cv::Mat &src, cv::Mat &dst, const featureVec &feature);
};
