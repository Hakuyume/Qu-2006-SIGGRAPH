#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

  auto gabor = cv::getGaborKernel(cv::Size(25, 25), 2, CV_PI / 2, 5, 1);
  cv::Mat result;
  cv::filter2D(input & select, result, -1, gabor);

  cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
  cv::imshow("result", result);

  cv::waitKey(0);

  return 0;
}
