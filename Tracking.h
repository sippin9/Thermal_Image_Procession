// Tracking.h
#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

class Tracking {
public:
    cv::Mat createDiagonalMatrix(const cv::Scalar& value, int size1, int size2);
    cv::Mat PreSVD(const cv::Mat& imRectRight);
    cv::Mat AdaptiveFilter(const cv::Mat& vv);
    cv::Mat guidedFilter(const cv::Mat& srcMat, int radius, double eps);
};

#endif // TRACKING_H
