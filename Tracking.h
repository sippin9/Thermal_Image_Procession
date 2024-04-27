// Tracking.h
#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

class Tracking {
public:
    static cv::Mat PreProcess(const cv::Mat& im);
    static cv::Mat PreProcess1(const cv::Mat& im);
    static cv::Mat PreSVD(const cv::Mat& imRectRight);
    static cv::Mat createDiagonalMatrix(const cv::Scalar& value, int size1, int size2);
    static cv::Mat PreGamma(const cv::Mat& imRectLeft, float gamma);
    static cv::Mat guidedFilter(const cv::Mat& srcMat, int radius, double eps);
    static cv::Mat AdaptiveFilter(const cv::Mat& vv);
};

#endif // TRACKING_H
