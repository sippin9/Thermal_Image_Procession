// Tracking.h
#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>


class Tracking {
public:
    static cv::Mat PreProcess(const cv::Mat& im);
    static cv::Mat PreProcess1(const cv::Mat& im);
    static cv::Mat PreSVD(const cv::Mat& imRectRight);
    static cv::Mat createDiagonalMatrix(const cv::Scalar& value, int size1, int size2);
    static cv::Mat PreGamma(const cv::Mat& imRectLeft, float gamma);
    static cv::Mat guidedFilter(const cv::Mat& srcMat, int radius, double eps);
    static cv::Mat AdaptiveFilter(const cv::Mat& vv);
    static void FAST_t(const cv::Mat& _img, std::vector<cv::KeyPoint>& keypoints, bool nonmax_suppression);
    static void mmakeOffsets(int pixel[25], int rowStride, int patternSize);
};

#endif // TRACKING_H
