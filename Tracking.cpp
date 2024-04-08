// Tracking.cpp
#include "Tracking.h"

using namespace std;

cv::Mat Tracking::createDiagonalMatrix(const cv::Scalar& value, int size1, int size2) {
    cv::Mat diagonalMatrix = cv::Mat::zeros(size1, size1, CV_32F);
    for (int i = 0; i < size1; ++i)
        diagonalMatrix.at<float>(i, i) += value[0];
    return diagonalMatrix;
}

cv::Mat Tracking::PreSVD(const cv::Mat& imRectRight) {
    // Convert the image to grayscale if needed
    if (imRectRight.channels() == 3)
        cvtColor(imRectRight, imRectRight, CV_RGB2GRAY);
    else if (imRectRight.channels() == 4)
        cvtColor(imRectRight, imRectRight, CV_RGBA2GRAY);
    
    cv::Mat grayImage;
    imRectRight.convertTo(grayImage, CV_32FC1);

    // Perform SVD on the grayscale image
    cv::Mat svdU, svdS, svdVt;
    
    cv::SVD::compute(grayImage, svdS, svdU, svdVt);
    
    //Width of U: 512
    //Height of U: 512
    //Width of S: 1
    //Height of S: 512
    //Width of Vt: 640
    //Height of Vt: 512

    // Delete the largest eigenvalue by setting it to zero
    cv::Mat singularValues = svdS.diag();
    cv::Scalar maxSingularValue;
    cv::Point maxSingularValueIdx;
    cv::minMaxLoc(singularValues, nullptr, &maxSingularValue[0], nullptr, &maxSingularValueIdx);
    singularValues.at<float>(maxSingularValueIdx) = 0.0f;
    cv::Mat updatedS = cv::Mat::diag(singularValues);

    // Adjust the eigenvalues by adding each eigenvalue with the average of the eigenvalues
    cv::Scalar meanSingularValue = cv::mean(updatedS);
    cv::Mat adjustedS = createDiagonalMatrix(meanSingularValue, svdS.rows, svdS.cols);

    svdU.convertTo(svdU, CV_32F);
    adjustedS.convertTo(adjustedS, CV_32F);
    svdVt.convertTo(svdVt, CV_32F);

    // Reconstruct the image using the updated SVD components
    cv::Mat reconstructedImage = svdU * adjustedS * svdVt;
    reconstructedImage.convertTo(reconstructedImage, CV_8U);
    

// Apply Non-Local Means denoising
    cv::Mat denoisedImage;
    cv::fastNlMeansDenoising(/*reconstructedImage*/imRectRight, denoisedImage);

    // Apply median filtering
    cv::Mat enhancedImage;
    cv::medianBlur(denoisedImage, enhancedImage, 3);

    cv::Mat returnImage;
    enhancedImage.convertTo(returnImage, CV_8U);

    return returnImage;
}

cv::Mat Tracking::guidedFilter(const cv::Mat& srcMat, int radius, double eps)
{
    //Use the source Mat as guided Mat
    cv::Mat guidedMat;
    srcMat.convertTo(guidedMat, CV_64FC1);

    //Median Filtering Calc
    cv::Mat mean_I, mean_II;
    boxFilter(guidedMat, mean_I, CV_64FC1, cv::Size(radius, radius));
    boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, cv::Size(radius, radius));

    //Correlation Calc
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);

    //a,b Filtering Calc
    cv::Mat a = var_I / (var_I + eps);
    cv::Mat b = mean_I - a.mul(mean_I);
    cv::Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_64FC1, cv::Size(radius, radius));
    boxFilter(b, mean_b, CV_64FC1, cv::Size(radius, radius));

    //Result
    cv::Mat resultImage;
    cv::Mat dstImage = mean_a.mul(guidedMat) + mean_b;

    dstImage.convertTo(resultImage, CV_8U);
    return resultImage;
}

cv::Mat Tracking::AdaptiveFilter(const cv::Mat& v)
{
    //v - original image; u - kept (low-frequency part); n - (high-frequency part)
    //n = t (Texture) + s (Fixed Pattern Noise)
    cv::Mat u, n, s;

    /**********************************
     * Guided Filter
    ***********************************/

    u = guidedFilter(v, 3, 200);
    //cout<<cv::depthToString(v.depth())<<endl;
    n = v - u;
    s = cv::Mat::zeros(v.size(), CV_64FC1);

    /**********************************
     * Calculate the HDS1d(i) of image
    ***********************************/

    //Step 1. Calc the local grad of image
    cv::Mat grad_vx, grad_vy, abs_grad_vx, abs_grad_vy, grad_v;
    cv::Mat grad_ux;

    cv::Sobel(v, grad_vx, CV_64FC1, 1, 0, 3);
    cv::convertScaleAbs(grad_vx, abs_grad_vx);
    cv::Sobel(v, grad_vy, CV_64FC1, 0, 1, 3);
    cv::convertScaleAbs(grad_vy, abs_grad_vy);
    cv::addWeighted(abs_grad_vx, 0.5, abs_grad_vy, 0.5, 0, grad_v);

    cv::Sobel(u, grad_ux, CV_64FC1, 1, 0, 3);
    cv::Scalar mean_ux, std_ux;
    cv::meanStdDev(grad_ux, mean_ux, std_ux);
    double sigma_r1 = 10 * std_ux[0];

    //Step 2. Construct HDS1d(i)
    cv::Mat HDS = cv::Mat::zeros(v.size(), v.type());
    for (int y = 0; y < v.rows; ++y)
        for (int x = 0; x < v.cols; ++x)
        {
            double numerator = 0;
            double denominator = 0;
            //Set Nh window = 1*9
            int x_min = max(0, x - 4);
            int x_max = min(v.cols - 1, x + 4);
            for (int j = x_min; j <= x_max; ++j)
            {
                double ui = u.at<uchar>(y, x);
                double uj = u.at<uchar>(y, j);
                double Ki = std::exp(-(ui - uj) * (ui - uj) / (2 * sigma_r1 * sigma_r1));
                numerator += Ki * v.at<uchar>(y, j) / 361;
                denominator += Ki;
            }
            HDS.at<double>(y, x) = numerator / denominator;
        }

    /**********************************
     * Construct FPN s(i)
    ***********************************/
    for (int y = 0; y < v.rows; ++y)
        for (int x = 0; x < v.cols; ++x)
        {
            double Ki = v.rows * HDS.at<double>(y, x);
            double nj = 0;
            if (Ki < 1) continue;
            int y_min = max(0, y - int(Ki / 2));
            int y_max = min(v.rows - 1, y + int(Ki / 2));
            for (int j = y_min; j <= y_max; ++j)
                nj += n.at<uchar>(j, x);
            s.at<double>(y, x) = nj / Ki;
            if (nj / Ki == 0 && y < 500 && y>100)
            {
                cout << "y: " << y << " x: " << x << " nj:" << nj << " Ki:" << Ki << endl;
            }
        }

    s.convertTo(s, CV_16U);
    cv::Mat result;
    result = v - s;
    cout << "done" << endl;
    return result;
}
