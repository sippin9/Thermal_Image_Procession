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

cv::Mat Tracking::PreGamma(const cv::Mat& imRectLeft, float gamma)
{
    // Apply gamma correction
    cv::Mat enhanced = imRectLeft;
    for (int i = 0; i < enhanced.rows; ++i)
        for (int j = 0; j < enhanced.cols; ++j)
        {
            double normalizedValue = enhanced.at<uchar>(i, j) / 255.0;
            double correctedValue = std::pow(normalizedValue, gamma) * 255.0;
            enhanced.at<uchar>(i, j) = static_cast<uchar>(correctedValue);
        }
    enhanced.convertTo(enhanced, CV_8U);
    return enhanced;
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

    dstImage.convertTo(resultImage, CV_64FC1);
    return resultImage;
}

cv::Mat Tracking::AdaptiveFilter(const cv::Mat& vv)
{
    //v - original image; u - kept (low-frequency part); n - (high-frequency part)
    //n = t (Texture) + s (Fixed Pattern Noise)
    cv::Mat v = cv::Mat::zeros(vv.size(), vv.type());
    vv.convertTo(v, CV_64FC1);
    v = v / 255;
    cv::Mat u = cv::Mat::zeros(v.size(), v.type());
    cv::Mat n = cv::Mat::zeros(v.size(), v.type());
    cv::Mat s = cv::Mat::zeros(v.size(), v.type());

    /**********************************
     * Guided Filter
    ***********************************/

    u = guidedFilter(v, 3, 0.01);
    //cout<<cv::depthToString(u.depth())<<endl;
    n = v - u;

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
    double sigma_r1 = 15 * std_ux[0];

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
                double ui = u.at<double>(y, x);
                double uj = u.at<double>(y, j);
                double Ki = std::exp(-(ui - uj) * (ui - uj) / (2 * sigma_r1 * sigma_r1));
                numerator += Ki * v.at<double>(y, j);
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
                nj += n.at<double>(j, x);
            s.at<double>(y, x) = nj / Ki;
            // nj becomes negative
        }
    s.convertTo(s, CV_64FC1);

    cv::Mat imageresult = cv::Mat::zeros(v.size(), CV_8U);
    for (int y = 0; y < imageresult.rows; ++y)
        for (int x = 0; x < imageresult.cols; ++x)
        {
            double val = 255 * (v.at<double>(y, x) - s.at<double>(y, x));
            imageresult.at<uchar>(y, x) = (int)val;
        }

    //cv::Mat denoisedImage;
    //cv::fastNlMeansDenoising(/*reconstructedImage*/imageresult, denoisedImage);

    // Apply median filtering
    /*
    cv::Mat enhancedImage;
    cv::medianBlur(denoisedImage, enhancedImage, 3);
    enhancedImage.convertTo(enhancedImage, CV_8U);
    */
    //imageresult = imageresult * 255;
    //cout << "Adaptive done" << endl;
    return imageresult;
}

cv::Mat Tracking::PreProcess(const cv::Mat& im)
{
    // Get the dimensions of the input image
    int width = im.cols;
    int height = im.rows;

    // Define the region of interest (ROI) coordinates
    int roiX = (width - 640) / 2;
    int roiY = (height - 304) / 2;
    int roiWidth = 640;
    int roiHeight = 304;

    // Create a ROI (Region of Interest) from the input image
    cv::Rect roiRect(roiX, roiY, roiWidth, roiHeight);
    cv::Mat inputImage = im(roiRect).clone();

    double valMin, valMax;
    cv::Mat mImGrayP = inputImage;
    cv::minMaxLoc(mImGrayP, &valMin, &valMax);

    cv::Mat mImGrayRead = cv::Mat::zeros(mImGrayP.size(), CV_8U);
    for (int y = 0; y < mImGrayP.rows; ++y)
        for (int x = 0; x < mImGrayP.cols; ++x)
        {
            double nn = (mImGrayP.at<ushort>(y, x) - valMin) * 255 / (valMax - valMin);
            mImGrayRead.at<uchar>(y, x) = (int)nn;
        }

    cv::Mat mImGrayGamma = PreGamma(mImGrayRead, 0.65);

    cv::Mat mImGrayHist = cv::Mat::zeros(mImGrayP.size(), CV_8U);
    //cv::equalizeHist(mImGrayRead, mImGrayHist);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(8); // (int)(4.(88)/256)
    clahe->setTilesGridSize(cv::Size(8, 8)); // 将图像分为8*8块
    clahe->apply(mImGrayRead, mImGrayHist);

    cv::Mat mImGrayAdapt = 1 * mImGrayHist + 0 * mImGrayGamma;

    //Adaptive FPN filter useful for dark images
    mImGrayP = AdaptiveFilter(mImGrayAdapt);
    //mImGrayP = mImGrayAdapt;

    //Too slow as denoising
    //cv::fastNlMeansDenoising(mImGrayP, mImGrayP);

    return mImGrayP;
}

cv::Mat Tracking::PreProcess1(const cv::Mat& inputImage) {
    static double fx1, fy1, fx2, fy2;
    fx1 = 429.433;
    fx2 = 788.413;
    fy1 = 429.531;
    fy2 = 790.926;

    cv::Mat medianImage;
    cv::medianBlur(inputImage, medianImage, 5);

    cv::Mat scaledImage;
    cv::resize(medianImage, scaledImage, cv::Size(), fx1 / fx2, fy1 / fy2);

    // Get the dimensions of the input image
    int width = scaledImage.cols;
    int height = scaledImage.rows;

    // Define the region of interest (ROI) coordinates
    int roiX = (width - 640) / 2;
    int roiY = (height - 304) / 2;
    int roiWidth = 640;
    int roiHeight = 304;

    // Create a ROI (Region of Interest) from the input image
    cv::Rect roiRect(roiX, roiY, roiWidth, roiHeight);
    cv::Mat croppedImage = scaledImage(roiRect).clone();

    //cv::Mat claheImage;
    //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    //clahe->setClipLimit(8); // (int)(4.(88)/256)
    //clahe->setTilesGridSize(cv::Size(8, 8)); // 将图像分为8*8块
    //clahe->apply(croppedImage, claheImage);

    //cv::Mat denoisedImage;
    //cv::medianBlur(croppedImage, denoisedImage, 5);
    //cv::bilateralFilter(croppedImage, denoisedImage, 9, 75, 75);
    //cv::fastNlMeansDenoising(croppedImage, denoisedImage, 10, 7, 21);

    cv::Mat gammaImage = PreGamma(croppedImage, 0.5);

    // Return image
    return gammaImage;
}
