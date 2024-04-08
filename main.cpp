// main.cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "Tracking.h"

int main() {
    // Initialize OpenCV and create a Tracking object
    cv::Mat inputImage = cv::imread("F:/_SLAM/LLVIP/infrared/test/190092.jpg"); // Change "input_image.jpg" to your input image file path
    if (inputImage.empty()) {
        std::cerr << "Error: Could not read the image file." << std::endl;
        return -1;
    }

    // Process the image using the Tracking class
    Tracking tracker;
    cv::Mat processedImage = tracker.guidedFilter(inputImage, 3, 200);
    cv::Mat HighImage = inputImage - processedImage;

    // Save the processed image to a file
    cv::imwrite("processed_image.jpg", inputImage); // Change "processed_image.jpg" to your output image file path
    cv::imwrite("residual_image.jpg", HighImage);

    std::cout << "Image processing complete. Output saved as processed_image.jpg." << std::endl;

    // 创建并打开输出文件
    std::ofstream outputFile("output.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to create output file." << std::endl;
        return 1;
    }

    // 将图像灰度值写入到输出文件
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            // 获取像素的灰度值
            int pixelValue = static_cast<int>(inputImage.at<uchar>(y, x));
            // 写入灰度值到输出文件
            outputFile << pixelValue << " ";
        }
        outputFile << std::endl; // 每一行结束后换行
    }

    // 关闭输出文件
    outputFile.close();

    std::cout << "Output file has been generated successfully." << std::endl;

    return 0;
}
