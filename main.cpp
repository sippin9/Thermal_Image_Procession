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

    // ������������ļ�
    std::ofstream outputFile("output.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to create output file." << std::endl;
        return 1;
    }

    // ��ͼ��Ҷ�ֵд�뵽����ļ�
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            // ��ȡ���صĻҶ�ֵ
            int pixelValue = static_cast<int>(inputImage.at<uchar>(y, x));
            // д��Ҷ�ֵ������ļ�
            outputFile << pixelValue << " ";
        }
        outputFile << std::endl; // ÿһ�н�������
    }

    // �ر�����ļ�
    outputFile.close();

    std::cout << "Output file has been generated successfully." << std::endl;

    return 0;
}
