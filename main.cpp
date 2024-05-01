#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <vector>
#include "Tracking.h"

namespace fs = std::filesystem;

int main() { 
    
    // Counter for processed images
    int processedCount = 0;
    
    /*
    //
    //    Part1: Infrared Left
    //

    // Directory containing the sequence of images
    std::string sequenceDir = "F:/_SLAM/STheReO/image/stereo_thermal_14_left/";
    std::string outputDir = "F:/_SLAM/STheReO/image/stereo_thermal_14_left_adapt/";

    std::vector<std::string> filenames;

    // Iterate over each file in the directory and collect filenames
    for (const auto& entry : fs::directory_iterator(sequenceDir)) {
        if (fs::is_regular_file(entry.path())) {
            filenames.push_back(entry.path().string());
        }
    }

    // Sort filenames
    std::sort(filenames.begin(), filenames.end());

    // Iterate over each file in the directory
    for (const auto& imagePath : filenames) {
        // Read the image
        cv::Mat rawImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        if (rawImage.empty()) {
            std::cerr << "Error: Could not read the image file: " << imagePath << std::endl;
            continue; // Move to the next image if this one couldn't be read
        }

        // Transform the 14-bit thermal images into 8-bit using normalization
        cv::Mat normalizedImage = Tracking::PreProcess(rawImage);

        // Create new filename based on the sequence number
        std::string filename = std::to_string(processedCount + 1) + ".png";

        // Save the processed image with the same filename
        std::string processedImagePath = outputDir + filename;
        cv::imwrite(processedImagePath, normalizedImage);

        // Increment the processed count
        processedCount++;

        // Break out of the loop if processedCount reaches 200
        if (processedCount >= 200)
            break;
    }
    
    // Directory containing the sequence of images
    sequenceDir = "F:/_SLAM/STheReO/image/stereo_thermal_14_left_adapt/";
    outputDir = "F:/_SLAM/STheReO/image/stereo_thermal_14_left_edges/";

    std::vector<std::string> filenames01;

    // Iterate over each file in the directory and collect filenames
    for (const auto& entry : fs::directory_iterator(sequenceDir)) {
        if (fs::is_regular_file(entry.path())) {
            filenames01.push_back(entry.path().string());
        }
    }

    // Sort filenames
    std::sort(filenames01.begin(), filenames01.end());

    processedCount = 0;

    // Iterate over each file in the directory
    for (const auto& imagePath : filenames01) {
        // Read the image
        cv::Mat rawImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (rawImage.empty()) {
            std::cerr << "Error: Could not read the image file: " << imagePath << std::endl;
            continue; // Move to the next image if this one couldn't be read
        }

        // Apply Canny edge detection
        cv::Mat edges;
        double lower_thresh = 180; // Lower threshold for Canny
        double upper_thresh = 200; // Upper threshold for Canny
        cv::Canny(rawImage, edges, lower_thresh, upper_thresh);

        //cv::FAST();

        // Create new filename based on the sequence number
        std::string filename = "thermal_" + std::to_string(processedCount + 1) + "_edges.png";

        // Save the processed image with the new filename
        std::string processedImagePath = outputDir + filename;
        cv::imwrite(processedImagePath, edges);

        // Increment the processed count
        processedCount++;

        // Break out of the loop if processedCount reaches 200
        if (processedCount >= 200)
            break;
    }
    

    //
    //  Part2: Visible Left
    //
    
    processedCount = 0;
    
    std::string sequenceDir1 = "F:/_SLAM/STheReO/image/stereo_left/";
    std::string outputDir1 = "F:/_SLAM/STheReO/image/stereo_left_output/";

    std::vector<std::string> filenames1;

    // Iterate over each file in the directory and collect filenames
    for (const auto& entry : fs::directory_iterator(sequenceDir1)) {
        if (fs::is_regular_file(entry.path())) {
            filenames1.push_back(entry.path().string());
        }
    }

    // Sort filenames
    std::sort(filenames1.begin(), filenames1.end());

    // Iterate over each file in the directory
    for (const auto& imagePath : filenames1) {
        // Read the image
        cv::Mat rawImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        if (rawImage.empty()) {
            std::cerr << "Error: Could not read the image file: " << imagePath << std::endl;
            continue; // Move to the next image if this one couldn't be read
        }

        // Calibrate visible
        cv::Mat normalizedImage = Tracking::PreProcess1(rawImage);

        // Create new filename based on the sequence number
        std::string filename = std::to_string(processedCount + 1) + ".png";

        // Save the processed image with the same filename
        std::string processedImagePath = outputDir1 + filename;
        cv::imwrite(processedImagePath, normalizedImage);

        // Increment the processed count
        processedCount++;

        // Break out of the loop if processedCount reaches 200
        if (processedCount >= 200)
            break;
    }

    // Directory containing the sequence of images
    sequenceDir1 = "F:/_SLAM/STheReO/image/stereo_left_output/";
    outputDir1 = "F:/_SLAM/STheReO/image/stereo_left_edges/";

    std::vector<std::string> filenames11;

    // Iterate over each file in the directory and collect filenames
    for (const auto& entry : fs::directory_iterator(sequenceDir1)) {
        if (fs::is_regular_file(entry.path())) {
            filenames11.push_back(entry.path().string());
        }
    }

    // Sort filenames
    std::sort(filenames11.begin(), filenames11.end());

    processedCount = 0;

    // Iterate over each file in the directory
    for (const auto& imagePath : filenames11) {
        // Read the image
        cv::Mat rawImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (rawImage.empty()) {
            std::cerr << "Error: Could not read the image file: " << imagePath << std::endl;
            continue; // Move to the next image if this one couldn't be read
        }

        // Apply Canny edge detection
        cv::Mat edges;
        double lower_thresh = 30; // Lower threshold for Canny
        double upper_thresh = 90; // Upper threshold for Canny
        cv::Canny(rawImage, edges, lower_thresh, upper_thresh);

        // Create new filename based on the sequence number
        std::string filename = std::to_string(processedCount + 1) + "_edges.png";

        // Save the processed image with the new filename
        std::string processedImagePath = outputDir1 + filename;
        cv::imwrite(processedImagePath, edges);

        // Increment the processed count
        processedCount++;

        // Break out of the loop if processedCount reaches 200
        if (processedCount >= 200)
            break;
    }
    */

    //
    //  Part3: ORB Matcher
    //
    
    // Define the paths to the folders containing left and right images
    std::string left_folder_path = "F:/_SLAM/STheReO/image/stereo_left_edges/";
    std::string right_folder_path = "F:/_SLAM/STheReO/image/stereo_thermal_14_left_edges/";

    // Define the folder to save the matched images
    std::string output_folder_path = "F:/_SLAM/STheReO/image/matched_images/";
    std::string output_folder_path1 = "F:/_SLAM/STheReO/image/fast_left/";
    std::string output_folder_path2 = "F:/_SLAM/STheReO/image/fast_right/";

    // Initialize ORB detector and descriptor extractor
    cv::Ptr<cv::ORB> orb_detector = cv::ORB::create();

    // Loop through left images folder
    for (const auto& entry : fs::directory_iterator(left_folder_path)) {
        std::string left_image_path = entry.path().string();

        // Read left image
        cv::Mat left_image = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);

        // Check if the image is successfully read
        if (left_image.empty()) {
            std::cerr << "Could not read the left image: " << left_image_path << std::endl;
            continue; // Move to the next image
        }

        // Extract the filename without extension
        std::string filename = fs::path(left_image_path).filename().string();

        // Construct the corresponding right image path
        std::string right_image_path = right_folder_path + "thermal_" + filename; // Assuming same filenames in right folder

        // Read right image
        cv::Mat right_image = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

        // Check if the right image is successfully read
        if (right_image.empty()) {
            std::cerr << "Could not read the right image: " << right_image_path << std::endl;
            continue; // Move to the next image
        }

        // Detect ORB keypoints and compute descriptors for left and right images
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        Tracking::FAST_t(left_image,	//待检测的图像,可见光图像
            keypoints1,			//存储角点位置的容器
            true);				//使能非极大值抑制
        Tracking::FAST_t(right_image,	//待检测的图像,红外,高噪音
            keypoints2,			//存储角点位置的容器
            true);				//使能非极大值抑制

        // Draw keypoints on both images
        cv::Mat left_image_with_keypoints, right_image_with_keypoints;
        cv::drawKeypoints(left_image, keypoints1, left_image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(right_image, keypoints2, right_image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        // Save matched image to output folder
        std::string output_image_path1 = output_folder_path1 + filename;
        std::string output_image_path2 = output_folder_path2 + filename;
        cv::imwrite(output_image_path1, left_image_with_keypoints);
        cv::imwrite(output_image_path2, right_image_with_keypoints);
        
        /*
        cv::Mat descriptors1, descriptors2;
        orb_detector->detectAndCompute(left_image, cv::noArray(), keypoints1, descriptors1);
        orb_detector->detectAndCompute(right_image, cv::noArray(), keypoints2, descriptors2);

        // Convert descriptors to float type required by FLANN matcher
        if (descriptors1.type() != CV_32F) {
            descriptors1.convertTo(descriptors1, CV_32F);
        }
        if (descriptors2.type() != CV_32F) {
            descriptors2.convertTo(descriptors2, CV_32F);
        }
        

        // Use FLANN based matcher
        cv::FlannBasedMatcher matcher;
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        // Draw matches
        cv::Mat matched_image;
        cv::drawMatches(left_image, keypoints1, right_image, keypoints2, matches, matched_image);

        // Save matched image to output folder
        std::string output_image_path = output_folder_path + filename;
        cv::imwrite(output_image_path, matched_image);

        std::cout << "Saved matched image: " << output_image_path << std::endl;
        */
    }
    

    return 0;
}
