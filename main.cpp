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

void filterMatchesByEpipolarConstraint(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches) {
    // Convert keypoints into Point2f
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Compute the Fundamental matrix using RANSAC to filter out outliers
    std::vector<uchar> inliers_mask(matches.size());
    cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99, inliers_mask);

    // Filter matches using the inliers mask
    std::vector<cv::DMatch> inlier_matches;
    for (size_t i = 0; i < matches.size(); i++) {
        if (inliers_mask[i]) {
            inlier_matches.push_back(matches[i]);
        }
    }

    // Update the original matches with the filtered inlier matches
    matches.swap(inlier_matches);
}


int main() { 
    /*
    // Counter for processed images
    int processedCount = 0;
    
    
    //
    //    Part1: Infrared Left
    //

    // Directory containing the sequence of images
    std::string sequenceDir = "F:/_SLAM/Valley/image/stereo_thermal_14_left/";
    std::string outputDir = "F:/_SLAM/Valley/image/stereo_thermal_14_left_adapt/";

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
    sequenceDir = "F:/_SLAM/Valley/image/stereo_thermal_14_left_adapt/";
    outputDir = "F:/_SLAM/Valley/image/stereo_thermal_14_left_edges/";

    std::vector<std::string> filenames01;

    // Iterate over each file in the directory and collect filenames
    for (const auto& entry : fs::directory_iterator(sequenceDir)) {
        if (fs::is_regular_file(entry.path())) {
            filenames01.push_back(entry.path().string());
        }
    }

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
        std::string filename = fs::path(imagePath).filename().string();

        // Save the processed image with the new filename
        std::string processedImagePath = outputDir + filename;
        cv::imwrite(processedImagePath, edges);

    }

    //
    //  Part2: Visible Left
    //
    
    processedCount = 0;
    
    std::string sequenceDir1 = "F:/_SLAM/Valley/image/stereo_left/";
    std::string outputDir1 = "F:/_SLAM/Valley/image/stereo_left_adapt/";

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
    sequenceDir1 = "F:/_SLAM/Valley/image/stereo_left_adapt/";
    outputDir1 = "F:/_SLAM/Valley/image/stereo_left_edges/";

    std::vector<std::string> filenames11;

    // Iterate over each file in the directory and collect filenames
    for (const auto& entry : fs::directory_iterator(sequenceDir1)) {
        if (fs::is_regular_file(entry.path())) {
            filenames11.push_back(entry.path().string());
        }
    }

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
        std::string filename = fs::path(imagePath).filename().string();

        // Save the processed image with the new filename
        std::string processedImagePath = outputDir1 + filename;
        cv::imwrite(processedImagePath, edges);
    }
    
    
    //
    //  Part3: ORB Matcher
    //
    
    // Define the paths to the folders containing left and right images
    std::string left_folder_path = "F:/_SLAM/Valley/image/stereo_thermal_14_left_edges/";
    std::string left_folder_path_des = "F:/_SLAM/Valley/image/stereo_thermal_14_left_adapt/";
    std::string right_folder_path = "F:/_SLAM/Valley/image/stereo_left_edges/";
    std::string right_folder_path_des = "F:/_SLAM/Valley/image/stereo_left_adapt/";

    // Define the folder to save the matched images
    std::string output_folder_path = "F:/_SLAM/Valley/image/matched_images/";
    std::string output_folder_path_canny = "F:/_SLAM/Valley/image/matched_images_canny/";
    std::string output_folder_path1 = "F:/_SLAM/Valley/image/fast_left/";
    std::string output_folder_path2 = "F:/_SLAM/Valley/image/fast_right/";

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
        std::string right_image_path = right_folder_path + filename; // Assuming same filenames in right folder
        std::string left_image_path_des = left_folder_path_des + filename;
        std::string right_image_path_des = right_folder_path_des + filename;

        // Read right image
        cv::Mat right_image = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat left_image_des = cv::imread(left_image_path_des, cv::IMREAD_GRAYSCALE);
        cv::Mat right_image_des = cv::imread(right_image_path_des, cv::IMREAD_GRAYSCALE);
        if (left_image_des.empty()) {
            std::cerr << "Could not read the leftdes image: " << left_image_path_des << std::endl;
            continue; // Move to the next image
        }
        // Check if the right image is successfully read
        if (right_image_des.empty()) {
            std::cerr << "Could not read the des image: " << right_image_path_des << std::endl;
            continue; // Move to the next image
        }

        // Detect ORB keypoints and compute descriptors for left and right images
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        cv::Mat descriptors11, descriptors22;

        Tracking::FAST_t(left_image,	//待检测的图像,可见光图像
            keypoints1,			//存储角点位置的容器
            true);				//使能非极大值抑制
        Tracking::FAST_t(right_image,	//待检测的图像,红外,高噪音
            keypoints2,			//存储角点位置的容器
            true);				//使能非极大值抑制

        // Draw keypoints on both images
        cv::Mat left_image_with_keypoints, right_image_with_keypoints;
        cv::drawKeypoints(left_image_des, keypoints1, left_image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(right_image_des, keypoints2, right_image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        // Save matched image to output folder
        std::string output_image_path1 = output_folder_path1 + filename;
        std::string output_image_path2 = output_folder_path2 + filename;
        cv::imwrite(output_image_path1, left_image_with_keypoints);
        cv::imwrite(output_image_path2, right_image_with_keypoints);      
        
        // Create ORB descriptor extractor
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        cv::Ptr<cv::ORB> orb1 = cv::ORB::create();

        // Compute descriptors
        orb->compute(left_image_des, keypoints1, descriptors1);
        orb->compute(right_image_des, keypoints2, descriptors2);

        orb1->compute(left_image, keypoints1, descriptors11);
        orb1->compute(right_image, keypoints2, descriptors22);

        // Convert descriptors to float type required by FLANN matcher
        if (descriptors1.type() != CV_32F) {
            descriptors1.convertTo(descriptors1, CV_32F);
        }
        if (descriptors2.type() != CV_32F) {
            descriptors2.convertTo(descriptors2, CV_32F);
        }  
        if (descriptors11.type() != CV_32F) {
            descriptors11.convertTo(descriptors11, CV_32F);
        }
        if (descriptors22.type() != CV_32F) {
            descriptors22.convertTo(descriptors22, CV_32F);
        }

        // Use FLANN based matcher
        cv::FlannBasedMatcher matcher;
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        cv::FlannBasedMatcher matcher1;
        std::vector<cv::DMatch> matches1;
        matcher1.match(descriptors11, descriptors22, matches1);

        // Filter matches based on the epipolar constraint
        filterMatchesByEpipolarConstraint(keypoints1, keypoints2, matches);
        filterMatchesByEpipolarConstraint(keypoints1, keypoints2, matches1);

        // Draw matches
        cv::Mat matched_image;
        cv::drawMatches(left_image_des, keypoints1, right_image_des, keypoints2, matches, matched_image);

        cv::Mat matched_image1;
        cv::drawMatches(left_image, keypoints1, right_image, keypoints2, matches1, matched_image1);

        // Save matched image to output folder
        std::string output_image_path = output_folder_path + filename;
        cv::imwrite(output_image_path, matched_image);

        std::string output_image_path_canny = output_folder_path_canny + filename;
        cv::imwrite(output_image_path_canny, matched_image1);
    }
    */

    //
    //  Part4: Edge Fusion
    //

    
    // Load the images from the two cameras
    cv::Mat image11 = cv::imread("F:/_SLAM/Valley/image/stereo_thermal_14_left_edges/1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image12 = cv::imread("F:/_SLAM/Valley/image/stereo_left_edges/1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image1 = cv::imread("F:/_SLAM/Valley/image/stereo_thermal_14_left_adapt/1.png", cv::IMREAD_GRAYSCALE);

    cv::Mat image21 = cv::imread("F:/_SLAM/Valley/image/stereo_thermal_14_left_edges/2.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image22 = cv::imread("F:/_SLAM/Valley/image/stereo_left_edges/2.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("F:/_SLAM/Valley/image/stereo_thermal_14_left_adapt/1.png", cv::IMREAD_GRAYSCALE);

    // Define the homography matrix (example values here, you should use your computed matrix)
    cv::Mat homography_matrix = (cv::Mat_<double>(3, 3) << 1.026881464, -0.009055988, -20.19330972,
        0.009055988, 1.026881464, -15.005,
        0.0, 0.0, 1.0);
    
    // Create a matrix to store the transformed image
    cv::Mat transformed_image1, transformed_image2, transformed_image1raw, transformed_image2raw;
    // Warp the first image to align with the second image
    cv::warpPerspective(image11, transformed_image1, homography_matrix, image12.size());
    cv::warpPerspective(image21, transformed_image2, homography_matrix, image22.size());
    cv::warpPerspective(image1, transformed_image1raw, homography_matrix, image1.size());
    cv::warpPerspective(image2, transformed_image2raw, homography_matrix, image2.size());
    
    // Create a matrix to store the fused image
    cv::Mat fused_image1, fused_image2;
    // Average the transformed image and the second image
    cv::addWeighted(transformed_image1, 0.5, image12, 0.5, 0, fused_image1);
    cv::addWeighted(transformed_image2, 0.5, image22, 0.5, 0, fused_image2);
    
    std::string output_image_path_addedge1 = "F:/_SLAM/Valley/image/fused1.png";
    std::string output_image_path_addedge2 = "F:/_SLAM/Valley/image/fused2.png";
    cv::imwrite(output_image_path_addedge1, fused_image1);
    cv::imwrite(output_image_path_addedge2, fused_image2);
    
    std::vector<cv::KeyPoint> keypointfuse1, keypointfuse2;
    cv::Mat descriptors1, descriptors2;

    //TODO: Enhance score in FAST-t: 255 scores 2, 123 scores 1

    Tracking::FAST_t(fused_image1,	//待检测的图像,可见光图像
        keypointfuse1,			//存储角点位置的容器
        true);				//使能非极大值抑制
    Tracking::FAST_t(fused_image2,	//待检测的图像,可见光图像
        keypointfuse2,			//存储角点位置的容器
        true);				//使能非极大值抑制
    
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->compute(transformed_image1raw, keypointfuse1, descriptors1);
    orb->compute(transformed_image2raw, keypointfuse2, descriptors2);
    
    // Convert descriptors to float type required by FLANN matcher
    if (descriptors1.type() != CV_32F) {
        descriptors1.convertTo(descriptors1, CV_32F);
    }
    if (descriptors2.type() != CV_32F) {
        descriptors2.convertTo(descriptors2, CV_32F);
    }

    // Use FLANN based matcher
    cv::FlannBasedMatcher matcherfuse;
    std::vector<cv::DMatch> matchesfuse;
    matcherfuse.match(descriptors1, descriptors2, matchesfuse);
    
    // Filter matches based on the epipolar constraint
    filterMatchesByEpipolarConstraint(keypointfuse1, keypointfuse2, matchesfuse);

    // Draw keypoints on both images
    cv::Mat left_image_with_keypoints, right_image_with_keypoints;
    cv::drawKeypoints(transformed_image1raw, keypointfuse1, left_image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(transformed_image2raw, keypointfuse2, right_image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imwrite("F:/_SLAM/Valley/image/fast1.png", left_image_with_keypoints);
    cv::imwrite("F:/_SLAM/Valley/image/fast2.png", right_image_with_keypoints);

    cv::Mat matched_imagefuse;
    cv::drawMatches(transformed_image1raw, keypointfuse1, transformed_image2raw, keypointfuse2, matchesfuse, matched_imagefuse);

    std::string output_fusedimage_path = "F:/_SLAM/Valley/image/match.png";
    cv::imwrite(output_fusedimage_path, matched_imagefuse);
    
    return 0;
}