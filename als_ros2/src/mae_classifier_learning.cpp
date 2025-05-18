/****************************************************************************
 * als_ros: An Advanced Localization System for ROS use with 2D LiDAR
 * Copyright (C) 2022 Naoki Akai
 *
 * Licensed under the Apache License, Version 2.0 (the “License”);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @author Naoki Akai
 ****************************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <als_ros/Pose.h>
#include <als_ros/ClassifierDatasetGenerator.h>
#include <als_ros/MAEClassifier.h>

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("mae_classifier_learning");

    std::vector<std::string> trainDirs, testDirs;
    std::string classifierDir;
    double maxResidualError, maeHistogramBinWidth;

    node->declare_parameter("train_dirs", std::vector<std::string>());
    node->declare_parameter("test_dirs", std::vector<std::string>());
    node->declare_parameter("classifier_dir", std::string(""));
    node->declare_parameter("max_residual_error", 1.0);
    node->declare_parameter("histogram_bin_width", 0.01);

    node->get_parameter("train_dirs", trainDirs);
    node->get_parameter("test_dirs", testDirs);
    node->get_parameter("classifier_dir", classifierDir);
    node->get_parameter("max_residual_error", maxResidualError);
    node->get_parameter("histogram_bin_width", maeHistogramBinWidth);

    RCLCPP_INFO(node->get_logger(), "train_dirs:");
    for (const auto &dir : trainDirs){
        RCLCPP_INFO(node->get_logger(), "  %s", dir.c_str());
    }
    RCLCPP_INFO(node->get_logger(), "test_dirs:");
    for (const auto &dir : testDirs) {
        RCLCPP_INFO(node->get_logger(), "  %s", dir.c_str());
    }
    RCLCPP_INFO(node->get_logger(), "classifier_dir: %s", classifierDir.c_str());
    RCLCPP_INFO(node->get_logger(), "max_residual_error: %.6f", maxResidualError);
    RCLCPP_INFO(node->get_logger(), "histogram_bin_width: %.6f", maeHistogramBinWidth);

    std::vector<als_ros::Pose> gtPosesTrain, successPosesTrain, failurePosesTrain;
    std::vector<sensor_msgs::msg::LaserScan> scansTrain;
    std::vector<std::vector<double>> successResidualErrorsTrain, failureResidualErrorsTrain;

    std::vector<als_ros::Pose> gtPosesTest, successPosesTest, failurePosesTest;
    std::vector<sensor_msgs::msg::LaserScan> scansTest;
    std::vector<std::vector<double>> successResidualErrorsTest, failureResidualErrorsTest;

    als_ros::ClassifierDatasetGenerator generator(false);
    generator.setTrainDirs(trainDirs);
    generator.setTestDirs(testDirs);
    generator.readTrainDataset(gtPosesTrain, successPosesTrain, failurePosesTrain, scansTrain, successResidualErrorsTrain, failureResidualErrorsTrain);
    generator.readTestDataset(gtPosesTest, successPosesTest, failurePosesTest, scansTest, successResidualErrorsTest, failureResidualErrorsTest);

    als_ros::MAEClassifier classifier;
    classifier.setClassifierDir(classifierDir);
    classifier.setMaxResidualError(maxResidualError);
    classifier.setMAEHistogramBinWidth(maeHistogramBinWidth);
    classifier.learnThreshold(successResidualErrorsTrain, failureResidualErrorsTrain);
    classifier.writeClassifierParams(successResidualErrorsTest, failureResidualErrorsTest);
    classifier.writeDecisionLikelihoods();

    return 0;
}
