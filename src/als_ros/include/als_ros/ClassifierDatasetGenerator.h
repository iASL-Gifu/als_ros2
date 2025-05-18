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

#ifndef __CLASSIFIER_DATASET_GENERATOR_H__
#define __CLASSIFIER_DATASET_GENERATOR_H__

#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <opencv2/opencv.hpp>
#include <als_ros/Pose.h>

namespace als_ros {

class Obstacle {
public:
    double x_, y_, s_;
    // x: x position
    // y: y position
    // s: size

    Obstacle(void):
        x_(0.0), y_(0.0), s_(1.0) {}

    Obstacle(double x, double y, double s):
        x_(x), y_(y), s_(s) {}
}; // class Obstacle

class ClassifierDatasetGenerator : public rclcpp::Node {
private:
    std::string mapName_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr mapSub_;

    int generateSampleNum_;
    std::string saveDir_;
    std::vector<std::string> trainDirs_, testDirs_;

    nav_msgs::msg::OccupancyGrid map_;
    cv::Mat distMap_;
    double mapResolution_;
    Pose mapOrigin_;
    int mapWidth_, mapHeight_;
    bool gotMap_;
    double freeSpaceMinX_, freeSpaceMaxX_, freeSpaceMinY_, freeSpaceMaxY_;

    int obstaclesNum_;

    double angleMin_, angleMax_, angleIncrement_, rangeMin_, rangeMax_, scanAngleNoise_, scanRangeNoise_;
    double validScanRateTH_;

    double failurePositionalErrorTH_, failureAngularErrorTH_;
    double positionalErrorMax_, angularErrorMax_;

public:
    void generateDataset(void) {
        for (int i = 0; i < generateSampleNum_; ++i) {
            std::vector<Obstacle> obstacles = generateObstacles();
            nav_msgs::msg::OccupancyGrid simMap = buildSimulationMap(obstacles);
            Pose gtPose, successPose, failurePose;
            generatePoses(gtPose, successPose, failurePose);
            sensor_msgs::msg::LaserScan scan = simulateScan(gtPose, simMap);
            if (!isValidScan(scan)) {
                RCLCPP_INFO(this->get_logger(), "Simulated scan is invalid.");
                i--;
                continue;
            }
            std::vector<double> successResidualErrors = getResidualErrors(successPose, scan);
            std::vector<double> failureResidualErrors = getResidualErrors(failurePose, scan);
            saveData(gtPose, successPose, failurePose, scan, successResidualErrors, failureResidualErrors);
            if ((i + 1) % 10 == 0)
                RCLCPP_INFO(this->get_logger(), "%.2lf [%%] process done.", (double)(i + 1) / (double)generateSampleNum_ * 100.0);
            if (!rclcpp::ok()) {
                break;
            }
        }
    }

    ClassifierDatasetGenerator(bool auto_start_generation = true) : Node("classifier_dataset_generator"),
        mapName_("/map"),
        generateSampleNum_(1000),
        saveDir_("/tmp/classifier_dataset/"),
        obstaclesNum_(20),
        angleMin_(-135.0),
        angleMax_(135.0),
        angleIncrement_(0.25),
        rangeMin_(0.02),
        rangeMax_(30.0),
        scanAngleNoise_(0.01),
        scanRangeNoise_(0.05),
        validScanRateTH_(0.1),
        failurePositionalErrorTH_(0.2),
        failureAngularErrorTH_(2.0),
        positionalErrorMax_(0.5),
        angularErrorMax_(5.0),
        gotMap_(false)
    {
        if (!auto_start_generation) return;

        RCLCPP_INFO(this->get_logger(), "ClassifierDatasetGenerator is initializing...");

        srand((unsigned int)time(NULL));

        this->declare_parameter("map_name", mapName_);
        this->get_parameter("map_name", mapName_);

        this->declare_parameter("generate_sample_num", generateSampleNum_);
        this->get_parameter("generate_sample_num", generateSampleNum_);

        this->declare_parameter("save_dir", saveDir_);
        this->get_parameter("save_dir", saveDir_);

        this->declare_parameter("obstacles_num", obstaclesNum_);
        this->get_parameter("obstacles_num", obstaclesNum_);

        this->declare_parameter("angle_min", angleMin_);
        this->get_parameter("angle_min", angleMin_);

        this->declare_parameter("angle_max", angleMax_);
        this->get_parameter("angle_max", angleMax_);

        this->declare_parameter("angle_increment", angleIncrement_);
        this->get_parameter("angle_increment", angleIncrement_);

        this->declare_parameter("range_min", rangeMin_);
        this->get_parameter("range_min", rangeMin_);

        this->declare_parameter("range_max", rangeMax_);
        this->get_parameter("range_max", rangeMax_);

        this->declare_parameter("scan_angle_noise", scanAngleNoise_);
        this->get_parameter("scan_angle_noise", scanAngleNoise_);

        this->declare_parameter("scan_range_noise", scanRangeNoise_);
        this->get_parameter("scan_range_noise", scanRangeNoise_);

        this->declare_parameter("valid_scan_rate_th", validScanRateTH_);
        this->get_parameter("valid_scan_rate_th", validScanRateTH_);

        this->declare_parameter("failure_positional_error_th", failurePositionalErrorTH_);
        this->get_parameter("failure_positional_error_th", failurePositionalErrorTH_);

        this->declare_parameter("failure_angular_error_th", failureAngularErrorTH_);
        this->get_parameter("failure_angular_error_th", failureAngularErrorTH_);

        this->declare_parameter("positional_error_max", positionalErrorMax_);
        this->get_parameter("positional_error_max", positionalErrorMax_);

        this->declare_parameter("angular_error_max", angularErrorMax_);
        this->get_parameter("angular_error_max", angularErrorMax_);

        RCLCPP_INFO(this->get_logger(), "  map_name: %s", mapName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  generate_sample_num: %d", generateSampleNum_);
        RCLCPP_INFO(this->get_logger(), "  save_dir: %s", saveDir_.c_str());
        RCLCPP_INFO(this->get_logger(), "  obstacles_num: %d", obstaclesNum_);
        RCLCPP_INFO(this->get_logger(), "  angle_min: %.3f", angleMin_);
        RCLCPP_INFO(this->get_logger(), "  angle_max: %.3f", angleMax_);
        RCLCPP_INFO(this->get_logger(), "  angle_increment: %.3f", angleIncrement_);
        RCLCPP_INFO(this->get_logger(), "  range_min: %.3f", rangeMin_);
        RCLCPP_INFO(this->get_logger(), "  range_max: %.3f", rangeMax_);
        RCLCPP_INFO(this->get_logger(), "  scan_angle_noise: %.3f", scanAngleNoise_);
        RCLCPP_INFO(this->get_logger(), "  scan_range_noise: %.3f", scanRangeNoise_);
        RCLCPP_INFO(this->get_logger(), "  valid_scan_rate_th: %.3f", validScanRateTH_);
        RCLCPP_INFO(this->get_logger(), "  failure_positional_error_th: %.3f", failurePositionalErrorTH_);
        RCLCPP_INFO(this->get_logger(), "  failure_angular_error_th: %.3f", failureAngularErrorTH_);
        RCLCPP_INFO(this->get_logger(), "  positional_error_max: %.3f", positionalErrorMax_);
        RCLCPP_INFO(this->get_logger(), "  angular_error_max: %.3f", angularErrorMax_);

        std::string cmd;
        int retVal;
        cmd = "mkdir -p " + saveDir_;
        retVal = system(cmd.c_str());
        cmd = "rm -rf " + saveDir_ + "/*";
        retVal = system(cmd.c_str());

        angleMin_ *= M_PI / 180.0;
        angleMax_ *= M_PI / 180.0;
        angleIncrement_ *= M_PI / 180.0;
        scanAngleNoise_ *= M_PI / 180.0;
        failureAngularErrorTH_ *= M_PI / 180.0;
        angularErrorMax_ *= M_PI / 180.0;

        mapSub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            mapName_, 1, std::bind(&ClassifierDatasetGenerator::mapCB, this, std::placeholders::_1));

        int cnt = 0;
        rclcpp::Rate rate(10);
        while (rclcpp::ok()) {
            rclcpp::spin_some(this->get_node_base_interface());
            if (gotMap_) break;
            cnt++;
            if (cnt >= 300) {
                RCLCPP_ERROR(this->get_logger(), "A map message might not be published.");
                rclcpp::shutdown();
                return;
            }
            rate.sleep();
        }
        RCLCPP_INFO(this->get_logger(), "Map data is received.");
        RCLCPP_INFO(this->get_logger(), "Start generating dataset.");

        generateDataset();
    }

    inline void setTrainDirs(std::vector<std::string> trainDirs) {
        trainDirs_ = trainDirs;
    }

    inline void setTestDirs(std::vector<std::string> testDirs) {
        testDirs_ = testDirs;
    }

    void readTrainDataset(std::vector<Pose> &gtPoses, std::vector<Pose> &successPoses, std::vector<Pose> &failurePoses,
        std::vector<sensor_msgs::msg::LaserScan> &scans, std::vector<std::vector<double>> &successResidualErrors, std::vector<std::vector<double>> &failureResidualErrors)
    {
        readDataset(trainDirs_, gtPoses, successPoses, failurePoses, scans, successResidualErrors, failureResidualErrors);
    }

    void readTestDataset(std::vector<Pose> &gtPoses, std::vector<Pose> &successPoses, std::vector<Pose> &failurePoses,
        std::vector<sensor_msgs::msg::LaserScan> &scans, std::vector<std::vector<double>> &successResidualErrors, std::vector<std::vector<double>> &failureResidualErrors)
    {
        readDataset(testDirs_, gtPoses, successPoses, failurePoses, scans, successResidualErrors, failureResidualErrors);
    }

private:
    inline double nrand(double n) {
        return (n * sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * rand() / RAND_MAX));
    }

    inline double urand(double min, double max) {
        return ((max - min)  * (double)rand() / RAND_MAX + min);
    }

    inline bool onMap(int u, int v) {
        if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_)
            return true;
        else
            return false;
    }

    inline void xy2uv(double x, double y, int *u, int *v) {
        double dx = x - mapOrigin_.getX();
        double dy = y - mapOrigin_.getY();
        double yaw = -mapOrigin_.getYaw();
        double xx = dx * cos(yaw) - dy * sin(yaw);
        double yy = dx * sin(yaw) + dy * cos(yaw);
        *u = (int)(xx / mapResolution_);
        *v = (int)(yy / mapResolution_);
    }

    inline void uv2xy(int u, int v, double *x, double *y) {
        double xx = (double)u * mapResolution_;
        double yy = (double)v * mapResolution_;
        double yaw = mapOrigin_.getYaw();
        double dx = xx * cos(yaw) - yy * sin(yaw);
        double dy = xx * sin(yaw) + yy * cos(yaw);
        *x = dx + mapOrigin_.getX();
        *y = dy + mapOrigin_.getY();
    }

    inline int xy2node(double x, double y) {
        int u, v;
        xy2uv(x, y, &u, &v);
        if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_)
            return v * mapWidth_ + u;
        else
            return -1;
    }

    void mapCB(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        map_ = *msg;

        // perform distance transform to build the distance field
        mapWidth_ = msg->info.width;
        mapHeight_ = msg->info.height;
        mapResolution_ = msg->info.resolution;
        tf2::Quaternion q(msg->info.origin.orientation.x, 
            msg->info.origin.orientation.y, 
            msg->info.origin.orientation.z,
            msg->info.origin.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3 m(q);
        m.getRPY(roll, pitch, yaw);
        mapOrigin_.setX(msg->info.origin.position.x);
        mapOrigin_.setY(msg->info.origin.position.y);
        mapOrigin_.setYaw(yaw);

        cv::Mat binMap(mapHeight_, mapWidth_, CV_8UC1);
        bool isFirst = true;
        for (int v = 0; v < mapHeight_; v++) {
            for (int u = 0; u < mapWidth_; u++) {
                int node = v * mapWidth_ + u;
                int val = msg->data[node];
                if (val == 100) {
                    binMap.at<uchar>(v, u) = 0;
                } else {
                    binMap.at<uchar>(v, u) = 1;
                    if (val == 0) {
                        double x, y;
                        uv2xy(u, v, &x, &y);
                        if (isFirst) {
                            freeSpaceMinX_ = freeSpaceMaxX_ = x;
                            freeSpaceMinY_ = freeSpaceMaxY_ = y;
                            isFirst = false;
                        } else {
                            if (freeSpaceMinX_ > x)
                                freeSpaceMinX_ = x;
                            if (freeSpaceMaxX_ < x)
                                freeSpaceMaxX_ = x;
                            if (freeSpaceMinY_ > y)
                                freeSpaceMinY_ = y;
                            if (freeSpaceMaxY_ < y)
                                freeSpaceMaxY_ = y;
                        }
                    }
                }
            }
        }
        cv::Mat distMap(mapHeight_, mapWidth_, CV_32FC1);
        cv::distanceTransform(binMap, distMap, cv::DIST_L2, 5);
        for (int v = 0; v < mapHeight_; v++) {
            for (int u = 0; u < mapWidth_; u++) {
                float d = distMap.at<float>(v, u) * (float)mapResolution_;
                distMap.at<float>(v, u) = d;
            }
        }
        distMap_ = distMap;
        gotMap_ = true;
    }

    std::vector<Obstacle> generateObstacles(void) {
        std::vector<Obstacle> obstacles(obstaclesNum_);
        for (int i = 0; i < obstaclesNum_; ++i) {
            obstacles[i].x_ = urand(freeSpaceMinX_, freeSpaceMaxX_);
            obstacles[i].y_ = urand(freeSpaceMinY_, freeSpaceMaxY_);
            obstacles[i].s_ = 1.0 + nrand(0.5);
        }
        return obstacles;
    }

    nav_msgs::msg::OccupancyGrid buildSimulationMap(std::vector<Obstacle> obstacles) {
        nav_msgs::msg::OccupancyGrid simMap = map_;
        for (int i = 0; i < (int)obstacles.size(); ++i) {
            double helfSize = obstacles[i].s_ / 2.0;
            for (double x = obstacles[i].x_ - helfSize; x <= obstacles[i].x_ + helfSize; x += mapResolution_) {
                for (double y = obstacles[i].y_ - helfSize; y <= obstacles[i].y_ + helfSize; y += mapResolution_) {
                    int node = xy2node(x, y);
                    if (node >= 0)
                        simMap.data[node] = 100;
                }
            }
        }
        return simMap;
    }

    void generatePoses(Pose &gtPose, Pose &successPose, Pose &failurePose) {
        for (;;) {
            double x = urand(freeSpaceMinX_, freeSpaceMaxX_);
            double y = urand(freeSpaceMinY_, freeSpaceMaxY_);
            double yaw = urand(-M_PI, M_PI);
            int node = xy2node(x, y);
            if (node < 0)
                continue;
            int u, v;
            xy2uv(x, y, &u, &v);
            float dist = distMap_.at<float>(v, u);
            if (map_.data[node] == 0 && dist > 0.5f) {
                gtPose.setX(x);
                gtPose.setY(y);
                gtPose.setYaw(yaw);
                break;
            }
        }

        double x = urand(-failurePositionalErrorTH_ * 0.99, failurePositionalErrorTH_ * 0.99);
        double y = urand(-failurePositionalErrorTH_ * 0.99, failurePositionalErrorTH_ * 0.99);
        double yaw = urand(-failureAngularErrorTH_ * 0.99, failureAngularErrorTH_ * 0.99);
        successPose.setX(gtPose.getX() + x);
        successPose.setY(gtPose.getY() + y);
        successPose.setYaw(gtPose.getYaw() + yaw);

        for (;;) {
            double x = urand(-positionalErrorMax_, positionalErrorMax_);
            double y = urand(-positionalErrorMax_, positionalErrorMax_);
            double yaw = urand(-angularErrorMax_, angularErrorMax_);
            if (fabs(x) >= failurePositionalErrorTH_ || fabs(y) >= failureAngularErrorTH_ || fabs(yaw) >= failureAngularErrorTH_) {
                failurePose.setX(gtPose.getX() + x);
                failurePose.setY(gtPose.getY() + y);
                failurePose.setYaw(gtPose.getYaw() + yaw);
                break;
            }
        }
    }

    sensor_msgs::msg::LaserScan simulateScan(Pose gtPose, nav_msgs::msg::OccupancyGrid simMap) {
        sensor_msgs::msg::LaserScan scan;
        scan.angle_min = angleMin_;
        scan.angle_max = angleMax_;
        scan.angle_increment = angleIncrement_;
        scan.range_min = rangeMin_;
        scan.range_max = rangeMax_;
        int scanNum = (int)((angleMax_ - angleMin_) / angleIncrement_) + 1;
        scan.ranges.resize(scanNum, 0.0);
        for (int i = 0; i < scanNum; ++i) {
            double t = gtPose.getYaw() + (double)i * angleIncrement_ + angleMin_ + nrand(scanAngleNoise_);
            double x = gtPose.getX();
            double y = gtPose.getY();
            double dx = mapResolution_ * cos(t);
            double dy = mapResolution_ * sin(t);
            double range = -1.0;
            for (double r = 0.0; r <= rangeMax_; r += mapResolution_) {
                int node = xy2node(x, y);
                if (node > 0) {
                    if (simMap.data[node] == 100) {
                        range = r;
                        break;
                    }
                } else {
                    break;
                }
                x += dx;
                y += dy;
            }
            if (range >= 0.0)
                scan.ranges[i] = range + nrand(scanRangeNoise_);
        }
        return scan;
    }

    bool isValidScan(sensor_msgs::msg::LaserScan scan) {
        int validScanNum = 0;
        for (int i = 0; i < (int)scan.ranges.size(); ++i) {
            double r = scan.ranges[i];
            if (scan.range_min <= r && r <= scan.range_max)
                validScanNum++;
        }
        double validScanRate = (double)validScanNum / (double)scan.ranges.size();
        if (validScanRate >= validScanRateTH_)
            return true;
        else
            return false;
    }

    std::vector<double> getResidualErrors(Pose pose, sensor_msgs::msg::LaserScan scan) {
        int size = (int)scan.ranges.size();
        std::vector<double> residualErrors(size);
        for (int i = 0; i < size; ++i) {
            double r = scan.ranges[i];
            if (r <= scan.range_min || scan.range_max <= r) {
                residualErrors[i] = -1.0;
                continue;
            }
            double t = (double)i * scan.angle_increment + scan.angle_min + pose.getYaw();
            double x = r * cos(t) + pose.getX();
            double y = r * sin(t) + pose.getY();
            int u, v;
            xy2uv(x, y, &u, &v);
            if (onMap(u, v)) {
                double dist = (double)distMap_.at<float>(v, u);
                residualErrors[i] = dist;
            } else {
                residualErrors[i] = -1.0;
            }
        }
        return residualErrors;
    }

    void saveData(Pose gtPose, Pose successPose, Pose failurePose, sensor_msgs::msg::LaserScan scan, std::vector<double> successResidualErrors, std::vector<double> failureResidualErrors) {
        static bool isFirst = true;
        static int cnt = 0;
        std::string fname;
        FILE *fp;
        if (isFirst) {
            fname = saveDir_ + "/scan_param.txt";
            fp = fopen(fname.c_str(), "w");
            fprintf(fp, "angle_min: %lf\n", angleMin_);
            fprintf(fp, "angle_max: %lf\n", angleMax_);
            fprintf(fp, "angle_increment: %lf\n", angleIncrement_);
            fprintf(fp, "range_min: %lf\n", rangeMin_);
            fprintf(fp, "range_max: %lf\n", rangeMax_);
            fclose(fp);

            std::string cmd = "mkdir -p " + saveDir_ + "/scan_and_residual_errors/";
            int retVal = system(cmd.c_str());
            isFirst = false;
        }

        fname = saveDir_ + "/gt_poses.txt";
        fp = fopen(fname.c_str(), "a");
        fprintf(fp, "%lf %lf %lf\n", gtPose.getX(), gtPose.getY(), gtPose.getYaw());
        fclose(fp);

        fname = saveDir_ + "/success_poses.txt";
        fp = fopen(fname.c_str(), "a");
        fprintf(fp, "%lf %lf %lf\n", successPose.getX(), successPose.getY(), successPose.getYaw());
        fclose(fp);

        fname = saveDir_ + "/failure_poses.txt";
        fp = fopen(fname.c_str(), "a");
        fprintf(fp, "%lf %lf %lf\n", failurePose.getX(), failurePose.getY(), failurePose.getYaw());
        fclose(fp);

        fname = saveDir_ + "/scan_and_residual_errors/" + std::to_string(cnt) + ".txt";
        fp = fopen(fname.c_str(), "w");
        for (int i = 0; i < (int)scan.ranges.size(); ++i)
            fprintf(fp, "%d %lf %lf %lf\n", i, scan.ranges[i], successResidualErrors[i], failureResidualErrors[i]);
        fclose(fp);

        cnt++;
    }

    void readDataset(std::vector<std::string> dirs, std::vector<Pose> &gtPoses, std::vector<Pose> &successPoses, std::vector<Pose> &failurePoses,
        std::vector<sensor_msgs::msg::LaserScan> &scans, std::vector<std::vector<double>> &successResidualErrors, std::vector<std::vector<double>> &failureResidualErrors)
    {
        FILE *fp;
        std::string fname;
        double x, y, yaw;
        for (int dirIdx = 0; dirIdx < (int)dirs.size(); ++dirIdx) {
            fname = dirs[dirIdx] + "/scan_param.txt";
            fp = fopen(fname.c_str(), "r");
            char buf[32];
            int retVal;
            retVal = fscanf(fp, "%s %lf", buf, &angleMin_);
            retVal = fscanf(fp, "%s %lf", buf, &angleMax_);
            retVal = fscanf(fp, "%s %lf", buf, &angleIncrement_);
            retVal = fscanf(fp, "%s %lf", buf, &rangeMin_);
            retVal = fscanf(fp, "%s %lf", buf, &rangeMax_);
            retVal = fclose(fp);

            sensor_msgs::msg::LaserScan scan;
            scan.angle_min = angleMin_;
            scan.angle_max = angleMax_;
            scan.angle_increment = angleIncrement_;
            scan.range_min = rangeMin_;
            scan.range_max = rangeMax_;
            int scanSize = (int)((angleMax_ - angleMin_) / angleIncrement_) + 1;
            scan.ranges.resize(scanSize);
            scan.intensities.resize(scanSize);

            fname = dirs[dirIdx] + "/gt_poses.txt";
            fp = fopen(fname.c_str(), "r");
            while (fscanf(fp, "%lf %lf %lf\n", &x, &y, &yaw) != EOF) {
                Pose p(x, y, yaw);
                gtPoses.push_back(p);
            }
            fclose(fp);

            fname = dirs[dirIdx] + "/success_poses.txt";
            fp = fopen(fname.c_str(), "r");
            while (fscanf(fp, "%lf %lf %lf\n", &x, &y, &yaw) != EOF) {
                Pose p(x, y, yaw);
                successPoses.push_back(p);
            }
            fclose(fp);

            fname = dirs[dirIdx] + "/failure_poses.txt";
            fp = fopen(fname.c_str(), "r");
            while (fscanf(fp, "%lf %lf %lf\n", &x, &y, &yaw) != EOF) {
                Pose p(x, y, yaw);
                failurePoses.push_back(p);
            }
            fclose(fp);

            int dataNum = (int)gtPoses.size();
            for (int i = 0; i < dataNum; ++i) {
                fname = dirs[dirIdx] + "/scan_and_residual_errors/" + std::to_string(i) + ".txt";
                fp = fopen(fname.c_str(), "r");
                int idx;
                std::vector<double> srErrors(scanSize), frErrors(scanSize);
                for (int j = 0; j < scanSize; ++j)
                    retVal = fscanf(fp, "%d %f %lf %lf\n", &idx, &scan.ranges[j], &srErrors[j], &frErrors[j]);
                scans.push_back(scan);
                successResidualErrors.push_back(srErrors);
                failureResidualErrors.push_back(frErrors);
                fclose(fp);
            }
        }
    }
}; // class ClassifierDatasetGenerator

} // namespace als_ros

#endif // __CLASSIFIER_DATASET_GENERATOR_H__
