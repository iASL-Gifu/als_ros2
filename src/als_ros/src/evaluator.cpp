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
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

typedef struct {
    double x, y;
} Point;

class Evaluator : public rclcpp::Node {
private:
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gtPoseSub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr reliabilitySub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr glPosesSub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scanSub_;
    std::string mapFrame_, laserFrame_;
    std::shared_ptr<tf2_ros::TransformListener> tfListener_;
    std::shared_ptr<tf2_ros::Buffer> tfBuffer_;
    std::vector<geometry_msgs::msg::PoseStamped> gtPoses_;
    std::vector<geometry_msgs::msg::PoseArray> glPoses_;
    std::vector<sensor_msgs::msg::LaserScan> scans_;
    bool canUpdateGTPoses_, canUpdateGLPoses_, canUpdateScan_;
    bool saveGTPose_, saveGLPoses_, saveScan_;

public:
    Evaluator(void): Node("evaluator"),
        mapFrame_("map"),
        laserFrame_("laser"),
        canUpdateGTPoses_(true),
        canUpdateGLPoses_(true),
        canUpdateScan_(true)
    {
        gtPoseSub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/scanned_ground_truth_pose", 1, std::bind(&Evaluator::gtPoseCB, this, std::placeholders::_1));
        reliabilitySub_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
            "/reliability", 1, std::bind(&Evaluator::reliabilityCB, this, std::placeholders::_1));
        glPosesSub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/gl_sampled_poses", 1, std::bind(&Evaluator::glPosesCB, this, std::placeholders::_1));
        scanSub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 1, std::bind(&Evaluator::scanCB, this, std::placeholders::_1));
        tfBuffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tfBuffer_->setUsingDedicatedThread(true);
        tfListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_, this, false);
        saveGTPose_ = false;
        saveGLPoses_ = false;
        saveScan_ = false;
    }

    void gtPoseCB(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        if (!canUpdateGTPoses_)
            return;
        gtPoses_.insert(gtPoses_.begin(), *msg);
        if (gtPoses_.size() >= 100)
            gtPoses_.resize(100);
    }

    int getSynchronizedGTPoses(double time) {
        for (int i = 0; i < (int)gtPoses_.size(); ++i) {
            double t = rclcpp::Time(gtPoses_[i].header.stamp).seconds();
            if (t < time)
                return i;
        }
        return -1;
    }

    void glPosesCB(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        if (!canUpdateGLPoses_)
            return;
        glPoses_.insert(glPoses_.begin(), *msg);
        if (glPoses_.size() >= 100)
            glPoses_.resize(100);
    }

    int getSynchronizedGLPoses(double time) {
        for (int i = 0; i < (int)glPoses_.size(); ++i) {
            double t = rclcpp::Time(glPoses_[i].header.stamp).seconds();
            if (t < time)
                return i;
        }
        return -1;
    }

    void scanCB(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        if (!canUpdateScan_)
            return;
        scans_.insert(scans_.begin(), *msg);
        if (scans_.size() >= 100)
            scans_.resize(100);
    }

    int getSynchronizedScan(double time) {
        for (int i = 0; i < (int)scans_.size(); ++i) {
            double t = rclcpp::Time(scans_[i].header.stamp).seconds();
            if (t < time)
                return i;
        }
        return -1;
    }

    std::vector<Point> makeArrowPoints(double x, double y, double yaw, double len) {
        Point p0, p1, p2, p3;
        p0.x = x;
        p0.y = y;
        p1.x = p0.x + len * cos(yaw);
        p1.y = p0.y + len * sin(yaw);
        p2.x = p1.x + len / 3.0 * cos(yaw + 135.0 * M_PI / 180.0);
        p2.y = p1.y + len / 3.0 * sin(yaw + 135.0 * M_PI / 180.0);
        p3.x = p1.x + len / 3.0 * cos(yaw - 135.0 * M_PI / 180.0);
        p3.y = p1.y + len / 3.0 * sin(yaw - 135.0 * M_PI / 180.0);

        std::vector<Point> arrowPoints;
        arrowPoints.push_back(p0);
        arrowPoints.push_back(p1);
        arrowPoints.push_back(p2);
        arrowPoints.push_back(p3);
        arrowPoints.push_back(p1);
        return arrowPoints;
    }

    void reliabilityCB(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
        static bool isFirst = true;
        static double firstTime;
        static FILE *fp = fopen("/tmp/als_ros_reliability.txt", "w");
        double time = rclcpp::Time(msg->header.stamp).seconds();
        if (isFirst) {
            firstTime = time;
            isFirst = false;
        }

        geometry_msgs::msg::PoseStamped gtPose;
        if (saveGTPose_) {
            canUpdateGTPoses_ = false;
            int gtPoseIdx = getSynchronizedGTPoses(time);
            if (gtPoseIdx < 0) {
                canUpdateGTPoses_ = true;
                return;
            }
            gtPose = gtPoses_[gtPoseIdx];
            canUpdateGTPoses_ = true;
        }

        geometry_msgs::msg::PoseArray glPoses;
        if (saveGLPoses_) {
            canUpdateGLPoses_ = false;
            int glPoseIdx = getSynchronizedGLPoses(time);
            if (glPoseIdx < 0) {
                canUpdateGLPoses_ = true;
                return;
            }
            glPoses = glPoses_[glPoseIdx];
            canUpdateGLPoses_ = true;
        }

        sensor_msgs::msg::LaserScan scan;
        if (saveScan_) {
            canUpdateScan_ = false;
            int scanIdx = getSynchronizedScan(time);
            if (scanIdx < 0) {
                canUpdateScan_ = true;
                return;
            }
            scan = scans_[scanIdx];
            canUpdateScan_ = true;
        }

        geometry_msgs::msg::TransformStamped tfMap2Laser;
        try {
            tfMap2Laser = tfBuffer_->lookupTransform(mapFrame_, laserFrame_, msg->header.stamp);
        } catch (tf2::TransformException &ex) {
            return;
        }

        double gtX, gtY, gtYaw;
        if (saveGTPose_) {
            tf2::Quaternion gtQuat(gtPose.pose.orientation.x,
                gtPose.pose.orientation.y,
                gtPose.pose.orientation.z,
                gtPose.pose.orientation.w);
            double gtRoll, gtPitch;
            tf2::Matrix3x3(gtQuat).getRPY(gtRoll, gtPitch, gtYaw);
            gtX = gtPose.pose.position.x;
            gtY = gtPose.pose.position.y;
        }

        tf2::Transform tf_map2laser;
        tf2::fromMsg(tfMap2Laser.transform, tf_map2laser);
        tf2::Quaternion quat = tf_map2laser.getRotation();
        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        double x = tfMap2Laser.transform.translation.x;
        double y = tfMap2Laser.transform.translation.y;

        if (saveGTPose_) {
            double dx = gtX - x;
            double dy = gtY - y;
            double dl = sqrt(dx * dx + dy * dy);
            double dyaw = gtYaw - yaw;
            while (dyaw < -M_PI)
                dyaw += 2.0 * M_PI;
            while (dyaw > M_PI)
                dyaw -= 2.0 * M_PI;
            printf("%.2lf [sec], %.3lf [m], %.3lf [m], %.3lf [m], %.3lf [deg], %lf %lf %lf\n",
                time - firstTime, dx, dy, dl, dyaw * 180.0 / M_PI, msg->vector.x, msg->vector.y, msg->vector.z);

            fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                time - firstTime, gtX, gtY, gtYaw, x, y, yaw, 
                dx, dy, dl, dyaw, msg->vector.x, msg->vector.y, msg->vector.z);
        } else {
            printf("%.2lf [sec], %.3lf [m], %.3lf [m], %.3lf [deg], %lf %lf %lf\n",
                time - firstTime, x, y, yaw * 180.0 / M_PI,
                msg->vector.x, msg->vector.y, msg->vector.z);

            fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf\n",
                time - firstTime, x, y, yaw, 
                msg->vector.x, msg->vector.y, msg->vector.z);
        }

        if (saveGLPoses_) {
            FILE *fp2;
            fp2 = fopen("/tmp/als_ros_scaned_ground_truth.txt", "w");
            std::vector<Point> gtArrowPoints = makeArrowPoints(gtX, gtY, gtYaw, 1.0);
            for (int i = 0; i < (int)gtArrowPoints.size(); ++i)
                fprintf(fp2, "%lf %lf\n", gtArrowPoints[i].x, gtArrowPoints[i].y);
            fprintf(fp2, "\n");
            fclose(fp2);

            fp2 = fopen("/tmp/als_ros_mcl_estimate.txt", "w");
            std::vector<Point> mclArrowPoints = makeArrowPoints(x, y, yaw, 1.0);
            for (int i = 0; i < (int)mclArrowPoints.size(); ++i)
                fprintf(fp2, "%lf %lf\n", mclArrowPoints[i].x, mclArrowPoints[i].y);
            fprintf(fp2, "\n");
            fclose(fp2);

            double tmpYaw = 0.0, estX = 0.0, estY = 0.0, estYaw = 0.0;
            double wo = 1.0 / (double)glPoses.poses.size();
            fp2 = fopen("/tmp/als_ros_gl_sampled_poses.txt", "w");
            for (int i = 0; i < (int)glPoses.poses.size(); ++i) {
                tf2::Quaternion quat(glPoses.poses[i].orientation.x,
                    glPoses.poses[i].orientation.y,
                    glPoses.poses[i].orientation.z,
                    glPoses.poses[i].orientation.w);
                double roll, pitch, yaw;
                tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
                double x = glPoses.poses[i].position.x;
                double y = glPoses.poses[i].position.y;
                std::vector<Point> arrowPoints = makeArrowPoints(x, y, yaw, 1.0);
                for (int j = 0; j < (int)arrowPoints.size(); ++j)
                    fprintf(fp2, "%lf %lf\n", arrowPoints[j].x, arrowPoints[j].y);
                fprintf(fp2, "\n");

                estX += x * wo;
                estY += y * wo;
                double dyaw = tmpYaw - yaw;
                while (dyaw < -M_PI)
                    dyaw += 2.0 * M_PI;
                while (dyaw > M_PI)
                    dyaw -= 2.0 * M_PI;
                estYaw += dyaw * wo;
            }
            fclose(fp2);

            estYaw = tmpYaw - estYaw;
            static FILE *fpGT = fopen("/tmp/als_ros_scaned_ground_truth_poses.txt", "w");
            static FILE *fpEst = fopen("/tmp/als_ros_mcl_poses.txt", "w");
            static FILE *fpGLAve = fopen("/tmp/als_ros_gl_sampled_poses_ave.txt", "w");
            fprintf(fpGT, "%lf %lf %lf\n", gtX, gtY, gtYaw);
            fprintf(fpEst, "%lf %lf %lf\n", x, y, yaw);
            fprintf(fpGLAve, "%lf %lf %lf\n", estX, estY, estYaw);
        }

        if (saveScan_) {
            FILE *fp2 = fopen("/tmp/als_ros_scan_points.txt", "w");
            for (int i = 0; i < (int)scan.ranges.size(); ++i) {
                double r = scan.ranges[i];
                if (r < scan.range_min || scan.range_max < r)
                    continue;
                double t = (double)i * scan.angle_increment + scan.angle_min + gtYaw;
                double x = r * cos(t) + gtX;
                double y = r * sin(t) + gtY;
                fprintf(fp2, "%lf %lf\n", x, y);
            }
            fclose(fp2);
        }
    }
}; // class Evaluator

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Evaluator>());

    rclcpp::shutdown();
    return 0;
}
