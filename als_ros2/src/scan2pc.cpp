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

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <cmath>

class Scan2PC : public rclcpp::Node {
private:
    std::string scanName_, pcName_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scanSub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pcPub_;

public:
    Scan2PC(void) : Node("scan2pc"),
        scanName_("/scan"),
        pcName_("/scan_point_cloud")
    {
        this->declare_parameter<std::string>("scan_name", scanName_);
        this->get_parameter("scan_name", scanName_);

        this->declare_parameter<std::string>("pc_name", pcName_);
        this->get_parameter("pc_name", pcName_);

        scanSub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            scanName_, 1, std::bind(&Scan2PC::scanCB, this, std::placeholders::_1));
        pcPub_ = this->create_publisher<sensor_msgs::msg::PointCloud>(
            pcName_, 1);
    }

    void scanCB(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        sensor_msgs::msg::PointCloud pc;
        pc.header = msg->header;
        for (int i = 0; i < (int)msg->ranges.size(); ++i) {
            double r = msg->ranges[i];
            if (r <= msg->range_min || msg->range_max <= r)
                continue;
            double t = msg->angle_min + (double)i * msg->angle_increment;
            double x = r * cos(t);
            double y = r * sin(t);
            geometry_msgs::msg::Point32 p;
            p.x = x;
            p.y = y;
            p.z = 0.0;
            pc.points.push_back(p);
        }
        pcPub_->publish(pc);
    }
}; // class Scan2PC

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Scan2PC>());

    rclcpp::shutdown();
    return 0;
}
