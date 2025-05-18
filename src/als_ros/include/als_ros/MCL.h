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

#ifndef __MCL_H__
#define __MCL_H__

#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <als_ros/Pose.h>
#include <als_ros/Particle.h>
#include <als_ros/MAEClassifier.h>


namespace als_ros {

class MCL : public rclcpp::Node {
private:
    // subscribers
    std::string scanName_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scanSub_;

    std::string odomName_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub_;

    std::string mapName_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr mapSub_;

    std::string glSampledPosesName_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr glSampledPosesSub_;

    std::string initialPoseName_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialPoseSub_;

    // publishers
    std::string poseName_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr posePub_;

    std::string particlesName_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particlesPub_;

    std::string unknownScanName_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr unknownScanPub_;

    std::string residualErrorsName_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr residualErrorsPub_;

    std::string reliabilityName_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr reliabilityPub_;

    std::string reliabilityMarkerName_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr reliabilityMarkerPub_;

    // tf frames
    std::string laserFrame_, baseLinkFrame_, mapFrame_, odomFrame_;
    bool broadcastTF_, useOdomTF_;

    // poses
    double initialPoseX_, initialPoseY_, initialPoseYaw_;
    Pose mclPose_, baseLink2Laser_, odomPose_;
    rclcpp::Time mclPoseStamp_, odomPoseStamp_, glSampledPosesStamp_;

    // particles
    int particlesNum_;
    std::vector<Particle> particles_;
    double initialNoiseX_, initialNoiseY_, initialNoiseYaw_;
    bool useAugmentedMCL_, addRandomParticlesInResampling_;
    double randomParticlesRate_;
    std::vector<double> randomParticlesNoise_;
    int glParticlesNum_;
    std::vector<Particle> glParticles_;
    geometry_msgs::msg::PoseArray glSampledPoses_;
    bool canUpdateGLSampledPoses_, canUseGLSampledPoses_, isGLSampledPosesUpdated_;
    double glSampledPoseTimeTH_, gmmPositionalVariance_, gmmAngularVariance_;
    double predDistUnifRate_;

    // map
    cv::Mat distMap_;
    double mapResolution_;
    Pose mapOrigin_;
    int mapWidth_, mapHeight_;
    bool gotMap_;

    // motion
    double deltaX_, deltaY_, deltaDist_, deltaYaw_;
    double deltaXSum_, deltaYSum_, deltaDistSum_, deltaYawSum_, deltaTimeSum_;
    std::vector<double> resampleThresholds_;
    std::vector<double> odomNoiseDDM_, odomNoiseODM_;
    bool useOmniDirectionalModel_;

    // measurements
    sensor_msgs::msg::LaserScan scan_, unknownScan_;
    bool canUpdateScan_;
    std::vector<bool> likelihoodShiftedSteps_;

    // measurement model
    // 0: likelihood field model, 1: beam model, 2: class conditional measurement model
    int measurementModelType_;
    double zHit_, zShort_, zMax_, zRand_;
    double varHit_, lambdaShort_, lambdaUnknown_;
    double normConstHit_, denomHit_, pRand_;
    double measurementModelRandom_, measurementModelInvalidScan_;
    double pKnownPrior_, pUnknownPrior_, unknownScanProbThreshold_;
    double alphaSlow_, alphaFast_, omegaSlow_, omegaFast_;
    int scanStep_;
    bool rejectUnknownScan_, publishUnknownScan_, publishResidualErrors_;
    bool gotScan_, scanMightInvalid_;
    double resampleThresholdESS_;

    // localization result
    double totalLikelihood_, averageLikelihood_, maxLikelihood_;
    double amclRandomParticlesRate_, effectiveSampleSize_;
    int maxLikelihoodParticleIdx_;

    // other parameters
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;
    std::shared_ptr<tf2_ros::TransformListener> tfListener_;
    std::shared_ptr<tf2_ros::Buffer> tfBuffer_;
    bool isInitialized_;
    double localizationHz_;
    double transformTolerance_;

    // reliability estimation
    bool estimateReliability_;
    int classifierType_;
    double reliability_;
    std::vector<double> reliabilities_, glSampledPosesReliabilities_;
    std::vector<double> relTransDDM_, relTransODM_;

    // mean absolute error (MAE)-based failure detector
    MAEClassifier maeClassifier_;
    std::string maeClassifierDir_;
    std::vector<double> maes_, glSampledPosesMAEs_;

    // global-localization-based pose sampling
    bool useGLPoseSampler_, fuseGLPoseSamplerOnlyUnreliable_;

    // pose record
    bool writePose_;
    std::string poseLogFile_;

    // constant parameters
    const double rad2deg_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    bool startMCL_;

public:
    // inline setting functions
    inline void setCanUpdateScan(bool canUpdateScan) {
        canUpdateScan_ = canUpdateScan;
        if (!canUpdateScan_) {
            int invalidScanNum = 0;
            for (int i = 0; i < (int)scan_.ranges.size(); ++i) {
                double r = scan_.ranges[i];
                if (r < scan_.range_min || scan_.range_max < r)
                    invalidScanNum++;
            }
            double invalidScanRate = (double)invalidScanNum / (int)scan_.ranges.size();
            if (invalidScanRate > 0.95) {
                scanMightInvalid_ = true;
                RCLCPP_ERROR(this->get_logger(), "MCL scan might invalid.");
            } else {
                scanMightInvalid_ = false;
            }
        }
    }

    // inline getting functions
    inline double getLocalizationHz(void) { return localizationHz_; }
    inline std::string getMapFrame(void) { return mapFrame_; }
    inline sensor_msgs::msg::LaserScan getScan(void) { return scan_; }
    inline int getParticlesNum(void) { return particlesNum_; }
    inline Pose getParticlePose(int i) { return particles_[i].getPose(); }
    inline double getParticleW(int i) { return particles_[i].getW(); }
    inline Pose getBaseLink2Laser(void) { return baseLink2Laser_; }
    inline int getScanStep(void) { return scanStep_; }
    inline int getMaxLikelihoodParticleIdx(void) { return maxLikelihoodParticleIdx_; }
    inline double getNormConstHit(void) { return normConstHit_; }
    inline double getDenomHit(void) { return denomHit_; }
    inline double getZHit(void) { return zHit_; }
    inline double getMeasurementModelRandom(void) { return measurementModelRandom_; }

    // inline setting functions
    inline void setMCLPoseStamp(rclcpp::Time stamp) { mclPoseStamp_ = stamp; }
    inline void setParticleW(int i, double w) { particles_[i].setW(w); }
    inline void setTotalLikelihood(double totalLikelihood) { totalLikelihood_ = totalLikelihood; }
    inline void setAverageLikelihood(double averageLikelihood) { averageLikelihood_ = averageLikelihood; }
    inline void setMaxLikelihood(double maxLikelihood) { maxLikelihood_ = maxLikelihood; }
    inline void setMaxLikelihoodParticleIdx(int maxLikelihoodParticleIdx) { maxLikelihoodParticleIdx_ = maxLikelihoodParticleIdx; }

    // inline other functions
    void clearLikelihoodShiftedSteps(void) { likelihoodShiftedSteps_.clear(); }
    void addLikelihoodShiftedSteps(bool flag) { likelihoodShiftedSteps_.push_back(flag); }

    MCL() : Node("mcl"),
        scanName_("/scan"),
        odomName_("/odom"),
        mapName_("/map"),
        poseName_("/mcl_pose"),
        particlesName_("/mcl_particles"),
        unknownScanName_("/unknown_scan"),
        residualErrorsName_("/residual_errors"),
        reliabilityName_("/reliability"),
        glSampledPosesName_("/gl_sampled_poses"),
        reliabilityMarkerName_("/reliability_marker_name"),
        laserFrame_("laser"),
        baseLinkFrame_("base_link"),
        mapFrame_("map"),
        odomFrame_("odom"),
        broadcastTF_(true),
        useOdomTF_(true),
        initialPoseX_(0.0),
        initialPoseY_(0.0),
        initialPoseYaw_(0.0),
        initialNoiseX_(0.02),
        initialNoiseY_(0.02),
        initialNoiseYaw_(0.02),
        particlesNum_(100),
        useAugmentedMCL_(false),
        addRandomParticlesInResampling_(true),
        randomParticlesRate_(0.1),
        randomParticlesNoise_({0.1, 0.1, 0.01}),
        odomNoiseDDM_({1.0, 0.5, 0.5, 1.0}),
        odomNoiseODM_({1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0}),
        deltaXSum_(0.0),
        deltaYSum_(0.0),
        deltaDistSum_(0.0),
        deltaYawSum_(0.0),
        deltaTimeSum_(0.0),
        resampleThresholds_({-1.0, -1.0, -1.0, -1.0, -1.0}),
        useOmniDirectionalModel_(false),
        measurementModelType_(0),
        pKnownPrior_(0.5),
        pUnknownPrior_(0.5),
        scanStep_(10),
        zHit_(0.9),
        zShort_(0.2),
        zMax_(0.05),
        zRand_(0.05),
        varHit_(0.1),
        lambdaShort_(3.0),
        lambdaUnknown_(1.0),
        alphaSlow_(0.001),
        alphaFast_(0.9),
        omegaSlow_(0.0),
        omegaFast_(0.0),
        rejectUnknownScan_(false),
        publishUnknownScan_(false),
        publishResidualErrors_(false),
        resampleThresholdESS_(0.5),
        estimateReliability_(true),
        relTransDDM_({0.0, 0.0}),
        relTransODM_({0.0, 0.0, 0.0}),
        classifierType_(0),
        maeClassifierDir_("/home/akai/Dropbox/git/als_ros/src/als_ros/classifiers/MAE/"),
        useGLPoseSampler_(false),
        fuseGLPoseSamplerOnlyUnreliable_(false),
        glSampledPoseTimeTH_(0.5),
        gmmPositionalVariance_(0.1),
        gmmAngularVariance_(0.1),
        predDistUnifRate_(0.05),
        localizationHz_(10.0),
        transformTolerance_(1.0),
        gotMap_(false),
        scanMightInvalid_(false),
        gotScan_(false),
        isInitialized_(true),
        canUpdateScan_(true),
        canUpdateGLSampledPoses_(true),
        canUseGLSampledPoses_(false),
        isGLSampledPosesUpdated_(false),
        writePose_(false),
        poseLogFile_("/tmp/als_ros_pose.txt"),
        rad2deg_(180.0 / M_PI),
        startMCL_(true)
    {
        RCLCPP_INFO(this->get_logger(), "MCL is initializing...");

        // topic and frame names
        this->declare_parameter<std::string>("scan_name", scanName_);
        this->get_parameter("scan_name", scanName_);

        this->declare_parameter<std::string>("odom_name", odomName_);
        this->get_parameter("odom_name", odomName_);

        this->declare_parameter<std::string>("map_name", mapName_);
        this->get_parameter("map_name", mapName_);

        this->declare_parameter<std::string>("pose_name", poseName_);
        this->get_parameter("pose_name", poseName_);

        this->declare_parameter<std::string>("particles_name", particlesName_);
        this->get_parameter("particles_name", particlesName_);

        this->declare_parameter<std::string>("unknown_scan_name", unknownScanName_);
        this->get_parameter("unknown_scan_name", unknownScanName_);

        this->declare_parameter<std::string>("residual_errors_name", residualErrorsName_);
        this->get_parameter("residual_errors_name", residualErrorsName_);

        this->declare_parameter<std::string>("reliability_name", reliabilityName_);
        this->get_parameter("reliability_name", reliabilityName_);

        this->declare_parameter<std::string>("gl_sampled_poses_name", glSampledPosesName_);
        this->get_parameter("gl_sampled_poses_name", glSampledPosesName_);

        this->declare_parameter<std::string>("laser_frame", laserFrame_);
        this->get_parameter("laser_frame", laserFrame_);

        this->declare_parameter<std::string>("base_link_frame", baseLinkFrame_);
        this->get_parameter("base_link_frame", baseLinkFrame_);

        this->declare_parameter<std::string>("map_frame", mapFrame_);
        this->get_parameter("map_frame", mapFrame_);

        this->declare_parameter<std::string>("odom_frame", odomFrame_);
        this->get_parameter("odom_frame", odomFrame_);

        this->declare_parameter<bool>("broadcast_tf", broadcastTF_);
        this->get_parameter("broadcast_tf", broadcastTF_);

        this->declare_parameter<bool>("use_odom_tf", useOdomTF_);
        this->get_parameter("use_odom_tf", useOdomTF_);

        // Particle filter parameters
        this->declare_parameter<double>("initial_pose_x", initialPoseX_);
        this->get_parameter("initial_pose_x", initialPoseX_);

        this->declare_parameter<double>("initial_pose_y", initialPoseY_);
        this->get_parameter("initial_pose_y", initialPoseY_);

        this->declare_parameter<double>("initial_pose_yaw", initialPoseYaw_);
        this->get_parameter("initial_pose_yaw", initialPoseYaw_);

        this->declare_parameter<double>("initial_noise_x", initialNoiseX_);
        this->get_parameter("initial_noise_x", initialNoiseX_);

        this->declare_parameter<double>("initial_noise_y", initialNoiseY_);
        this->get_parameter("initial_noise_y", initialNoiseY_);

        this->declare_parameter<double>("initial_noise_yaw", initialNoiseYaw_);
        this->get_parameter("initial_noise_yaw", initialNoiseYaw_);

        this->declare_parameter<int>("particle_num", particlesNum_);
        this->get_parameter("particle_num", particlesNum_);

        this->declare_parameter<bool>("use_augmented_mcl", useAugmentedMCL_);
        this->get_parameter("use_augmented_mcl", useAugmentedMCL_);

        this->declare_parameter<bool>("add_random_particles_in_resampling", addRandomParticlesInResampling_);
        this->get_parameter("add_random_particles_in_resampling", addRandomParticlesInResampling_);

        this->declare_parameter<double>("random_particles_rate", randomParticlesRate_);
        this->get_parameter("random_particles_rate", randomParticlesRate_);

        // vector パラメータ（例: std::vector<double>）もサポートされます
        this->declare_parameter<std::vector<double>>("random_particles_noise", randomParticlesNoise_);
        this->get_parameter("random_particles_noise", randomParticlesNoise_);

        // motion
        this->declare_parameter<std::vector<double>>("odom_noise_ddm", odomNoiseDDM_);
        this->get_parameter("odom_noise_ddm", odomNoiseDDM_);

        this->declare_parameter<std::vector<double>>("odom_noise_odm", odomNoiseODM_);
        this->get_parameter("odom_noise_odm", odomNoiseODM_);

        this->declare_parameter<bool>("use_omni_directional_model", useOmniDirectionalModel_);
        this->get_parameter("use_omni_directional_model", useOmniDirectionalModel_);

        // measurement model
        this->declare_parameter<int>("measurement_model_type", measurementModelType_);
        this->get_parameter("measurement_model_type", measurementModelType_);

        this->declare_parameter<int>("scan_step", scanStep_);
        this->get_parameter("scan_step", scanStep_);

        this->declare_parameter<double>("z_hit", zHit_);
        this->get_parameter("z_hit", zHit_);

        this->declare_parameter<double>("z_short", zShort_);
        this->get_parameter("z_short", zShort_);

        this->declare_parameter<double>("z_max", zMax_);
        this->get_parameter("z_max", zMax_);

        this->declare_parameter<double>("z_rand", zRand_);
        this->get_parameter("z_rand", zRand_);

        this->declare_parameter<double>("var_hit", varHit_);
        this->get_parameter("var_hit", varHit_);

        this->declare_parameter<double>("lambda_short", lambdaShort_);
        this->get_parameter("lambda_short", lambdaShort_);

        this->declare_parameter<double>("lambda_unknown", lambdaUnknown_);
        this->get_parameter("lambda_unknown", lambdaUnknown_);

        this->declare_parameter<double>("known_class_prior", pKnownPrior_);
        this->get_parameter("known_class_prior", pKnownPrior_);

        this->declare_parameter<double>("unknown_scan_prob_threshold", unknownScanProbThreshold_);
        this->get_parameter("unknown_scan_prob_threshold", unknownScanProbThreshold_);

        this->declare_parameter<double>("alpha_slow", alphaSlow_);
        this->get_parameter("alpha_slow", alphaSlow_);

        this->declare_parameter<double>("alpha_fast", alphaFast_);
        this->get_parameter("alpha_fast", alphaFast_);

        this->declare_parameter<bool>("reject_unknown_scan", rejectUnknownScan_);
        this->get_parameter("reject_unknown_scan", rejectUnknownScan_);

        this->declare_parameter<bool>("publish_unknown_scan", publishUnknownScan_);
        this->get_parameter("publish_unknown_scan", publishUnknownScan_);

        this->declare_parameter<bool>("publish_residual_errors", publishResidualErrors_);
        this->get_parameter("publish_residual_errors", publishResidualErrors_);

        this->declare_parameter<double>("resample_threshold_ess", resampleThresholdESS_);
        this->get_parameter("resample_threshold_ess", resampleThresholdESS_);

        this->declare_parameter<std::vector<double>>("resample_thresholds", resampleThresholds_);
        this->get_parameter("resample_thresholds", resampleThresholds_);

        pUnknownPrior_ = 1.0 - pKnownPrior_;

        // reliability estimation
        this->declare_parameter<bool>("estimate_reliability", estimateReliability_);
        this->get_parameter("estimate_reliability", estimateReliability_);

        this->declare_parameter<std::vector<double>>("rel_trans_ddm", relTransDDM_);
        this->get_parameter("rel_trans_ddm", relTransDDM_);

        this->declare_parameter<std::vector<double>>("rel_trans_odm", relTransODM_);
        this->get_parameter("rel_trans_odm", relTransODM_);

        // failure detector
        this->declare_parameter<int>("classifier_type", classifierType_);
        this->get_parameter("classifier_type", classifierType_);

        this->declare_parameter<std::string>("mae_classifier_dir", maeClassifierDir_);
        this->get_parameter("mae_classifier_dir", maeClassifierDir_);

        // global-localization-based pose sampling
        this->declare_parameter<bool>("use_gl_pose_sampler", useGLPoseSampler_);
        this->get_parameter("use_gl_pose_sampler", useGLPoseSampler_);

        this->declare_parameter<bool>("fuse_gl_pose_sampler_only_unreliable", fuseGLPoseSamplerOnlyUnreliable_);
        this->get_parameter("fuse_gl_pose_sampler_only_unreliable", fuseGLPoseSamplerOnlyUnreliable_);

        this->declare_parameter<double>("gl_sampled_pose_time_th", glSampledPoseTimeTH_);
        this->get_parameter("gl_sampled_pose_time_th", glSampledPoseTimeTH_);

        this->declare_parameter<double>("gmm_positional_variance", gmmPositionalVariance_);
        this->get_parameter("gmm_positional_variance", gmmPositionalVariance_);

        this->declare_parameter<double>("gmm_angular_variance", gmmAngularVariance_);
        this->get_parameter("gmm_angular_variance", gmmAngularVariance_);

        this->declare_parameter<double>("pred_dist_unif_rate", predDistUnifRate_);
        this->get_parameter("pred_dist_unif_rate", predDistUnifRate_);

        // write pose
        this->declare_parameter<bool>("write_pose", writePose_);
        this->get_parameter("write_pose", writePose_);

        this->declare_parameter<std::string>("pose_log_file", poseLogFile_);
        this->get_parameter("pose_log_file", poseLogFile_);

        // other parameters
        this->declare_parameter<double>("localization_hz", localizationHz_);
        this->get_parameter("localization_hz", localizationHz_);

        this->declare_parameter<double>("transform_tolerance", transformTolerance_);
        this->get_parameter("transform_tolerance", transformTolerance_);

        this->declare_parameter<bool>("start_mcl", startMCL_);
        this->get_parameter("start_mcl", startMCL_);

        RCLCPP_INFO(this->get_logger(), "  scan_name: %s", scanName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  odom_name: %s", odomName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  map_name: %s", mapName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  pose_name: %s", poseName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  particles_name: %s", particlesName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  unknown_scan_name: %s", unknownScanName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  residual_errors_name: %s", residualErrorsName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  reliability_name: %s", reliabilityName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  gl_sampled_poses_name: %s", glSampledPosesName_.c_str());
        RCLCPP_INFO(this->get_logger(), "  laser_frame: %s", laserFrame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  base_link_frame: %s", baseLinkFrame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  map_frame: %s", mapFrame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  odom_frame: %s", odomFrame_.c_str());
        RCLCPP_INFO(this->get_logger(), "  broadcast_tf: %s", broadcastTF_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  use_odom_tf: %s", useOdomTF_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "Initial pose and particle filter parameters:");
        RCLCPP_INFO(this->get_logger(), "  initial_pose_x: %.3f", initialPoseX_);
        RCLCPP_INFO(this->get_logger(), "  initial_pose_y: %.3f", initialPoseY_);
        RCLCPP_INFO(this->get_logger(), "  initial_pose_yaw: %.3f", initialPoseYaw_);
        RCLCPP_INFO(this->get_logger(), "  initial_noise_x: %.3f", initialNoiseX_);
        RCLCPP_INFO(this->get_logger(), "  initial_noise_y: %.3f", initialNoiseY_);
        RCLCPP_INFO(this->get_logger(), "  initial_noise_yaw: %.3f", initialNoiseYaw_);
        RCLCPP_INFO(this->get_logger(), "  particle_num: %d", particlesNum_);
        RCLCPP_INFO(this->get_logger(), "  use_augmented_mcl: %s", useAugmentedMCL_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  add_random_particles_in_resampling: %s", addRandomParticlesInResampling_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  random_particles_rate: %.3f", randomParticlesRate_);
        RCLCPP_INFO(this->get_logger(), "  random_particles_noise:");
        for (size_t i = 0; i < randomParticlesNoise_.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "    [%zu]: %.3f", i, randomParticlesNoise_[i]);
        }
        RCLCPP_INFO(this->get_logger(), "  odom_noise_ddm:");
        for (size_t i = 0; i < odomNoiseDDM_.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "    [%zu]: %.3f", i, odomNoiseDDM_[i]);
        }
        RCLCPP_INFO(this->get_logger(), "  odom_noise_odm:");
        for (size_t i = 0; i < odomNoiseODM_.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "    [%zu]: %.3f", i, odomNoiseODM_[i]);
        }
        RCLCPP_INFO(this->get_logger(), "  use_omni_directional_model: %s", useOmniDirectionalModel_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  measurement_model_type: %d", measurementModelType_);
        RCLCPP_INFO(this->get_logger(), "  scan_step: %d", scanStep_);
        RCLCPP_INFO(this->get_logger(), "  z_hit: %.3f", zHit_);
        RCLCPP_INFO(this->get_logger(), "  z_short: %.3f", zShort_);
        RCLCPP_INFO(this->get_logger(), "  z_max: %.3f", zMax_);
        RCLCPP_INFO(this->get_logger(), "  z_rand: %.3f", zRand_);
        RCLCPP_INFO(this->get_logger(), "  var_hit: %.3f", varHit_);
        RCLCPP_INFO(this->get_logger(), "  lambda_short: %.3f", lambdaShort_);
        RCLCPP_INFO(this->get_logger(), "  lambda_unknown: %.3f", lambdaUnknown_);
        RCLCPP_INFO(this->get_logger(), "  known_class_prior: %.3f", pKnownPrior_);
        RCLCPP_INFO(this->get_logger(), "  unknown_scan_prob_threshold: %.3f", unknownScanProbThreshold_);
        RCLCPP_INFO(this->get_logger(), "  alpha_slow: %.3f", alphaSlow_);
        RCLCPP_INFO(this->get_logger(), "  alpha_fast: %.3f", alphaFast_);
        RCLCPP_INFO(this->get_logger(), "  reject_unknown_scan: %s", rejectUnknownScan_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  publish_unknown_scan: %s", publishUnknownScan_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  publish_residual_errors: %s", publishResidualErrors_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  resample_threshold_ess: %.3f", resampleThresholdESS_);
        RCLCPP_INFO(this->get_logger(), "  resample_thresholds:");
        for (size_t i = 0; i < resampleThresholds_.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "    [%zu]: %.3f", i, resampleThresholds_[i]);
        }
        RCLCPP_INFO(this->get_logger(), "  estimate_reliability: %s", estimateReliability_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  rel_trans_ddm:");
        for (size_t i = 0; i < relTransDDM_.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "    [%zu]: %.3f", i, relTransDDM_[i]);
        }
        RCLCPP_INFO(this->get_logger(), "  rel_trans_odm:");
        for (size_t i = 0; i < relTransODM_.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "    [%zu]: %.3f", i, relTransODM_[i]);
        }
        RCLCPP_INFO(this->get_logger(), "  classifier_type: %d", classifierType_);
        RCLCPP_INFO(this->get_logger(), "  mae_classifier_dir: %s", maeClassifierDir_.c_str());
        RCLCPP_INFO(this->get_logger(), "  use_gl_pose_sampler: %s", useGLPoseSampler_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  fuse_gl_pose_sampler_only_unreliable: %s", fuseGLPoseSamplerOnlyUnreliable_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  gl_sampled_pose_time_th: %.3f", glSampledPoseTimeTH_);
        RCLCPP_INFO(this->get_logger(), "  gmm_positional_variance: %.3f", gmmPositionalVariance_);
        RCLCPP_INFO(this->get_logger(), "  gmm_angular_variance: %.3f", gmmAngularVariance_);
        RCLCPP_INFO(this->get_logger(), "  pred_dist_unif_rate: %.3f", predDistUnifRate_);
        RCLCPP_INFO(this->get_logger(), "  write_pose: %s", writePose_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "  pose_log_file: %s", poseLogFile_.c_str());
        RCLCPP_INFO(this->get_logger(), "  localization_hz: %.3f", localizationHz_);
        RCLCPP_INFO(this->get_logger(), "  transform_tolerance: %.3f", transformTolerance_);
        RCLCPP_INFO(this->get_logger(), "  start_mcl: %s", startMCL_ ? "true" : "false");

        // set tf broadcaster
        tfBroadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        tfBuffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tfBuffer_->setUsingDedicatedThread(true);
        tfListener_ = std::make_shared<tf2_ros::TransformListener>(*tfBuffer_, this, false);

        // set subscribers
        scanSub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            scanName_, 10, std::bind(&MCL::scanCB, this, std::placeholders::_1));
        odomSub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odomName_, 100, std::bind(&MCL::odomCB, this, std::placeholders::_1));
        mapSub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            mapName_, 1, std::bind(&MCL::mapCB, this, std::placeholders::_1));
        initialPoseSub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 1, std::bind(&MCL::initialPoseCB, this, std::placeholders::_1));
        if (useGLPoseSampler_)
            glSampledPosesSub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
                glSampledPosesName_, 1, std::bind(&MCL::glSampledPosesCB, this, std::placeholders::_1));

        // set publishers
        posePub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            poseName_, 1);
        particlesPub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
            particlesName_, 1);
        if (publishUnknownScan_)
            unknownScanPub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
                unknownScanName_, 1);
        if (publishResidualErrors_)
            residualErrorsPub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
                residualErrorsName_, 1);
        if (estimateReliability_) {
            reliabilityPub_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>(
                reliabilityName_, 1);
            reliabilityMarkerPub_ = this->create_publisher<visualization_msgs::msg::Marker>(
                reliabilityMarkerName_, 1);
        }

        // degree to radian
        initialPoseYaw_ *= M_PI / 180.0;

        // set initial pose
        mclPose_.setPose(initialPoseX_, initialPoseY_, initialPoseYaw_);
        resetParticlesDistribution();
        odomPose_.setPose(0.0, 0.0, 0.0);
        deltaX_ = deltaY_ = deltaDist_ = deltaYaw_ = 0.0;

        // get the relative pose from the base link to the laser from the tf tree
        int cnt = 0;
        rclcpp::Rate rate(10);
        while (rclcpp::ok()) {
            rclcpp::spin_some(this->get_node_base_interface());
            try {
                geometry_msgs::msg::TransformStamped tfStamped =
                    tfBuffer_->lookupTransform(
                        baseLinkFrame_, laserFrame_, tf2::TimePointZero,
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(2)));
                tf2::Quaternion quat;
                tf2::fromMsg(tfStamped.transform.rotation, quat);
                double roll, pitch, yaw;
                tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
                baseLink2Laser_.setX(tfStamped.transform.translation.x);
                baseLink2Laser_.setY(tfStamped.transform.translation.y);
                baseLink2Laser_.setYaw(yaw);
                break;
            } catch (tf2::TransformException &ex) {
                cnt++;
                if (cnt >= 300) {
                    RCLCPP_ERROR(this->get_logger(), "Cannot get the relative pose from the base link to the laser from the tf tree."
                        " Did you set the static transform publisher between %s to %s?",
                        baseLinkFrame_.c_str(), laserFrame_.c_str());
                    rclcpp::shutdown();
                    return;
                }
            }
            rate.sleep();
        }

        // check map
        cnt = 0;
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

        // check scan
        cnt = 0;
        while (rclcpp::ok()) {
            rclcpp::spin_some(this->get_node_base_interface());
            if (gotScan_) break;
            cnt++;
            if (cnt >= 300) {
                RCLCPP_ERROR(this->get_logger(), "Cannot get a scan message."
                    " Did you publish the scan?"
                    " Expected scan topic name is %s\n", scanName_.c_str());
                rclcpp::shutdown();
                return;
            }
            rate.sleep();
        }

        // reliability estimation
        if (estimateReliability_) {
            resetReliabilities();

            // failure detector
            if (classifierType_ == 0) {
                maeClassifier_.setClassifierDir(maeClassifierDir_);
                maeClassifier_.readClassifierParams();
                maes_.resize(particlesNum_);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Incorrect classifier type was selected."
                    " The expected type is 0, but %d was selected.", classifierType_);
                rclcpp::shutdown();
                return;
            }
        }

        // fixed parameters for the measurement models
        normConstHit_ = 1.0 / sqrt(2.0 * varHit_ * M_PI);
        denomHit_ = 1.0 / (2.0 * varHit_);
        pRand_ = 1.0 / (scan_.range_max / mapResolution_);
        measurementModelRandom_ = zRand_ * pRand_;
        measurementModelInvalidScan_ = zMax_ + zRand_ * pRand_;

        isInitialized_ = true;
        RCLCPP_INFO(this->get_logger(), "MCL is ready to perform\n");

        // set timer
        if (startMCL_) {
            timer_ = this->create_wall_timer(
                std::chrono::milliseconds((int)(1000.0 / localizationHz_)),
                std::bind(&MCL::timerCB, this));
        }
    }

    void timerCB() {
        updateParticlesByMotionModel();
        setCanUpdateScan(false);
        calculateLikelihoodsByMeasurementModel();
        calculateLikelihoodsByDecisionModel();
        calculateGLSampledPosesLikelihood();
        calculateAMCLRandomParticlesRate();
        calculateEffectiveSampleSize();
        estimatePose();
        resampleParticles();
        // plotScan();
        // plotWorld(50.0);
        publishROSMessages();
        broadcastTF();
        // plotLikelihoodMap();
        setCanUpdateScan(true);
        printResult();
    }

    void updateParticlesByMotionModel(void) {
        double deltaX = deltaX_;
        double deltaY = deltaY_;
        double deltaDist = deltaDist_;
        double deltaYaw = deltaYaw_;
        deltaX_ = deltaY_ = deltaDist_ = deltaYaw_ = 0.0;
        deltaXSum_ += fabs(deltaX);
        deltaYSum_ += fabs(deltaY);
        deltaDistSum_ += fabs(deltaDist);
        deltaYawSum_ += fabs(deltaYaw);

        if (!useOmniDirectionalModel_) {
            // differential drive model
            double yaw = mclPose_.getYaw();
            double t = yaw + deltaYaw / 2.0;
            double x = mclPose_.getX() + deltaDist * cos(t);
            double y = mclPose_.getY() + deltaDist * sin(t);
            yaw += deltaYaw;
            mclPose_.setPose(x, y, yaw);
            double dist2 = deltaDist * deltaDist;
            double yaw2 = deltaYaw * deltaYaw;
            double distRandVal = dist2 * odomNoiseDDM_[0] + yaw2 * odomNoiseDDM_[1];
            double yawRandVal = dist2 * odomNoiseDDM_[2] + yaw2 * odomNoiseDDM_[3];
            for (int i = 0; i < particlesNum_; ++i) {
                double ddist = deltaDist + nrand(distRandVal);
                double dyaw = deltaYaw + nrand(yawRandVal);
                double yaw = particles_[i].getYaw();
                double t = yaw + dyaw / 2.0;
                double x = particles_[i].getX() + ddist * cos(t);
                double y = particles_[i].getY() + ddist * sin(t);
                yaw += dyaw;
                particles_[i].setPose(x, y, yaw);
                if (estimateReliability_) {
                    double decayRate = 1.0 - (relTransDDM_[0] * ddist * ddist + relTransDDM_[1] * dyaw * dyaw);
                    if (decayRate <= 0.0)
                        decayRate = 10.0e-6;
                    reliabilities_[i] *= decayRate;
                }
            }
        } else {
            // omni directional model
            double yaw = mclPose_.getYaw();
            double t = yaw + deltaYaw / 2.0;
            double x = mclPose_.getX() + deltaX * cos(t) + deltaY * cos(t + M_PI / 2.0f);
            double y = mclPose_.getY() + deltaX * sin(t) + deltaY * sin(t + M_PI / 2.0f);;
            yaw += deltaYaw;
            mclPose_.setPose(x, y, yaw);
            double x2 = deltaX * deltaX;
            double y2 = deltaY * deltaY;
            double yaw2 = deltaYaw * deltaYaw;
            double xRandVal = x2 * odomNoiseODM_[0] + y2 * odomNoiseODM_[1] + yaw2 * odomNoiseODM_[2];
            double yRandVal = x2 * odomNoiseODM_[3] + y2 * odomNoiseODM_[4] + yaw2 * odomNoiseODM_[5];
            double yawRandVal = x2 * odomNoiseODM_[6] + y2 * odomNoiseODM_[7] + yaw2 * odomNoiseODM_[8];
            for (int i = 0; i < particlesNum_; ++i) {
                double dx = deltaX + nrand(xRandVal);
                double dy = deltaY + nrand(yRandVal);
                double dyaw = deltaYaw + nrand(yawRandVal);
                double yaw = particles_[i].getYaw();
                double t = yaw + dyaw / 2.0;
                double x = particles_[i].getX() + dx * cos(t) + dy * cos(t + M_PI / 2.0f);
                double y = particles_[i].getY() + dx * sin(t) + dy * sin(t + M_PI / 2.0f);;
                yaw += dyaw;
                particles_[i].setPose(x, y, yaw);
                if (estimateReliability_) {
                    double decayRate = 1.0 - (relTransODM_[0] * dx * dx + relTransODM_[1] * dy * dy + relTransODM_[2] * dyaw * dyaw);
                    if (decayRate <= 0.0)
                        decayRate = 10.0e-6;
                    reliabilities_[i] *= decayRate;
                }
            }
        }
    }

    void calculateLikelihoodsByMeasurementModel(void) {
        if (scanMightInvalid_)
            return;

        if (rejectUnknownScan_ && (measurementModelType_ == 0 || measurementModelType_ == 1))
            rejectUnknownScan();

        mclPoseStamp_ = scan_.header.stamp;
        double xo = baseLink2Laser_.getX();
        double yo = baseLink2Laser_.getY();
        double yawo = baseLink2Laser_.getYaw();
        std::vector<Pose> sensorPoses(particlesNum_);
        for (int i = 0; i < particlesNum_; ++i) {
            double yaw = particles_[i].getYaw();
            double sensorX = xo * cos(yaw) - yo * sin(yaw) + particles_[i].getX();
            double sensorY = xo * sin(yaw) + yo * cos(yaw) + particles_[i].getY();
            double sensorYaw = yawo + yaw;
            Pose sensorPose(sensorX, sensorY, sensorYaw);
            sensorPoses[i] = sensorPose;
            particles_[i].setW(0.0);
        }

        likelihoodShiftedSteps_.clear();
        for (int i = 0; i < (int)scan_.ranges.size(); i += scanStep_) {
            double range = scan_.ranges[i];
            double rangeAngle = (double)i * scan_.angle_increment + scan_.angle_min;
            double max;
            for (int j = 0; j < particlesNum_; ++j) {
                double p;
                if (measurementModelType_ == 0)
                    p = calculateLikelihoodFieldModel(sensorPoses[j], range, rangeAngle);
                else if (measurementModelType_ == 1)
                    p = calculateBeamModel(sensorPoses[j], range, rangeAngle);
                else
                    p = calculateClassConditionalMeasurementModel(sensorPoses[j], range, rangeAngle);
                double w = particles_[j].getW();
                w += log(p);
                particles_[j].setW(w);
                if (j == 0) {
                    max = w;
                } else {
                    if (max < w)
                        max = w;
                }
            }

            // Too small values cannot be calculated.
            // The log sum values are shifted if the maximum value is less than threshold.
            if (max < -300.0) {
                for (int j = 0; j < particlesNum_; ++j) {
                    double w = particles_[j].getW() + 300.0;
                    particles_[j].setW(w);
                }
                likelihoodShiftedSteps_.push_back(true);
            } else {
                likelihoodShiftedSteps_.push_back(false);
            }
        }

        double sum = 0.0;
        double max;
        int maxIdx;
        for (int i = 0; i < particlesNum_; ++i) {
            // The log sum is converted to the probability.
            double w = exp(particles_[i].getW());
            particles_[i].setW(w);
            sum += w;
            if (i == 0) {
                max = w;
                maxIdx = i;
            } else if (max < w) {
                max = w;
                maxIdx = i;
            }
        }
        totalLikelihood_ = sum;
        averageLikelihood_ = sum / (double)particlesNum_;
        maxLikelihood_ = max;
        maxLikelihoodParticleIdx_ = maxIdx;
    }

    void calculateLikelihoodsByDecisionModel(void) {
        if (scanMightInvalid_)
            return;

        if (!estimateReliability_)
            return;

        if (measurementModelType_ == 2 && classifierType_ == 0) {
            Pose mlPose = particles_[maxLikelihoodParticleIdx_].getPose();
            estimateUnknownScanWithClassConditionalMeasurementModel(mlPose);
        }

        double sum = 0.0;
        double max;
        int maxIdx;
        for (int i = 0; i < particlesNum_; ++i) {
            Pose particlePose = particles_[i].getPose();
            double measurementLikelihood = particles_[i].getW();
            std::vector<double> residualErrors = getResidualErrors<double>(particlePose);
            double decisionLikelihood;
            if (classifierType_ == 0) {
                double mae = maeClassifier_.getMAE(residualErrors);
                decisionLikelihood = maeClassifier_.calculateDecisionModel(mae, &reliabilities_[i]);
                maes_[i] = mae;
            }
            double w = measurementLikelihood * decisionLikelihood;
            particles_[i].setW(w);
            sum += w;
            if (i == 0) {
                max = w;
                maxIdx = 0;
            } else if (max < w) {
                max = w;
                maxIdx = i;
            }
        }
        totalLikelihood_ = sum;
        averageLikelihood_ = sum / (double)particlesNum_;
        maxLikelihood_ = max;
        maxLikelihoodParticleIdx_ = maxIdx;
        reliability_ = reliabilities_[maxIdx];
    }

    void calculateGLSampledPosesLikelihood(void) {
        if (!useGLPoseSampler_)
            return;
        if (scanMightInvalid_)
            return;
        if (estimateReliability_ && fuseGLPoseSamplerOnlyUnreliable_) {
            if (reliability_ >= 0.9)
                return;
        }

        glParticlesNum_ = (int)glSampledPoses_.poses.size();
        double dt = fabs(mclPoseStamp_.seconds() - glSampledPosesStamp_.seconds());
        if (dt > glSampledPoseTimeTH_ || glParticlesNum_ == 0 || !isGLSampledPosesUpdated_)
            return;

        canUpdateGLSampledPoses_ = isGLSampledPosesUpdated_ = false;
        double xo = baseLink2Laser_.getX();
        double yo = baseLink2Laser_.getY();
        double yawo = baseLink2Laser_.getYaw();
        std::vector<Pose> sensorPoses(glParticlesNum_);
        glParticles_.resize(glParticlesNum_);
        if (estimateReliability_) {
            glSampledPosesReliabilities_.resize(glParticlesNum_);
            if (classifierType_ == 0)
                glSampledPosesMAEs_.resize(glParticlesNum_);
        }

        for (int i = 0; i < glParticlesNum_; ++i) {
            tf2::Quaternion q(glSampledPoses_.poses[i].orientation.x, 
                glSampledPoses_.poses[i].orientation.y, 
                glSampledPoses_.poses[i].orientation.z,
                glSampledPoses_.poses[i].orientation.w);
            double roll, pitch, yaw;
            tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
            glParticles_[i].setPose(glSampledPoses_.poses[i].position.x, glSampledPoses_.poses[i].position.y, yaw);

            double sensorX = xo * cos(yaw) - yo * sin(yaw) + glParticles_[i].getX();
            double sensorY = xo * sin(yaw) + yo * cos(yaw) + glParticles_[i].getY();
            double sensorYaw = yawo + yaw;
            Pose sensorPose(sensorX, sensorY, sensorYaw);
            sensorPoses[i] = sensorPose;
            glParticles_[i].setW(0.0);
        }
        canUpdateGLSampledPoses_ = true;

        int idx = 0;
        for (int i = 0; i < (int)scan_.ranges.size(); i += scanStep_) {
            double range = scan_.ranges[i];
            double rangeAngle = (double)i * scan_.angle_increment + scan_.angle_min;
            double max;
            for (int j = 0; j < glParticlesNum_; ++j) {
                double p;
                if (measurementModelType_ == 0)
                    p = calculateLikelihoodFieldModel(sensorPoses[j], range, rangeAngle);
                else if (measurementModelType_ == 1)
                    p = calculateBeamModel(sensorPoses[j], range, rangeAngle);
                else
                    p = calculateClassConditionalMeasurementModel(sensorPoses[j], range, rangeAngle);
                double w = glParticles_[j].getW();
                w += log(p);
                glParticles_[j].setW(w);
            }

            if (likelihoodShiftedSteps_[idx]) {
                for (int j = 0; j < glParticlesNum_; ++j) {
                    double w = glParticles_[j].getW() + 300.0;
                    glParticles_[j].setW(w);
                }
            }
            idx++;
        }

        double normConst = 1.0 / sqrt(2.0 * M_PI * (gmmPositionalVariance_ + gmmPositionalVariance_ + gmmAngularVariance_));
        double angleResolution = 1.0 * M_PI / 180.0;
        double sum = totalLikelihood_;
        double max = maxLikelihood_;
        double gmmRate = 1.0 - predDistUnifRate_;
        int maxIdx = -1;
        for (int i = 0; i < glParticlesNum_; ++i) {
            // measurement likelihood
            double w = exp(glParticles_[i].getW());

            // decision likelihood
            if (estimateReliability_) {
                std::vector<double> residualErrors = getResidualErrors<double>(glParticles_[i].getPose());
                double decisionLikelihood;
                glSampledPosesReliabilities_[i] = 0.5;
                if (classifierType_ == 0) {
                    double mae = maeClassifier_.getMAE(residualErrors);
                    decisionLikelihood = maeClassifier_.calculateDecisionModel(mae, &glSampledPosesReliabilities_[i]);
                    glSampledPosesMAEs_[i] = mae;
                }
                w *= decisionLikelihood;
            }

            // predictive distribution likelihood
            double gmmVal = 0.0;
            for (int j = 0; j < particlesNum_; ++j) {
                double dx = glParticles_[i].getX() - particles_[j].getX();
                double dy = glParticles_[i].getY() - particles_[j].getY();
                double dyaw = glParticles_[i].getYaw() - particles_[j].getYaw();
                while (dyaw < -M_PI)
                    dyaw += 2.0 * M_PI;
                while (dyaw > M_PI)
                    dyaw -= 2.0 * M_PI;
                gmmVal += normConst * exp(-((dx * dx) / (2.0 * gmmPositionalVariance_) + (dy * dy) / (2.0 * gmmPositionalVariance_) + (dyaw * dyaw) / (2.0 * gmmAngularVariance_)));
            }
            double pGMM = (double)glParticlesNum_ * gmmVal * mapResolution_ * mapResolution_ * angleResolution / (double)particlesNum_;
            double predLikelihood = gmmRate * pGMM + predDistUnifRate_ * 10e-9;
            w *= predLikelihood;
            if (w > 1.0)
                w = 1.0;

            glParticles_[i].setW(w);
            sum += w;
            if (max < w) {
                max = w;
                maxIdx = i;
            }
        }

        if (std::isnan(sum)) {
            for (int i = 0; i < glParticlesNum_; ++i)
                glParticles_[i].setW(0.0);
            canUseGLSampledPoses_ = false;
            return;
        }

        totalLikelihood_ = sum;
        averageLikelihood_ = sum / (double)(particlesNum_ + glParticlesNum_);
        if (maxIdx >= 0) {
            maxLikelihood_ = max;
            maxLikelihoodParticleIdx_ = particlesNum_ + maxIdx;
            if (estimateReliability_)
                reliability_ = glSampledPosesReliabilities_[maxIdx];
        }
        canUseGLSampledPoses_ = true;
    }

    void calculateAMCLRandomParticlesRate(void) {
        if (!useAugmentedMCL_)
            return;
        if (scanMightInvalid_)
            return;
        omegaSlow_ += alphaSlow_ * (averageLikelihood_ - omegaSlow_);
        omegaFast_ += alphaFast_ * (averageLikelihood_ - omegaFast_);
        amclRandomParticlesRate_ = 1.0 - omegaFast_ / omegaSlow_;
        if (amclRandomParticlesRate_ < 0.0)
            amclRandomParticlesRate_ = 0.0;
    }

    void calculateEffectiveSampleSize(void) {
        if (scanMightInvalid_)
            return;

        double sum = 0.0;
        if (!useGLPoseSampler_ || !canUseGLSampledPoses_) {
            double wo = 1.0 / (double)particlesNum_;
            for (int i = 0; i < particlesNum_; ++i) {
                double w = particles_[i].getW() / totalLikelihood_;
                particles_[i].setW(w);
                sum += w * w;
            }
        } else {
            double wo = 1.0 / (double)(particlesNum_ + glParticlesNum_);
            for (int i = 0; i < particlesNum_; ++i) {
                double w = particles_[i].getW() / totalLikelihood_;
                particles_[i].setW(w);
                sum += w * w;
            }
            for (int i = 0; i < glParticlesNum_; ++i) {
                double w = glParticles_[i].getW() / totalLikelihood_;
                glParticles_[i].setW(w);
                sum += w * w;
            }
        }
        effectiveSampleSize_ = 1.0 / sum;
    }

    void estimatePose(void) {
        static FILE *fp;
        if (fp == NULL && writePose_) {
            fp = fopen(poseLogFile_.c_str(), "w");
            if (fp == NULL) {
                fprintf(stderr, "Cannot open a pose log file -> %s\n", poseLogFile_.c_str());
                writePose_ = false;
            }
        }

        if (scanMightInvalid_)
            return;

        double tmpYaw = mclPose_.getYaw();
        double x = 0.0, y = 0.0, yaw = 0.0;
        double sum = 0.0;
        for (int i = 0; i < particlesNum_; ++i) {
            double w = particles_[i].getW();
            x += particles_[i].getX() * w;
            y += particles_[i].getY() * w;
            double dyaw = tmpYaw - particles_[i].getYaw();
            while (dyaw < -M_PI)
                dyaw += 2.0 * M_PI;
            while (dyaw > M_PI)
                dyaw -= 2.0 * M_PI;
            yaw += dyaw * w;
            sum += w;
        }

        if (useGLPoseSampler_ && canUseGLSampledPoses_) {
            double x2 = x, y2 = y, yaw2 = yaw;
            for (int i = 0; i < glParticlesNum_; ++i) {
                double w = glParticles_[i].getW();
                x += glParticles_[i].getX() * w;
                y += glParticles_[i].getY() * w;
                double dyaw = tmpYaw - glParticles_[i].getYaw();
                while (dyaw < -M_PI)
                    dyaw += 2.0 * M_PI;
                while (dyaw > M_PI)
                    dyaw -= 2.0 * M_PI;
                yaw += dyaw * w;
                sum += w;
            }
            if (sum > 1.0)
                x = x2, y = y2, yaw = yaw2;
        }

        yaw = tmpYaw - yaw;
        mclPose_.setPose(x, y, yaw);

        if (writePose_)
            fprintf(fp, "%lf %lf %lf %lf\n", mclPoseStamp_.seconds(), x, y, yaw);
    }

    void resampleParticles(void) {
        if (scanMightInvalid_)
            return;

        double threshold = (double)particlesNum_ * resampleThresholdESS_;
        if (useGLPoseSampler_ && canUseGLSampledPoses_)
            threshold = (double)(particlesNum_ + glParticlesNum_) * resampleThresholdESS_;
        if (effectiveSampleSize_ > threshold)
            return;

        if (deltaXSum_ < resampleThresholds_[0] && deltaYSum_ < resampleThresholds_[1] &&
            deltaDistSum_ < resampleThresholds_[2] && deltaYawSum_ < resampleThresholds_[3] &&
            deltaTimeSum_ < resampleThresholds_[4])
            return;

        deltaXSum_ = deltaYSum_ = deltaDistSum_ = deltaYSum_ = deltaTimeSum_ = 0.0;
        std::vector<double> wBuffer;
        if (useGLPoseSampler_ && canUseGLSampledPoses_) {
            wBuffer.resize(particlesNum_ + glParticlesNum_);
            wBuffer[0] = particles_[0].getW();
            for (int i = 1; i < particlesNum_; ++i)
                wBuffer[i] = particles_[i].getW() + wBuffer[i - 1];
            for (int i = 0; i < glParticlesNum_; ++i)
                wBuffer[particlesNum_ + i] = glParticles_[i].getW() + wBuffer[particlesNum_ + i - 1];
        } else {
            wBuffer.resize(particlesNum_);
            wBuffer[0] = particles_[0].getW();
            for (int i = 1; i < particlesNum_; ++i)
                wBuffer[i] = particles_[i].getW() + wBuffer[i - 1];
        }

        std::vector<Particle> tmpParticles = particles_;
        std::vector<double>  tmpReliabilities;
        if (estimateReliability_)
            tmpReliabilities = reliabilities_;
        double wo = 1.0 / (double)particlesNum_;

        if (!addRandomParticlesInResampling_ && !useAugmentedMCL_) {
            // normal resampling
            for (int i = 0; i < particlesNum_; ++i) {
                double darts = (double)rand() / ((double)RAND_MAX + 1.0);
                bool isResampled = false;
                for (int j = 0; j < particlesNum_; ++j) {
                    if (darts < wBuffer[j]) {
                        particles_[i].setPose(tmpParticles[j].getPose());
                        if (estimateReliability_)
                            reliabilities_[i] = tmpReliabilities[j];
                        particles_[i].setW(wo);
                        isResampled = true;
                        break;
                    }
                }
                if (!isResampled && useGLPoseSampler_ && canUseGLSampledPoses_) {
                    for (int j = 0; j < glParticlesNum_; ++j) {
                        if (darts < wBuffer[particlesNum_ + j]) {
                            particles_[i].setPose(glParticles_[j].getPose());
                            if (estimateReliability_)
                                reliabilities_[i] = glSampledPosesReliabilities_[j];
                            particles_[i].setW(wo);
                            break;
                        }
                    }
                }
            }
        } else {
            // resampling and add random particles
            double randomParticlesRate = randomParticlesRate_;
            if (useAugmentedMCL_ && amclRandomParticlesRate_ > 0.0) {
                omegaSlow_ = omegaFast_ = 0.0;
                randomParticlesRate = amclRandomParticlesRate_;
            } else if (!addRandomParticlesInResampling_) {
                randomParticlesRate = 0.0;
            }
            int resampledParticlesNum = (int)((1.0 - randomParticlesRate) * (double)particlesNum_);
            int randomParticlesNum = particlesNum_ - resampledParticlesNum;
            for (int i = 0; i < resampledParticlesNum; ++i) {
                double darts = (double)rand() / ((double)RAND_MAX + 1.0);
                bool isResampled = false;
                for (int j = 0; j < particlesNum_; ++j) {
                    if (darts < wBuffer[j]) {
                        particles_[i].setPose(tmpParticles[j].getPose());
                        if (estimateReliability_)
                            reliabilities_[i] = tmpReliabilities[j];
                        particles_[i].setW(wo);
                        isResampled = true;
                        break;
                    }
                }
                if (!isResampled && useGLPoseSampler_ &&  canUseGLSampledPoses_) {
                    for (int j = 0; j < glParticlesNum_; ++j) {
                        if (darts < wBuffer[particlesNum_ + j]) {
                            particles_[i].setPose(glParticles_[j].getPose());
                            if (estimateReliability_)
                                reliabilities_[i] = glSampledPosesReliabilities_[j];
                            particles_[i].setW(wo);
                            break;
                        }
                    }
                }
            }

            double xo = mclPose_.getX();
            double yo = mclPose_.getY();
            double yawo = mclPose_.getYaw();
            for (int i = resampledParticlesNum; i < resampledParticlesNum + randomParticlesNum; ++i) {
                double x = xo + nrand(randomParticlesNoise_[0]);
                double y = yo + nrand(randomParticlesNoise_[1]);
                double yaw = yawo + nrand(randomParticlesNoise_[2]);
                particles_[i].setPose(x, y, yaw);
                particles_[i].setW(wo);
                if (estimateReliability_)
                    reliabilities_[i] = reliability_;
            }
        }
        canUseGLSampledPoses_ = false;
    }

    template <typename T>
    std::vector<T> getResidualErrors(Pose pose) {
        double yaw = pose.getYaw();
        double sensorX = baseLink2Laser_.getX() * cos(yaw) - baseLink2Laser_.getY() * sin(yaw) + pose.getX();
        double sensorY = baseLink2Laser_.getX() * sin(yaw) + baseLink2Laser_.getY() * cos(yaw) + pose.getY();
        double sensorYaw = baseLink2Laser_.getYaw() + yaw;
        int size = (int)scan_.ranges.size();
        std::vector<T> residualErrors(size);
        for (int i = 0; i < size; ++i) {
            double r = scan_.ranges[i];
            if (r <= scan_.range_min || scan_.range_max <= r) {
                residualErrors[i] = -1.0;
                continue;
            }
            double t = (double)i * scan_.angle_increment + scan_.angle_min + sensorYaw;
            double x = r * cos(t) + sensorX;
            double y = r * sin(t) + sensorY;
            int u, v;
            xy2uv(x, y, &u, &v);
            if (onMap(u, v)) {
                T dist = (T)distMap_.at<float>(v, u);
                residualErrors[i] = dist;
            } else {
                residualErrors[i] = -1.0;
            }
        }
        return residualErrors;
    }

    void plotScan(void) {
        static bool isFirst = true;
        FILE *fp;
        if (isFirst) {
            fp = fopen("/tmp/ascan.txt", "w");
            fclose(fp);
            isFirst = false;
        }
        fp = fopen("/tmp/ascan.txt", "a");
        double yaw = mclPose_.getYaw();
        double sensorX = baseLink2Laser_.getX() * cos(yaw) - baseLink2Laser_.getY() * sin(yaw) + mclPose_.getX();
        double sensorY = baseLink2Laser_.getX() * sin(yaw) + baseLink2Laser_.getY() * cos(yaw) + mclPose_.getY();
        double sensorYaw = baseLink2Laser_.getYaw() + yaw;
        int size = (int)scan_.ranges.size();
        for (int i = 0; i < size; ++i) {
            double r = scan_.ranges[i];
            if (r <= scan_.range_min || scan_.range_max <= r)
                continue;
            double t = (double)i * scan_.angle_increment + scan_.angle_min + sensorYaw;
            double x = r * cos(t) + sensorX;
            double y = r * sin(t) + sensorY;
            fprintf(fp, "%f %f\n", x, y);
        }
        fclose(fp);
    }

    void printResult(void) {
        std::cout << "MCL: x = " << mclPose_.getX() << " [m], y = " << mclPose_.getY() << " [m], yaw = " << mclPose_.getYaw() * rad2deg_ << " [deg]" << std::endl;
        std::cout << "Odom: x = " << odomPose_.getX() << " [m], y = " << odomPose_.getY() << " [m], yaw = " << odomPose_.getYaw() * rad2deg_ << " [deg]" << std::endl;
        std::cout << "total likelihood = " << totalLikelihood_ << std::endl;
        std::cout << "average likelihood = " << averageLikelihood_ << std::endl;
        std::cout << "max likelihood = " << maxLikelihood_ << std::endl;
        std::cout << "effective sample size = " << effectiveSampleSize_ << std::endl;
        if (useAugmentedMCL_)
            std::cout << "amcl random particles rate = " << amclRandomParticlesRate_ << std::endl;
        if (estimateReliability_ && classifierType_ == 0)
            std::cout << "reliability = " << reliability_ << " (mae = " << maeClassifier_.getMAE(getResidualErrors<double>(particles_[maxLikelihoodParticleIdx_].getPose())) << " [m], th = " << maeClassifier_.getFailureThreshold() << " [m])" << std::endl;
        std::cout << std::endl;
    }

    void publishROSMessages(void) {
        // pose
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = mapFrame_;
        pose.header.stamp = mclPoseStamp_;
        pose.pose.position.x = mclPose_.getX();
        pose.pose.position.y = mclPose_.getY();
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, mclPose_.getYaw());
        pose.pose.orientation = tf2::toMsg(q);
        posePub_->publish(pose);

        // particles
        geometry_msgs::msg::PoseArray particlesPoses;
        particlesPoses.header.frame_id = mapFrame_;
        particlesPoses.header.stamp = mclPoseStamp_;
        particlesPoses.poses.resize(particlesNum_);
        for (int i = 0; i < particlesNum_; ++i) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = particles_[i].getX();
            pose.position.y = particles_[i].getY();
            tf2::Quaternion q;
            q.setRPY(0.0, 0.0, particles_[i].getYaw());
            pose.orientation = tf2::toMsg(q);
            particlesPoses.poses[i] = pose;
        }
        particlesPub_->publish(particlesPoses);

        // unknown scan
        if (publishUnknownScan_ && (rejectUnknownScan_ || measurementModelType_ == 2)) {
            if (!estimateReliability_ && classifierType_ != 0 && measurementModelType_ == 2) {
                Pose mlPose = particles_[maxLikelihoodParticleIdx_].getPose();
                estimateUnknownScanWithClassConditionalMeasurementModel(mlPose);
            }
            unknownScanPub_->publish(unknownScan_);
        }

        // residual errors
        if (publishResidualErrors_) {
            sensor_msgs::msg::LaserScan residualErrors = scan_;
            residualErrors.intensities = getResidualErrors<float>(mclPose_);
            residualErrorsPub_->publish(residualErrors);
        }

        // reliability
        if (estimateReliability_) {
            geometry_msgs::msg::Vector3Stamped reliability;
            reliability.header.stamp = mclPoseStamp_;
            reliability.vector.x = reliability_;
            if (classifierType_ == 0) {
                double mae;
                if (maxLikelihoodParticleIdx_ < particlesNum_)
                    mae = maes_[maxLikelihoodParticleIdx_];
                else
                    mae = glSampledPosesMAEs_[maxLikelihoodParticleIdx_ - particlesNum_];
                reliability.vector.y = mae;
                reliability.vector.z = maeClassifier_.getFailureThreshold();
            }
            reliabilityPub_->publish(reliability);

            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = mapFrame_;
            marker.header.stamp = mclPoseStamp_;
            marker.ns = "reliability_marker_namespace";
            marker.id = 0;
            marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = mclPose_.getX();
            marker.pose.position.y = mclPose_.getY() - 3.0;
            marker.pose.position.z = 0.0;
            marker.scale.x = 0.0;
            marker.scale.y = 0.0;
            marker.scale.z = 2.0;
            marker.text = "Reliability: " + std::to_string((int)(reliability_ * 100.0)) + " %";
            marker.color.a = 1.0;
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 1.0;
            if (reliability_ < 0.9)
                marker.color.r = marker.color.g = 0.0;
            reliabilityMarkerPub_->publish(marker);
        }
    }

    void broadcastTF(void) {
        if (!broadcastTF_)
            return;

        geometry_msgs::msg::Pose poseOnMap;
        poseOnMap.position.x = mclPose_.getX();
        poseOnMap.position.y = mclPose_.getY();
        poseOnMap.position.z = 0.0;
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, mclPose_.getYaw());
        poseOnMap.orientation = tf2::toMsg(q);
        tf2::Transform map2baseTrans;
        tf2::convert(poseOnMap, map2baseTrans);

        rclcpp::Time transformExpiration = this->now() + rclcpp::Duration::from_seconds(transformTolerance_);

        if (useOdomTF_) {
            geometry_msgs::msg::Pose poseOnOdom;
            poseOnOdom.position.x = odomPose_.getX();
            poseOnOdom.position.y = odomPose_.getY();
            poseOnOdom.position.z = 0.0;
            tf2::Quaternion q;
            q.setRPY(0.0, 0.0, odomPose_.getYaw());
            poseOnOdom.orientation = tf2::toMsg(q);
            tf2::Transform odom2baseTrans;
            tf2::convert(poseOnOdom, odom2baseTrans);

            tf2::Transform map2odomTrans = map2baseTrans * odom2baseTrans.inverse();
            // add transform_tolerance: send a transform that is good up until a tolerance time so that odom can be used
            // ros::Time transformExpiration = (mclPoseStamp_ + ros::Duration(transformTolerance_));
            geometry_msgs::msg::TransformStamped map2odomStampedTrans;
            map2odomStampedTrans.header.stamp = transformExpiration;
            map2odomStampedTrans.header.frame_id = mapFrame_;
            map2odomStampedTrans.child_frame_id = odomFrame_;
            tf2::convert(map2odomTrans, map2odomStampedTrans.transform);
            tfBroadcaster_->sendTransform(map2odomStampedTrans);
        } else {
            // ros::Time transformExpiration = (mclPoseStamp_ + ros::Duration(transformTolerance_));
            geometry_msgs::msg::TransformStamped map2baseStampedTrans;
            map2baseStampedTrans.header.stamp = transformExpiration;
            map2baseStampedTrans.header.frame_id = mapFrame_;
            map2baseStampedTrans.child_frame_id = baseLinkFrame_;
            tf2::convert(map2baseTrans, map2baseStampedTrans.transform);
            tfBroadcaster_->sendTransform(map2baseStampedTrans);
        }
    }

    void plotLikelihoodMap(void) {
        double yaw = mclPose_.getYaw();
        double sensorX = baseLink2Laser_.getX() * cos(yaw) - baseLink2Laser_.getY() * sin(yaw) + mclPose_.getX();
        double sensorY = baseLink2Laser_.getX() * sin(yaw) + baseLink2Laser_.getY() * cos(yaw) + mclPose_.getY();
        double sensorYaw = baseLink2Laser_.getYaw() + yaw;
        double range = 0.5;

        std::vector<Pose> sensorPoses;
        for (double x = -range - mapResolution_; x <= range + mapResolution_; x += mapResolution_) {
            for (double y = -range - mapResolution_; y <= range + mapResolution_; y += mapResolution_) {
                Pose sensorPose(sensorX + x, sensorY + y, sensorYaw);
                sensorPoses.push_back(sensorPose);
            }
        }

        std::vector<double> likelihoods((int)sensorPoses.size(), 0.0);
        for (int i = 0; i < (int)scan_.ranges.size(); i += scanStep_) {
            double range = scan_.ranges[i];
            double rangeAngle = (double)i * scan_.angle_increment + scan_.angle_min;
            double max;
            for (int j = 0; j < (int)sensorPoses.size(); ++j) {
                double p;
                if (measurementModelType_ == 0)
                    p = calculateLikelihoodFieldModel(sensorPoses[j], range, rangeAngle);
                else if (measurementModelType_ == 1)
                    p = calculateBeamModel(sensorPoses[j], range, rangeAngle);
                else
                    p = calculateClassConditionalMeasurementModel(sensorPoses[j], range, rangeAngle);
                double w = likelihoods[j];
                w += log(p);
                likelihoods[j] = w;
                if (j == 0) {
                    max = w;
                } else {
                    if (max < w)
                        max = w;
                }
            }
            if (max < -300.0) {
                for (int j = 0; j < (int)sensorPoses.size(); ++j)
                    likelihoods[j] += 300.0;
            }
        }

        double sum = 0.0;
        double max;
        int maxIdx;
        for (int i = 0; i < (int)sensorPoses.size(); ++i) {
            double w = exp(likelihoods[i]);
            if (estimateReliability_) {
                double reliability = 0.5;
                if (classifierType_ == 0) {
                    std::vector<double> residualErrors = getResidualErrors<double>(sensorPoses[i]);
                    double mae = maeClassifier_.getMAE(residualErrors);
                    w *= maeClassifier_.calculateDecisionModel(mae, &reliability);
                }
            }
            likelihoods[i]  = w;
            sum += w;
        }

        FILE *fp;
        fp = fopen("/tmp/als_ros_likelihood_map.txt", "w");
        int cnt = 0;
        for (double x = -range - mapResolution_; x <= range + mapResolution_; x += mapResolution_) {
            for (double y = -range - mapResolution_; y <= range + mapResolution_; y += mapResolution_) {
                fprintf(fp, "%lf %lf %lf\n", x, y, likelihoods[cnt] / sum);
                cnt++;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);

        fp = fopen("/tmp/als_ros_scan_points.txt", "w");
        for (int i = 0; i < (int)scan_.ranges.size(); ++i) {
            double r = scan_.ranges[i];
            if (r < scan_.range_min || scan_.range_max < r)
                continue;
            double t = (double)i * scan_.angle_increment + scan_.angle_min + sensorYaw;
            double x = r * cos(t) + sensorX;
            double y = r * sin(t) + sensorY;
            fprintf(fp, "%lf %lf\n", x, y);
        }
        fclose(fp);

        if (measurementModelType_ == 2 || rejectUnknownScan_) {
            fp = fopen("/tmp/als_ros_unknown_scan_points.txt", "w");
            for (int i = 0; i < (int)unknownScan_.ranges.size(); ++i) {
                double r = unknownScan_.ranges[i];
                if (r < unknownScan_.range_min || unknownScan_.range_max < r)
                    continue;
                double t = (double)i * unknownScan_.angle_increment + unknownScan_.angle_min + sensorYaw;
                double x = r * cos(t) + sensorX;
                double y = r * sin(t) + sensorY;
                fprintf(fp, "%lf %lf\n", x, y);
            }
            fclose(fp);
        }

        static FILE *gp;
        if (gp == NULL) {
            gp = popen("gnuplot -persist", "w");
            fprintf(gp, "set colors classic\n");
            fprintf(gp, "set grid\n");
            fprintf(gp, "set size ratio 1 1\n");
            fprintf(gp, "set xlabel \"%s\"\n", "{/Symbol D}x [m]");
            fprintf(gp, "set ylabel \"%s\"\n", "{/Symbol D}y [m]");
            fprintf(gp, "set tics font \"Arial, 14\"\n");
            fprintf(gp, "set xlabel font \"Arial, 14\"\n");
            fprintf(gp, "set ylabel font \"Arial, 14\"\n");
            fprintf(gp, "set cblabel font \"Arial, 14\"\n");
            fprintf(gp, "set xrange [ %lf : %lf ]\n", -range, range);
            fprintf(gp, "set yrange [ %lf : %lf ]\n", -range, range);
            fprintf(gp, "set pm3d map interpolate 2, 2\n");
            fprintf(gp, "unset key\n");
            fprintf(gp, "unset cbtics\n");
        }
        fprintf(gp, "splot \"/tmp/als_ros_likelihood_map.txt\" with pm3d\n");
        fflush(gp);
    }

    void plotWorld(double plotRange) {
        static FILE *gp;
        FILE *fp;
        if (gp == NULL) {
            gp = popen("gnuplot -persist", "w");
            fprintf(gp, "set colors classic\n");
            fprintf(gp, "unset key\n");
            fprintf(gp, "set grid\n");
            fprintf(gp, "set size ratio 1 1\n");
            fprintf(gp, "set xlabel \"%s\"\n", "X [m]");
            fprintf(gp, "set ylabel \"%s\"\n", "Y [m]");
            fprintf(gp, "set tics font \"Arial, 14\"\n");
            fprintf(gp, "set xlabel font \"Arial, 14\"\n");
            fprintf(gp, "set ylabel font \"Arial, 14\"\n");

            fp = fopen("/tmp/als_ros_map_points.txt", "w");
            for (int u = 0; u < mapWidth_; u++) {
                for (int v = 0; v < mapHeight_; v++) {
                    if (distMap_.at<float>(v, u) == 0.0f) {
                        double x, y;
                        uv2xy(u, v, &x, &y);
                        fprintf(fp, "%lf %lf\n", x, y);
                    }
                }
            }
            fclose(fp);
        }

        fprintf(gp, "set xrange [ %lf : %lf ]\n", mclPose_.getX() - plotRange, mclPose_.getX() + plotRange);
        fprintf(gp, "set yrange [ %lf : %lf ]\n", mclPose_.getY() - plotRange, mclPose_.getY() + plotRange);

        double axesLength = 2.0;
        double x1 = mclPose_.getX() + axesLength * cos(mclPose_.getYaw());
        double y1 = mclPose_.getY() + axesLength * sin(mclPose_.getYaw());
        double x2 = mclPose_.getX() + axesLength * cos(mclPose_.getYaw() + M_PI / 2.0);
        double y2 = mclPose_.getY() + axesLength * sin(mclPose_.getYaw() + M_PI / 2.0);
        fp = fopen("/tmp/als_ros_robot_pose1.txt", "w");
        fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
        fprintf(fp, "%lf %lf\n", x1, y1);
        fclose(fp);
        fp = fopen("/tmp/als_ros_robot_pose2.txt", "w");
        fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
        fprintf(fp, "%lf %lf\n", x2, y2);
        fclose(fp);

        fp = fopen("/tmp/als_ros_scan_points.txt", "w");
        double yaw = mclPose_.getYaw();
        double sensorX = baseLink2Laser_.getX() * cos(yaw) - baseLink2Laser_.getY() * sin(yaw) + mclPose_.getX();
        double sensorY = baseLink2Laser_.getX() * sin(yaw) + baseLink2Laser_.getY() * cos(yaw) + mclPose_.getY();
        double sensorYaw = baseLink2Laser_.getYaw() + yaw;
        for (size_t i = 0; i < (int)scan_.ranges.size(); ++i) {
            double r = scan_.ranges[i];
            if (r < scan_.range_min || scan_.range_max < r)
                continue;
            double angle = scan_.angle_min + (double)i * scan_.angle_increment + sensorYaw;
            double x = sensorX + r * cos(angle);
            double y = sensorY + r * sin(angle);
            fprintf(fp, "%lf %lf\n", x, y);
        }
        fclose(fp);

        fprintf(gp, "plot \"%s\" with points pointtype 5 pointsize 0.1 lt -1, \
            \"%s\" pointtype 1 pointsize 0.8 lt 1, \"%s\" w l lt 2 lw 3, \"%s\" u 1:2 w l lt 1 lw 3\n", 
            "/tmp/als_ros_map_points.txt",
            "/tmp/als_ros_scan_points.txt",
            "/tmp/als_ros_robot_pose2.txt",
            "/tmp/als_ros_robot_pose1.txt");
        fflush(gp);
    }

private:
    inline double nrand(double n) { return (n * sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * rand() / RAND_MAX)); }

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
        double yaw = -mapOrigin_.getYaw();
        double dx = xx * cos(yaw) + yy * sin(yaw);
        double dy = -xx * sin(yaw) + yy * cos(yaw);
        *x = dx + mapOrigin_.getX();
        *y = dy + mapOrigin_.getY();
    }

    void scanCB(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        if (canUpdateScan_)
            scan_ = *msg;
        if (!gotScan_)
            gotScan_ = true;
    }

    void odomCB(const nav_msgs::msg::Odometry::SharedPtr msg) {
        static double prevTime;
        double currTime = rclcpp::Time(msg->header.stamp).seconds();
        if (isInitialized_) {
            prevTime = currTime;
            isInitialized_ = false;
            return;
        }
        double deltaTime = currTime - prevTime;
        if (deltaTime == 0.0)
            return;

        odomPoseStamp_ = msg->header.stamp;
        deltaX_ += msg->twist.twist.linear.x * deltaTime;
        deltaY_ += msg->twist.twist.linear.y * deltaTime;
        deltaDist_ += msg->twist.twist.linear.x * deltaTime;
        deltaYaw_ += msg->twist.twist.angular.z * deltaTime;
        while (deltaYaw_ < -M_PI)
            deltaYaw_ += 2.0 * M_PI;
        while (deltaYaw_ > M_PI)
            deltaYaw_ -= 2.0 * M_PI;
        deltaTimeSum_ += deltaTime;

        tf2::Quaternion q(msg->pose.pose.orientation.x, 
            msg->pose.pose.orientation.y, 
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        odomPose_.setPose(msg->pose.pose.position.x, msg->pose.pose.position.y, yaw);

        prevTime = currTime;
    }

    void mapCB(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        // perform distance transform to build the distance field
        mapWidth_ = msg->info.width;
        mapHeight_ = msg->info.height;
        mapResolution_ = msg->info.resolution;
        cv::Mat binMap(mapHeight_, mapWidth_, CV_8UC1);
        for (int v = 0; v < mapHeight_; v++) {
            for (int u = 0; u < mapWidth_; u++) {
                int node = v * mapWidth_ + u;
                int val = msg->data[node];
                if (val == 100)
                    binMap.at<uchar>(v, u) = 0;
                else
                    binMap.at<uchar>(v, u) = 1;
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
        tf2::Quaternion q(msg->info.origin.orientation.x, 
            msg->info.origin.orientation.y, 
            msg->info.origin.orientation.z,
            msg->info.origin.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        mapOrigin_.setX(msg->info.origin.position.x);
        mapOrigin_.setY(msg->info.origin.position.y);
        mapOrigin_.setYaw(yaw);
        gotMap_ = true;
    }

    void initialPoseCB(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        tf2::Quaternion q(msg->pose.pose.orientation.x, 
            msg->pose.pose.orientation.y, 
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        mclPose_.setPose(msg->pose.pose.position.x, msg->pose.pose.position.y, yaw);
        resetParticlesDistribution();
        if (estimateReliability_)
            resetReliabilities();
        isInitialized_ = true;
    }

    void glSampledPosesCB(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        if (canUpdateGLSampledPoses_) {
            glSampledPosesStamp_ = msg->header.stamp;
            glSampledPoses_ = *msg;
            isGLSampledPosesUpdated_ = true;
        }
    }

    void resetParticlesDistribution(void) {
        particles_.resize(particlesNum_);
        double xo = mclPose_.getX();
        double yo = mclPose_.getY();
        double yawo = mclPose_.getYaw();
        double wo = 1.0 / (double)particlesNum_;
        for (int i = 0; i < particlesNum_; ++i) {
            double x = xo + nrand(initialNoiseX_);
            double y = yo + nrand(initialNoiseY_);
            double yaw = yawo + nrand(initialNoiseYaw_);
            particles_[i].setPose(x, y, yaw);
            particles_[i].setW(wo);
        }
    }

    void resetReliabilities(void) {
        reliabilities_.resize(particlesNum_, 0.5);
    }

    void rejectUnknownScan(void) {
        unknownScan_ = scan_;
        double xo = baseLink2Laser_.getX();
        double yo = baseLink2Laser_.getY();
        double yawo = baseLink2Laser_.getYaw();
        double hitThreshold = 0.5 * mapResolution_;
        for (int i = 0; i < (int)unknownScan_.ranges.size(); ++i) {
            if (i % scanStep_ != 0) {
                unknownScan_.ranges[i] = 0.0;
                continue;
            }
            double r = unknownScan_.ranges[i];
            if (r <= unknownScan_.range_min || unknownScan_.range_max <= r) {
                unknownScan_.ranges[i] = 0.0;
                continue;
            }
            double laserYaw = (double)i * unknownScan_.angle_increment + unknownScan_.angle_min;
            double pShortSum = 0.0, pBeamSum = 0.0;
            for (int j = 0; j < particlesNum_; ++j) {
                double yaw = particles_[j].getYaw();
                double x = xo * cos(yaw) - yo * sin(yaw) + particles_[j].getX();
                double y = xo * sin(yaw) + yo * cos(yaw) + particles_[j].getY();
                double t = yawo + yaw + laserYaw;
                double dx = mapResolution_ * cos(t);
                double dy = mapResolution_ * sin(t);
                int u, v;
                double expectedRange = -1.0;
                for (double range = 0.0; range <= unknownScan_.range_max; range += mapResolution_) {
                    xy2uv(x, y, &u, &v);
                    if (onMap(u, v)) {
                        double dist = (double)distMap_.at<float>(v, u);
                        if (dist < hitThreshold) {
                            expectedRange = range;
                            break;
                        }
                    } else {
                        break;
                    }
                    x += dx;
                    y += dy;
                }
                if (r <= expectedRange) {
                    double error = expectedRange - r;
                    double pHit = normConstHit_ * exp(-(error * error) * denomHit_) * mapResolution_;
                    double pShort = lambdaShort_ * exp(-lambdaShort_ * r) / (1.0 - exp(-lambdaShort_ * unknownScan_.range_max)) * mapResolution_;
                    pShortSum += pShort;
                    pBeamSum += zHit_ * pHit + zShort_ * pShort + measurementModelRandom_;
                } else {
                    pBeamSum += measurementModelRandom_;
                }
            }
            double pShort = pShortSum;
            double pBeam = pBeamSum;
            double pUnknown = pShortSum / pBeamSum;
            if (pUnknown < unknownScanProbThreshold_) {
                unknownScan_.ranges[i] = 0.0;
            } else {
                // unknown scan is rejected from the scan message used for localization
                scan_.ranges[i] = 0.0;
            }
        }
    }

    double calculateLikelihoodFieldModel(Pose pose, double range, double rangeAngle) {
        if (range <= scan_.range_min || scan_.range_max <= range)
            return measurementModelInvalidScan_;

        double t = pose.getYaw() + rangeAngle;
        double x = range * cos(t) + pose.getX();
        double y = range * sin(t) + pose.getY();
        int u, v;
        xy2uv(x, y, &u, &v);
        double p;
        if (onMap(u, v)) {
            double dist = (double)distMap_.at<float>(v, u);
            double pHit = normConstHit_ * exp(-(dist * dist) * denomHit_) * mapResolution_;
            p = zHit_ * pHit + measurementModelRandom_;
        } else {
            p = measurementModelRandom_;
        }
        if (p > 1.0)
            p = 1.0;
        return p;
    }

    double calculateBeamModel(Pose pose, double range, double rangeAngle) {
        if (range <= scan_.range_min || scan_.range_max <= range)
            return measurementModelInvalidScan_;

        double t = pose.getYaw() + rangeAngle;
        double x = pose.getX();
        double y = pose.getY();
        double dx = mapResolution_ * cos(t);
        double dy = mapResolution_ * sin(t);
        int u, v;
        double expectedRange = -1.0;
        double hitThreshold = 0.5 * mapResolution_;
        for (double r = 0.0; r < scan_.range_max; r += mapResolution_) {
            xy2uv(x, y, &u, &v);
            if (onMap(u, v)) {
                double dist = (double)distMap_.at<float>(v, u);
                if (dist < hitThreshold) {
                    expectedRange = r;
                    break;
                }
            } else {
                break;
            }
            x += dx;
            y += dy;
        }

        double p;
        if (range <= expectedRange) {
            double error = expectedRange - range;
            double pHit = normConstHit_ * exp(-(error * error) * denomHit_) * mapResolution_;
            double pShort = lambdaShort_ * exp(-lambdaShort_ * range) / (1.0 - exp(-lambdaShort_ * scan_.range_max)) * mapResolution_;
            p = zHit_ * pHit + zShort_ * pShort + measurementModelRandom_;
        } else {
            p = measurementModelRandom_;
        }
        if (p > 1.0)
            p = 1.0;
        return p;
    }

    double calculateClassConditionalMeasurementModel(Pose pose, double range, double rangeAngle) {
        if (range <= scan_.range_min || scan_.range_max <= range)
            return measurementModelInvalidScan_;

        double t = pose.getYaw() + rangeAngle;
        double x = range * cos(t) + pose.getX();
        double y = range * sin(t) + pose.getY();
        double pUnknown = lambdaUnknown_ * exp(-lambdaUnknown_ * range) / (1.0 - exp(-lambdaUnknown_ * scan_.range_max)) * mapResolution_ * pUnknownPrior_;
        int u, v;
        xy2uv(x, y, &u, &v);
        double p = pUnknown;
        if (onMap(u, v)) {
            double dist = (double)distMap_.at<float>(v, u);
            double pHit = normConstHit_ * exp(-(dist * dist) * denomHit_) * mapResolution_;
            p += (zHit_ * pHit + measurementModelRandom_) * pKnownPrior_;
        } else {
            p += measurementModelRandom_ * pKnownPrior_;
        }
        if (p > 1.0)
            p = 1.0;
        return p;
    }

    void estimateUnknownScanWithClassConditionalMeasurementModel(Pose pose) {
        unknownScan_ = scan_;
        double yaw = pose.getYaw();
        double sensorX = baseLink2Laser_.getX() * cos(yaw) - baseLink2Laser_.getY() * sin(yaw) + pose.getX();
        double sensorY = baseLink2Laser_.getX() * sin(yaw) + baseLink2Laser_.getY() * cos(yaw) + pose.getY();
        double sensorYaw = baseLink2Laser_.getYaw() + yaw;
        for (int i = 0; i < (int)unknownScan_.ranges.size(); ++i) {
            double r = unknownScan_.ranges[i];
            if (r <= unknownScan_.range_min || unknownScan_.range_max <= r) {
                unknownScan_.ranges[i] = 0.0;
                continue;
            }
            double t = sensorYaw + (double)i * unknownScan_.angle_increment + unknownScan_.angle_min;
            double x = r * cos(t) + sensorX;
            double y = r * sin(t) + sensorY;
            int u, v;
            xy2uv(x, y, &u, &v);
            double pKnown;
            double pUnknown = lambdaUnknown_ * exp(-lambdaUnknown_ * r) / (1.0 - exp(-lambdaUnknown_ * unknownScan_.range_max)) * mapResolution_ * pUnknownPrior_;
            if (onMap(u, v)) {
                double dist = (double)distMap_.at<float>(v, u);
                double pHit = normConstHit_ * exp(-(dist * dist) * denomHit_) * mapResolution_;
                pKnown = (zHit_ * pHit + measurementModelRandom_) * pKnownPrior_;
            } else {
                pKnown = measurementModelRandom_ * pKnownPrior_;
            }
            double sum = pKnown + pUnknown;
            pUnknown /= sum;
            if (pUnknown < unknownScanProbThreshold_)
                unknownScan_.ranges[i] = 0.0;
        }
    }

}; // class MCL

} // namespace als_ros

#endif // __MCL_H__
