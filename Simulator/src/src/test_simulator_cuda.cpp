#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int32.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <yaml-cpp/yaml.h>
#include <visualization_msgs/Marker.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include "sensor_simulator.cuh"
#include "maps.hpp"

using namespace raycast;

class SensorSimulator
{
public:
    SensorSimulator(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh)
    {
        std::string config_path = CONFIG_FILE_PATH;
        pnh_.param("config_path", config_path, config_path);
        ROS_INFO_STREAM("Loading simulator config from: " << config_path);
        YAML::Node config = YAML::LoadFile(config_path);
        applyRosParamOverrides(config);

        camera_ = new CameraParams();
        camera_->fx = config["camera"]["fx"].as<float>();
        camera_->fy = config["camera"]["fy"].as<float>();
        camera_->cx = config["camera"]["cx"].as<float>();
        camera_->cy = config["camera"]["cy"].as<float>();
        camera_->image_width = config["camera"]["image_width"].as<int>();
        camera_->image_height = config["camera"]["image_height"].as<int>();
        camera_->max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
        camera_->normalize_depth = config["camera"]["normalize_depth"].as<bool>();
        const float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0f;
        quat_bc_ = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());

        lidar_ = new LidarParams();
        lidar_->vertical_lines = config["lidar"]["vertical_lines"].as<int>();
        lidar_->vertical_angle_start = config["lidar"]["vertical_angle_start"].as<float>();
        lidar_->vertical_angle_end = config["lidar"]["vertical_angle_end"].as<float>();
        lidar_->horizontal_num = config["lidar"]["horizontal_num"].as<int>();
        lidar_->horizontal_resolution = config["lidar"]["horizontal_resolution"].as<float>();
        lidar_->max_lidar_dist = config["lidar"]["max_lidar_dist"].as<float>();

        render_lidar_ = config["render_lidar"].as<bool>();
        render_depth_ = config["render_depth"].as<bool>();
        render_target_mask_ = config["render_target_mask"] ? config["render_target_mask"].as<bool>() : true;
        depth_pub_duration_ = ros::Duration(1.0 / config["depth_fps"].as<float>());
        lidar_pub_duration_ = ros::Duration(1.0 / config["lidar_fps"].as<float>());
        visualize_local_map_ = config["visualize_local_map"] ? config["visualize_local_map"].as<bool>() : true;
        target_radius_ = config["target"] && config["target"]["radius"] ? config["target"]["radius"].as<float>() : 0.35f;
        target_occlusion_margin_ = config["target"] && config["target"]["occlusion_margin"] ? config["target"]["occlusion_margin"].as<float>() : 0.3f;
        target_mask_bbox_scale_ = config["target"] && config["target"]["mask_bbox_scale"] ? config["target"]["mask_bbox_scale"].as<float>() : 1.2f;

        const std::string ply_file = config["ply_file"].as<std::string>();
        const bool use_random_map = config["random_map"].as<bool>();
        const float resolution = config["resolution"].as<float>();
        const float mock_map_leaf_size = config["mock_map_leaf_size"] ? config["mock_map_leaf_size"].as<float>() : resolution;
        const int occupy_threshold = config["occupy_threshold"].as<int>();
        const int seed = config["seed"].as<int>();
        int size_x = config["x_length"].as<int>();
        int size_y = config["y_length"].as<int>();
        int size_z = config["z_length"].as<int>();
        const int maze_type = config["maze_type"].as<int>();
        const double scale = 1.0 / resolution;
        size_x = size_x * scale;
        size_y = size_y * scale;
        size_z = size_z * scale;

        pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("mock_map", 1);
        local_map_visual_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_map_visual", 1);
        image_pub_ = nh_.advertise<sensor_msgs::Image>(config["depth_topic"].as<std::string>(), 1);
        target_mask_pub_ = nh_.advertise<sensor_msgs::Image>(
            config["target_mask_topic"] ? config["target_mask_topic"].as<std::string>() : std::string("/target_mask_image"), 1);
        target_depth_pub_ = nh_.advertise<sensor_msgs::Image>(
            config["target_depth_topic"] ? config["target_depth_topic"].as<std::string>() : std::string("/target_depth_image"), 1);
        point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(config["lidar_topic"].as<std::string>(), 1);
        collision_counter_pub_ = nh_.advertise<std_msgs::Int32>("/yopo/collision_counter", 1);
        collision_counter_total_pub_ = nh_.advertise<std_msgs::Int32>("/yopo/collision_counter_total", 1);
        target_collision_counter_pub_ = nh_.advertise<std_msgs::Int32>("/yopo/target_collision_counter", 1);
        target_collision_counter_total_pub_ = nh_.advertise<std_msgs::Int32>("/yopo/target_collision_counter_total", 1);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (use_random_map)
        {
            printf("1.Generate Random Map... \n");
            mocka::Maps::BasicInfo info;
            info.sizeX = size_x;
            info.sizeY = size_y;
            info.sizeZ = size_z;
            info.seed = seed;
            info.scale = scale;
            info.cloud = cloud;

            mocka::Maps map;
            map.setParam(config);
            map.setInfo(info);
            map.generate(maze_type);
        }
        else
        {
            printf("1.Reading Point Cloud %s... \n", ply_file.c_str());
            if (pcl::io::loadPLYFile(ply_file, *cloud) == -1)
                PCL_ERROR("Couldn't read PLY file \n");
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr map_visual_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (mock_map_leaf_size > resolution)
        {
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setInputCloud(cloud);
            voxel_filter.setLeafSize(mock_map_leaf_size, mock_map_leaf_size, mock_map_leaf_size);
            voxel_filter.filter(*map_visual_cloud);
        }
        else
        {
            *map_visual_cloud = *cloud;
        }

        pcl::toROSMsg(*map_visual_cloud, map_output_);
        map_output_.header.frame_id = "world";

        std::cout << "PointCloud size (raw/visual): " << cloud->points.size() << " / "
                  << map_visual_cloud->points.size() << std::endl;
        printf("2.Mapping... \n");
        grid_map_ = new GridMap(cloud, resolution, occupy_threshold);

        const ros::Time now = ros::Time::now();
        next_depth_pub_time_ = now;
        next_lidar_pub_time_ = now;
        odom_sub_ = nh_.subscribe(
            config["odom_topic"].as<std::string>(),
            1,
            &SensorSimulator::odomCallback,
            this,
            ros::TransportHints().tcpNoDelay());
        target_odom_sub_ = nh_.subscribe(
            config["target_odom_topic"] ? config["target_odom_topic"].as<std::string>() : std::string("/target/odom"),
            1,
            &SensorSimulator::targetOdomCallback,
            this,
            ros::TransportHints().tcpNoDelay());
        timer_map_ = nh_.createTimer(ros::Duration(1.0), &SensorSimulator::timerMapCallback, this);

        printf("3.Simulation Ready!\n");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void renderDepthCallback(const ros::Time &stamp);
    void renderLidarCallback(const ros::Time &stamp);
    void timerMapCallback(const ros::TimerEvent &);
    void publishCollisionCounters();
    void targetOdomCallback(const nav_msgs::Odometry::ConstPtr &msg);

private:
    void applyRosParamOverrides(YAML::Node &config);
    bool inStaticCollision() const;
    bool inTargetStaticCollision() const;
    void publishLocalMapVisual(const ros::Time &stamp);
    void overlayTargetAndMask(cv::Mat &depth_image, cv::Mat &target_mask) const;

    bool render_depth_{false};
    bool render_lidar_{false};
    bool render_target_mask_{true};
    bool visualize_local_map_{true};
    bool odom_init_{false};
    bool target_odom_init_{false};
    bool in_static_collision_{false};

    Eigen::Quaternionf quat_{Eigen::Quaternionf::Identity()};
    Eigen::Quaternionf quat_wc_{Eigen::Quaternionf::Identity()};
    Eigen::Quaternionf target_quat_{Eigen::Quaternionf::Identity()};
    Eigen::Quaternionf target_quat_wc_{Eigen::Quaternionf::Identity()};
    Eigen::Quaternionf quat_bc_{Eigen::Quaternionf::Identity()};
    Eigen::Vector3f pos_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f target_pos_{Eigen::Vector3f::Zero()};
    CameraParams *camera_{nullptr};
    LidarParams *lidar_{nullptr};
    GridMap *grid_map_{nullptr};
    pcl::PointCloud<pcl::PointXYZ> local_map_world_;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher pcl_pub_;
    ros::Publisher local_map_visual_pub_;
    ros::Publisher image_pub_;
    ros::Publisher target_mask_pub_;
    ros::Publisher target_depth_pub_;
    ros::Publisher point_cloud_pub_;
    ros::Publisher collision_counter_pub_;
    ros::Publisher collision_counter_total_pub_;
    ros::Publisher target_collision_counter_pub_;
    ros::Publisher target_collision_counter_total_pub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber target_odom_sub_;
    ros::Timer timer_map_;
    sensor_msgs::PointCloud2 map_output_;

    ros::Time next_depth_pub_time_;
    ros::Time next_lidar_pub_time_;
    ros::Duration depth_pub_duration_;
    ros::Duration lidar_pub_duration_;
    double depth_time_{0.0};
    double lidar_time_{0.0};
    int depth_count_{0};
    int lidar_count_{0};
    int collision_counter_{0};
    int target_collision_counter_{0};
    bool target_in_static_collision_{false};
    float target_radius_{0.35f};
    float target_occlusion_margin_{0.3f};
    float target_mask_bbox_scale_{1.2f};
};

void SensorSimulator::applyRosParamOverrides(YAML::Node &config)
{
    bool visualize_local_map = config["visualize_local_map"] ? config["visualize_local_map"].as<bool>() : true;
    pnh_.param("visualize_local_map", visualize_local_map, visualize_local_map);
    config["visualize_local_map"] = visualize_local_map;
}

bool SensorSimulator::inStaticCollision() const
{
    return grid_map_->mapQueryHost(Vector3f(pos_.x(), pos_.y(), pos_.z())) == 1;
}

bool SensorSimulator::inTargetStaticCollision() const
{
    if (!target_odom_init_)
        return false;
    return grid_map_->mapQueryHost(Vector3f(target_pos_.x(), target_pos_.y(), target_pos_.z())) == 1;
}

void SensorSimulator::renderDepthCallback(const ros::Time &stamp)
{
    if (!render_depth_)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat_wc_.w(), quat_wc_.x(), quat_wc_.y(), quat_wc_.z(),
                             pos_.x(), pos_.y(), pos_.z());
    cv::Mat depth_image;
    renderDepthImage(grid_map_, camera_, T_wc, depth_image);
    cv::Mat target_mask;
    if (render_target_mask_)
    {
        target_mask = cv::Mat::zeros(camera_->image_height, camera_->image_width, CV_8UC1);
        overlayTargetAndMask(depth_image, target_mask);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    depth_time_ += elapsed.count();
    depth_count_++;

    sensor_msgs::Image ros_image;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = stamp;
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = depth_image;
    cv_image.toImageMsg(ros_image);
    image_pub_.publish(ros_image);

    if (target_odom_init_)
    {
        cudaMat::SE3<float> target_T_wc(
            target_quat_wc_.w(), target_quat_wc_.x(), target_quat_wc_.y(), target_quat_wc_.z(),
            target_pos_.x(), target_pos_.y(), target_pos_.z());
        cv::Mat target_depth_image;
        renderDepthImage(grid_map_, camera_, target_T_wc, target_depth_image);
        sensor_msgs::Image target_depth_msg;
        cv_bridge::CvImage target_depth_bridge;
        target_depth_bridge.header.stamp = stamp;
        target_depth_bridge.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        target_depth_bridge.image = target_depth_image;
        target_depth_bridge.toImageMsg(target_depth_msg);
        target_depth_pub_.publish(target_depth_msg);
    }

    if (render_target_mask_)
    {
        sensor_msgs::Image mask_msg;
        cv_bridge::CvImage mask_bridge;
        mask_bridge.header.stamp = stamp;
        mask_bridge.encoding = sensor_msgs::image_encodings::MONO8;
        mask_bridge.image = target_mask;
        mask_bridge.toImageMsg(mask_msg);
        target_mask_pub_.publish(mask_msg);
    }
}

void SensorSimulator::overlayTargetAndMask(cv::Mat &depth_image, cv::Mat &target_mask) const
{
    if (!target_odom_init_)
        return;

    const Eigen::Matrix3f R_cw = quat_wc_.toRotationMatrix().transpose();
    const Eigen::Vector3f target_c = R_cw * (target_pos_ - pos_);
    if (target_c.x() <= 0.1f || target_c.x() > camera_->max_depth_dist)
        return;

    const float u_center = camera_->cx - camera_->fx * target_c.y() / target_c.x();
    const float v_center = camera_->cy - camera_->fy * target_c.z() / target_c.x();
    const int radius_px = std::max(2, std::min(24, static_cast<int>(std::ceil(camera_->fx * target_radius_ / target_c.x()))));
    const int u0 = std::max(0, static_cast<int>(std::floor(u_center)) - radius_px);
    const int u1 = std::min(camera_->image_width - 1, static_cast<int>(std::ceil(u_center)) + radius_px);
    const int v0 = std::max(0, static_cast<int>(std::floor(v_center)) - radius_px);
    const int v1 = std::min(camera_->image_height - 1, static_cast<int>(std::ceil(v_center)) + radius_px);
    if (u0 > u1 || v0 > v1)
        return;

    const int bbox_half = std::max(2, static_cast<int>(std::ceil(radius_px * target_mask_bbox_scale_)));
    const int bu0 = std::max(0, static_cast<int>(std::floor(u_center)) - bbox_half);
    const int bu1 = std::min(camera_->image_width - 1, static_cast<int>(std::ceil(u_center)) + bbox_half);
    const int bv0 = std::max(0, static_cast<int>(std::floor(v_center)) - bbox_half);
    const int bv1 = std::min(camera_->image_height - 1, static_cast<int>(std::ceil(v_center)) + bbox_half);
    cv::rectangle(target_mask, cv::Point(bu0, bv0), cv::Point(bu1, bv1), cv::Scalar(255), cv::FILLED);

    for (int v = v0; v <= v1; ++v)
    {
        for (int u = u0; u <= u1; ++u)
        {
            const float y_at_target_x = -(u - camera_->cx) / camera_->fx * target_c.x();
            const float z_at_target_x = -(v - camera_->cy) / camera_->fy * target_c.x();
            const float dy = y_at_target_x - target_c.y();
            const float dz = z_at_target_x - target_c.z();
            const float lateral_sq = dy * dy + dz * dz;
            const float radius_sq = target_radius_ * target_radius_;
            if (lateral_sq > radius_sq)
                continue;

            const float surface_depth = target_c.x() - std::sqrt(std::max(0.0f, radius_sq - lateral_sq));
            float &depth_ref = depth_image.at<float>(v, u);
            depth_ref = surface_depth;
        }
    }
}

void SensorSimulator::renderLidarCallback(const ros::Time &stamp)
{
    if (!render_lidar_)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat_.w(), quat_.x(), quat_.y(), quat_.z(),
                             pos_.x(), pos_.y(), pos_.z());
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    renderLidarPointcloud(grid_map_, lidar_, T_wc, lidar_points);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    lidar_time_ += elapsed.count();
    lidar_count_++;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = stamp;
    output.header.frame_id = "odom";
    point_cloud_pub_.publish(output);

    if (!visualize_local_map_)
    {
        local_map_world_.clear();
        return;
    }

    local_map_world_.clear();
    local_map_world_.points.reserve(lidar_points.points.size());
    for (const auto &point_local : lidar_points.points)
    {
        const float3 point_world = T_wc * make_float3(point_local.x, point_local.y, point_local.z);
        local_map_world_.points.emplace_back(point_world.x, point_world.y, point_world.z);
    }
    local_map_world_.width = local_map_world_.points.size();
    local_map_world_.height = 1;
    local_map_world_.is_dense = true;
    publishLocalMapVisual(stamp);
}

void SensorSimulator::timerMapCallback(const ros::TimerEvent &)
{
    if (pcl_pub_.getNumSubscribers() > 0)
        pcl_pub_.publish(map_output_);
}

void SensorSimulator::publishLocalMapVisual(const ros::Time &stamp)
{
    if (!visualize_local_map_)
        return;
    if (local_map_visual_pub_.getNumSubscribers() == 0)
        return;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(local_map_world_, output);
    output.header.stamp = stamp;
    output.header.frame_id = "world";
    local_map_visual_pub_.publish(output);
}

void SensorSimulator::publishCollisionCounters()
{
    std_msgs::Int32 msg;
    msg.data = collision_counter_;
    collision_counter_pub_.publish(msg);
    collision_counter_total_pub_.publish(msg);

    std_msgs::Int32 target_msg;
    target_msg.data = target_collision_counter_;
    target_collision_counter_pub_.publish(target_msg);
    target_collision_counter_total_pub_.publish(target_msg);
}

void SensorSimulator::targetOdomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    target_quat_.x() = msg->pose.pose.orientation.x;
    target_quat_.y() = msg->pose.pose.orientation.y;
    target_quat_.z() = msg->pose.pose.orientation.z;
    target_quat_.w() = msg->pose.pose.orientation.w;
    target_quat_wc_ = target_quat_ * quat_bc_;
    target_pos_.x() = msg->pose.pose.position.x;
    target_pos_.y() = msg->pose.pose.position.y;
    target_pos_.z() = msg->pose.pose.position.z;
    target_odom_init_ = true;
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    quat_.x() = msg->pose.pose.orientation.x;
    quat_.y() = msg->pose.pose.orientation.y;
    quat_.z() = msg->pose.pose.orientation.z;
    quat_.w() = msg->pose.pose.orientation.w;
    quat_wc_ = quat_ * quat_bc_;

    pos_.x() = msg->pose.pose.position.x;
    pos_.y() = msg->pose.pose.position.y;
    pos_.z() = msg->pose.pose.position.z;
    odom_init_ = true;

    const bool static_collision = inStaticCollision();
    if (static_collision && !in_static_collision_)
    {
        collision_counter_ += 1;
        ROS_WARN_THROTTLE(1.0, "Occupied-voxel collision detected. total=%d", collision_counter_);
    }
    in_static_collision_ = static_collision;
    const bool target_static_collision = inTargetStaticCollision();
    if (target_static_collision && !target_in_static_collision_)
    {
        target_collision_counter_ += 1;
        ROS_WARN_THROTTLE(1.0, "Target occupied-voxel collision detected. total=%d", target_collision_counter_);
    }
    target_in_static_collision_ = target_static_collision;
    publishCollisionCounters();

    const ros::Time tnow = ros::Time::now();
    if (fabs((tnow - next_depth_pub_time_).toSec()) > 10 * depth_pub_duration_.toSec())
        next_depth_pub_time_ = tnow;
    if (fabs((tnow - next_lidar_pub_time_).toSec()) > 10 * lidar_pub_duration_.toSec())
        next_lidar_pub_time_ = tnow;

    if (tnow >= next_depth_pub_time_)
    {
        next_depth_pub_time_ += depth_pub_duration_;
        renderDepthCallback(msg->header.stamp);
    }
    if (tnow >= next_lidar_pub_time_)
    {
        next_lidar_pub_time_ += lidar_pub_duration_;
        renderLidarCallback(msg->header.stamp);
    }

    const ros::Duration render_duration = ros::Time::now() - tnow;
    if (render_duration > depth_pub_duration_ || render_duration > lidar_pub_duration_)
    {
        ROS_WARN("Current Rendering time: %.2f ms, delay too much!", 1000 * render_duration.toSec());
        std::cout << "Average Depth Rendering time: " << (depth_time_ / (depth_count_ + 1e-8)) * 1000 << " ms" << std::endl;
        std::cout << "Average Lidar Rendering time: " << (lidar_time_ / (lidar_count_ + 1e-8)) * 1000 << " ms" << std::endl;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sensor_simulator_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    SensorSimulator sensor_simulator(nh, pnh);
    return 0;
}
