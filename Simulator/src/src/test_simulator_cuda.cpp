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
#include <boost/bind/bind.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "sensor_simulator.cuh"
#include "maps.hpp"

using namespace raycast;
using boost::placeholders::_1;

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
        render_swarm_detector_mask_ = config["render_swarm_detector_mask"] ? config["render_swarm_detector_mask"].as<bool>() : true;
        visualize_local_map_ = config["visualize_local_map"] ? config["visualize_local_map"].as<bool>() : true;
        depth_pub_duration_ = ros::Duration(1.0 / config["depth_fps"].as<float>());
        lidar_pub_duration_ = ros::Duration(1.0 / config["lidar_fps"].as<float>());

        swarm_enabled_ = config["swarm"]["enabled"].as<bool>();
        swarm_uav_num_ = std::max(1, config["swarm"]["uav_num"].as<int>());
        swarm_namespace_prefix_ = config["swarm"]["namespace_prefix"].as<std::string>();
        collision_radius_ = config["swarm"]["collision_radius"].as<float>();
        const YAML::Node target_size = config["target"]["ellipsoid_size"];
        target_ellipsoid_axes_ = Eigen::Vector3f(
            target_size[0].as<float>() * 0.5f,
            target_size[1].as<float>() * 0.5f,
            target_size[2].as<float>() * 0.5f);
        target_occlusion_margin_ = config["target"] && config["target"]["occlusion_margin"] ? config["target"]["occlusion_margin"].as<float>() : 0.3f;
        target_mask_min_forward_depth_ = config["target"] && config["target"]["mask_min_forward_depth"]
                                             ? config["target"]["mask_min_forward_depth"].as<float>()
                                             : 0.1f;
        mask_min_visible_pixels_ = config["target"] && config["target"]["mask_min_visible_pixels"]
                                           ? config["target"]["mask_min_visible_pixels"].as<int>()
                                           : 4;
        mask_min_visible_ratio_ = config["target"] && config["target"]["mask_min_visible_ratio"]
                                          ? config["target"]["mask_min_visible_ratio"].as<float>()
                                          : 0.03f;

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
        collision_counter_total_pub_ = nh_.advertise<std_msgs::Int32>("/yopo/collision_counter_total", 1);
        target_collision_counter_total_pub_ = nh_.advertise<std_msgs::Int32>("/yopo/target_collision_counter_total", 1);
        uav_collision_counter_total_pub_ = nh_.advertise<std_msgs::Int32>("/yopo/uav_collision_counter_total", 1);

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

        setupRobots(config);
        if (!swarm_enabled_)
        {
            target_odom_sub_ = nh_.subscribe(
                config["target_odom_topic"] ? config["target_odom_topic"].as<std::string>() : std::string("/target/odom"),
                1,
                &SensorSimulator::targetOdomCallback,
                this,
                ros::TransportHints().tcpNoDelay());
        }
        timer_map_ = nh_.createTimer(ros::Duration(1.0), &SensorSimulator::timerMapCallback, this);

        printf("3.Simulation Ready! robots=%zu swarm=%s\n", robots_.size(), swarm_enabled_ ? "true" : "false");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg, size_t robot_index);
    void targetOdomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void renderDepthCallback(size_t robot_index, const ros::Time &stamp);
    void renderLidarCallback(size_t robot_index, const ros::Time &stamp);
    void timerMapCallback(const ros::TimerEvent &);

private:
    struct RobotChannels
    {
        std::string name;
        std::string odom_topic;
        std::string depth_topic;
        std::string target_mask_topic;
        std::string target_depth_topic;
        std::string lidar_topic;

        ros::Publisher image_pub;
        ros::Publisher target_mask_pub;
        ros::Publisher target_depth_pub;
        ros::Publisher point_cloud_pub;
        ros::Publisher collision_pub;
        ros::Publisher uav_collision_pub;
        ros::Subscriber odom_sub;

        Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
        Eigen::Quaternionf quat_wc = Eigen::Quaternionf::Identity();
        Eigen::Vector3f pos = Eigen::Vector3f::Zero();
        pcl::PointCloud<pcl::PointXYZ> local_map_world;

        ros::Time next_depth_pub_time;
        ros::Time next_lidar_pub_time;
        int collision_counter = 0;
        int uav_collision_counter = 0;
        bool in_static_collision = false;
        bool in_uav_collision = false;
        bool odom_init = false;
    };

    void applyRosParamOverrides(YAML::Node &config);
    void setupRobots(const YAML::Node &config);
    std::vector<EllipsoidObstacle> getDynamicObstacles(size_t robot_index) const;
    bool inStaticCollision(const RobotChannels &robot) const;
    bool inDynamicCollision(size_t robot_index) const;
    bool inTargetStaticCollision() const;
    bool projectTarget(const RobotChannels &robot, const Eigen::Vector3f &target_pos,
                       Eigen::Vector3f &target_c, float &u_center, float &v_center,
                       int &radius_u_px, int &radius_v_px) const;
    bool targetVisibleInDepth(const cv::Mat &depth_image, const Eigen::Vector3f &target_c,
                              float u_center, float v_center, int radius_u_px, int radius_v_px) const;
    void drawTargetMask(cv::Mat &target_mask, float u_center, float v_center,
                        int radius_u_px, int radius_v_px) const;
    void overlaySingleTargetAndMask(RobotChannels &robot, cv::Mat &depth_image, cv::Mat &target_mask) const;
    void overlaySwarmDetectorMasks(size_t robot_index, const cv::Mat &depth_image, cv::Mat &target_mask) const;
    void publishCollisionCounter(size_t robot_index);
    void publishCollisionCounterTotals();
    void publishLocalMapVisual(const ros::Time &stamp);

    bool render_depth_{false};
    bool render_lidar_{false};
    bool render_target_mask_{true};
    bool render_swarm_detector_mask_{true};
    bool visualize_local_map_{true};
    bool swarm_enabled_{false};
    int swarm_uav_num_{1};
    float collision_radius_{0.155f};
    std::string swarm_namespace_prefix_{"uav"};

    Eigen::Quaternionf quat_bc_{Eigen::Quaternionf::Identity()};
    Eigen::Quaternionf target_quat_{Eigen::Quaternionf::Identity()};
    Eigen::Quaternionf target_quat_wc_{Eigen::Quaternionf::Identity()};
    Eigen::Vector3f target_pos_{Eigen::Vector3f::Zero()};
    bool target_odom_init_{false};
    bool target_in_static_collision_{false};

    CameraParams *camera_{nullptr};
    LidarParams *lidar_{nullptr};
    GridMap *grid_map_{nullptr};

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher pcl_pub_;
    ros::Publisher local_map_visual_pub_;
    ros::Publisher collision_counter_total_pub_;
    ros::Publisher target_collision_counter_total_pub_;
    ros::Publisher uav_collision_counter_total_pub_;
    ros::Subscriber target_odom_sub_;
    ros::Timer timer_map_;
    sensor_msgs::PointCloud2 map_output_;
    std::vector<RobotChannels> robots_;

    ros::Duration depth_pub_duration_;
    ros::Duration lidar_pub_duration_;
    double depth_time_{0.0};
    double lidar_time_{0.0};
    int depth_count_{0};
    int lidar_count_{0};
    int collision_counter_total_{0};
    int target_collision_counter_total_{0};
    int uav_collision_counter_total_{0};
    Eigen::Vector3f target_ellipsoid_axes_{0.155f, 0.155f, 0.07f};
    float target_occlusion_margin_{0.3f};
    float target_mask_min_forward_depth_{0.1f};
    int mask_min_visible_pixels_{4};
    float mask_min_visible_ratio_{0.03f};
};

void SensorSimulator::applyRosParamOverrides(YAML::Node &config)
{
    if (!config["swarm"])
        config["swarm"] = YAML::Node(YAML::NodeType::Map);
    if (!config["target"])
        config["target"] = YAML::Node(YAML::NodeType::Map);

    bool swarm_enabled = config["swarm"]["enabled"] ? config["swarm"]["enabled"].as<bool>() : false;
    int swarm_uav_num = config["swarm"]["uav_num"] ? config["swarm"]["uav_num"].as<int>() : 1;
    std::string namespace_prefix = config["swarm"]["namespace_prefix"] ? config["swarm"]["namespace_prefix"].as<std::string>() : "uav";
    double ring_radius = config["swarm"]["ring_radius"] ? config["swarm"]["ring_radius"].as<double>() : 8.0;
    double altitude = config["swarm"]["altitude"] ? config["swarm"]["altitude"].as<double>() : 2.0;
    double spawn_clear_radius = config["swarm"]["spawn_clear_radius"] ? config["swarm"]["spawn_clear_radius"].as<double>() : 2.2;
    double collision_radius = config["swarm"]["collision_radius"] ? config["swarm"]["collision_radius"].as<double>() : 0.155;
    double forward_distance = config["swarm"]["forward_distance"] ? config["swarm"]["forward_distance"].as<double>() : 50.0;
    double formation_start_x = config["swarm"]["formation_start_x"] ? config["swarm"]["formation_start_x"].as<double>() : -30.0;
    double formation_row_spacing = config["swarm"]["formation_row_spacing"] ? config["swarm"]["formation_row_spacing"].as<double>() : 0.8660254;
    double formation_lateral_spacing = config["swarm"]["formation_lateral_spacing"] ? config["swarm"]["formation_lateral_spacing"].as<double>() : 1.0;
    std::string formation_rows_csv = "4,3,2,1";
    if (config["swarm"]["formation_rows"])
    {
        std::ostringstream rows_stream;
        const auto rows = config["swarm"]["formation_rows"].as<std::vector<int>>();
        for (size_t i = 0; i < rows.size(); ++i)
        {
            if (i > 0)
                rows_stream << ",";
            rows_stream << rows[i];
        }
        formation_rows_csv = rows_stream.str();
    }
    bool visualize_local_map = config["visualize_local_map"] ? config["visualize_local_map"].as<bool>() : true;
    bool render_swarm_detector_mask = config["render_swarm_detector_mask"] ? config["render_swarm_detector_mask"].as<bool>() : true;
    pnh_.param("swarm_enabled", swarm_enabled, swarm_enabled);
    pnh_.param("swarm_uav_num", swarm_uav_num, swarm_uav_num);
    pnh_.param("swarm_namespace_prefix", namespace_prefix, namespace_prefix);
    pnh_.param("swarm_ring_radius", ring_radius, ring_radius);
    pnh_.param("swarm_altitude", altitude, altitude);
    pnh_.param("swarm_spawn_clear_radius", spawn_clear_radius, spawn_clear_radius);
    pnh_.param("swarm_collision_radius", collision_radius, collision_radius);
    pnh_.param("swarm_forward_distance", forward_distance, forward_distance);
    pnh_.param("swarm_formation_start_x", formation_start_x, formation_start_x);
    pnh_.param("swarm_formation_row_spacing", formation_row_spacing, formation_row_spacing);
    pnh_.param("swarm_formation_lateral_spacing", formation_lateral_spacing, formation_lateral_spacing);
    pnh_.param("swarm_formation_rows", formation_rows_csv, formation_rows_csv);
    pnh_.param("visualize_local_map", visualize_local_map, visualize_local_map);
    pnh_.param("render_swarm_detector_mask", render_swarm_detector_mask, render_swarm_detector_mask);

    config["swarm"]["enabled"] = swarm_enabled;
    config["swarm"]["uav_num"] = swarm_uav_num;
    config["swarm"]["namespace_prefix"] = namespace_prefix;
    config["swarm"]["ring_radius"] = ring_radius;
    config["swarm"]["altitude"] = altitude;
    config["swarm"]["spawn_clear_radius"] = spawn_clear_radius;
    config["swarm"]["collision_radius"] = collision_radius;
    config["swarm"]["forward_distance"] = forward_distance;
    config["swarm"]["formation_start_x"] = formation_start_x;
    config["swarm"]["formation_row_spacing"] = formation_row_spacing;
    config["swarm"]["formation_lateral_spacing"] = formation_lateral_spacing;
    YAML::Node formation_rows_node(YAML::NodeType::Sequence);
    std::stringstream rows_stream(formation_rows_csv);
    std::string row_token;
    while (std::getline(rows_stream, row_token, ','))
    {
        if (row_token.empty())
            continue;
        formation_rows_node.push_back(std::stoi(row_token));
    }
    if (formation_rows_node.size() > 0)
        config["swarm"]["formation_rows"] = formation_rows_node;
    config["visualize_local_map"] = visualize_local_map;
    config["render_swarm_detector_mask"] = render_swarm_detector_mask;
}

void SensorSimulator::setupRobots(const YAML::Node &config)
{
    robots_.clear();

    if (swarm_enabled_)
    {
        robots_.resize(swarm_uav_num_);
        for (int i = 0; i < swarm_uav_num_; ++i)
        {
            RobotChannels robot;
            robot.name = swarm_namespace_prefix_ + std::to_string(i);
            robot.odom_topic = "/" + robot.name + "/sim/odom";
            robot.depth_topic = "/" + robot.name + "/depth_image";
            robot.target_mask_topic = "/" + robot.name + "/target_mask_image";
            robot.lidar_topic = "/" + robot.name + "/lidar_points";
            robots_[i] = robot;
        }
    }
    else
    {
        RobotChannels robot;
        robot.name = "uav0";
        robot.odom_topic = config["odom_topic"].as<std::string>();
        robot.depth_topic = config["depth_topic"].as<std::string>();
        robot.target_mask_topic = config["target_mask_topic"] ? config["target_mask_topic"].as<std::string>() : "/target_mask_image";
        robot.target_depth_topic = config["target_depth_topic"] ? config["target_depth_topic"].as<std::string>() : "/target_depth_image";
        robot.lidar_topic = config["lidar_topic"].as<std::string>();
        robots_.push_back(robot);
    }

    const ros::Time now = ros::Time::now();
    for (size_t i = 0; i < robots_.size(); ++i)
    {
        robots_[i].image_pub = nh_.advertise<sensor_msgs::Image>(robots_[i].depth_topic, 1);
        robots_[i].target_mask_pub = nh_.advertise<sensor_msgs::Image>(robots_[i].target_mask_topic, 1);
        if (!robots_[i].target_depth_topic.empty())
            robots_[i].target_depth_pub = nh_.advertise<sensor_msgs::Image>(robots_[i].target_depth_topic, 1);
        robots_[i].point_cloud_pub = nh_.advertise<sensor_msgs::PointCloud2>(robots_[i].lidar_topic, 1);
        robots_[i].collision_pub = nh_.advertise<std_msgs::Int32>("/" + robots_[i].name + "/yopo/collision_counter", 1);
        robots_[i].uav_collision_pub = nh_.advertise<std_msgs::Int32>("/" + robots_[i].name + "/yopo/uav_collision_counter", 1);
        robots_[i].odom_sub = nh_.subscribe<nav_msgs::Odometry>(
            robots_[i].odom_topic,
            1,
            boost::bind(&SensorSimulator::odomCallback, this, _1, i),
            ros::VoidPtr(),
            ros::TransportHints().tcpNoDelay());
        robots_[i].next_depth_pub_time = now;
        robots_[i].next_lidar_pub_time = now;
    }
}

std::vector<EllipsoidObstacle> SensorSimulator::getDynamicObstacles(size_t robot_index) const
{
    std::vector<EllipsoidObstacle> obstacles;
    if (!swarm_enabled_ || robots_.size() <= 1)
    {
        if (!swarm_enabled_ && target_odom_init_)
        {
            EllipsoidObstacle obstacle;
            obstacle.center = Vector3f(target_pos_.x(), target_pos_.y(), target_pos_.z());
            obstacle.radii = Vector3f(
                target_ellipsoid_axes_.x(),
                target_ellipsoid_axes_.y(),
                target_ellipsoid_axes_.z());
            obstacles.push_back(obstacle);
        }
        return obstacles;
    }

    obstacles.reserve(robots_.size() - 1);
    for (size_t i = 0; i < robots_.size(); ++i)
    {
        if (i == robot_index || !robots_[i].odom_init)
            continue;

        EllipsoidObstacle obstacle;
        obstacle.center = Vector3f(robots_[i].pos.x(), robots_[i].pos.y(), robots_[i].pos.z());
        obstacle.radii = Vector3f(
            target_ellipsoid_axes_.x(),
            target_ellipsoid_axes_.y(),
            target_ellipsoid_axes_.z());
        obstacles.push_back(obstacle);
    }
    return obstacles;
}

bool SensorSimulator::inStaticCollision(const RobotChannels &robot) const
{
    return grid_map_->mapQueryHost(Vector3f(robot.pos.x(), robot.pos.y(), robot.pos.z())) == 1;
}

bool SensorSimulator::inDynamicCollision(size_t robot_index) const
{
    if (!swarm_enabled_)
        return false;

    for (size_t i = 0; i < robots_.size(); ++i)
    {
        if (i == robot_index || !robots_[i].odom_init)
            continue;

        if ((robots_[robot_index].pos - robots_[i].pos).norm() <= 2.0f * collision_radius_)
            return true;
    }
    return false;
}

bool SensorSimulator::inTargetStaticCollision() const
{
    if (!target_odom_init_)
        return false;
    return grid_map_->mapQueryHost(Vector3f(target_pos_.x(), target_pos_.y(), target_pos_.z())) == 1;
}

bool SensorSimulator::projectTarget(const RobotChannels &robot, const Eigen::Vector3f &target_pos,
                                    Eigen::Vector3f &target_c, float &u_center, float &v_center,
                                    int &radius_u_px, int &radius_v_px) const
{
    const Eigen::Matrix3f R_cw = robot.quat_wc.toRotationMatrix().transpose();
    target_c = R_cw * (target_pos - robot.pos);
    if (target_c.x() <= target_mask_min_forward_depth_ || target_c.x() > camera_->max_depth_dist)
        return false;

    u_center = camera_->cx - camera_->fx * target_c.y() / target_c.x();
    v_center = camera_->cy - camera_->fy * target_c.z() / target_c.x();
    radius_u_px = std::max(2, static_cast<int>(std::ceil(camera_->fx * target_ellipsoid_axes_.y() / target_c.x())));
    radius_v_px = std::max(2, static_cast<int>(std::ceil(camera_->fy * target_ellipsoid_axes_.z() / target_c.x())));
    const int u0 = std::max(0, static_cast<int>(std::floor(u_center)) - radius_u_px);
    const int u1 = std::min(camera_->image_width - 1, static_cast<int>(std::ceil(u_center)) + radius_u_px);
    const int v0 = std::max(0, static_cast<int>(std::floor(v_center)) - radius_v_px);
    const int v1 = std::min(camera_->image_height - 1, static_cast<int>(std::ceil(v_center)) + radius_v_px);
    return u0 <= u1 && v0 <= v1;
}

bool SensorSimulator::targetVisibleInDepth(const cv::Mat &depth_image, const Eigen::Vector3f &target_c,
                                           float u_center, float v_center,
                                           int radius_u_px, int radius_v_px) const
{
    const int u0 = std::max(0, static_cast<int>(std::floor(u_center)) - radius_u_px);
    const int u1 = std::min(camera_->image_width - 1, static_cast<int>(std::ceil(u_center)) + radius_u_px);
    const int v0 = std::max(0, static_cast<int>(std::floor(v_center)) - radius_v_px);
    const int v1 = std::min(camera_->image_height - 1, static_cast<int>(std::ceil(v_center)) + radius_v_px);

    int projected_pixels = 0;
    int visible_pixels = 0;

    for (int v = v0; v <= v1; ++v)
    {
        for (int u = u0; u <= u1; ++u)
        {
            const float y_at_target_x = -(u - camera_->cx) / camera_->fx * target_c.x();
            const float z_at_target_x = -(v - camera_->cy) / camera_->fy * target_c.x();
            const float y_norm = (y_at_target_x - target_c.y()) / target_ellipsoid_axes_.y();
            const float z_norm = (z_at_target_x - target_c.z()) / target_ellipsoid_axes_.z();
            const float lateral_norm_sq = y_norm * y_norm + z_norm * z_norm;
            if (lateral_norm_sq > 1.0f)
                continue;

            ++projected_pixels;
            const float surface_depth = target_c.x() - target_ellipsoid_axes_.x() *
                std::sqrt(std::max(0.0f, 1.0f - lateral_norm_sq));
            const float depth = depth_image.at<float>(v, u);
            if (!std::isfinite(depth) || depth <= 0.0f)
                continue;
            if (depth >= surface_depth - target_occlusion_margin_)
                ++visible_pixels;
        }
    }

    if (projected_pixels <= 0)
        return false;
    if (visible_pixels < mask_min_visible_pixels_)
        return false;
    return static_cast<float>(visible_pixels) / static_cast<float>(projected_pixels) >= mask_min_visible_ratio_;
}

void SensorSimulator::drawTargetMask(cv::Mat &target_mask, float u_center, float v_center,
                                     int radius_u_px, int radius_v_px) const
{
    cv::ellipse(
        target_mask,
        cv::Point(static_cast<int>(std::round(u_center)), static_cast<int>(std::round(v_center))),
        cv::Size(std::max(2, radius_u_px), std::max(2, radius_v_px)),
        0.0,
        0.0,
        360.0,
        cv::Scalar(255),
        cv::FILLED);
}

void SensorSimulator::overlaySingleTargetAndMask(RobotChannels &robot, cv::Mat &depth_image, cv::Mat &target_mask) const
{
    if (!target_odom_init_)
        return;

    Eigen::Vector3f target_c;
    float u_center = 0.0f;
    float v_center = 0.0f;
    int radius_u_px = 0;
    int radius_v_px = 0;
    if (!projectTarget(robot, target_pos_, target_c, u_center, v_center, radius_u_px, radius_v_px))
        return;
    if (!targetVisibleInDepth(depth_image, target_c, u_center, v_center, radius_u_px, radius_v_px))
        return;

    drawTargetMask(target_mask, u_center, v_center, radius_u_px, radius_v_px);
}

void SensorSimulator::overlaySwarmDetectorMasks(size_t robot_index, const cv::Mat &depth_image, cv::Mat &target_mask) const
{
    if (!swarm_enabled_ || robot_index >= robots_.size())
        return;
    const auto &robot = robots_[robot_index];
    if (!robot.odom_init)
        return;

    for (size_t i = 0; i < robots_.size(); ++i)
    {
        if (i == robot_index || !robots_[i].odom_init)
            continue;

        Eigen::Vector3f target_c;
        float u_center = 0.0f;
        float v_center = 0.0f;
        int radius_u_px = 0;
        int radius_v_px = 0;
        if (!projectTarget(robot, robots_[i].pos, target_c, u_center, v_center, radius_u_px, radius_v_px))
            continue;
        if (!targetVisibleInDepth(depth_image, target_c, u_center, v_center, radius_u_px, radius_v_px))
            continue;

        drawTargetMask(target_mask, u_center, v_center, radius_u_px, radius_v_px);
    }
}

void SensorSimulator::renderDepthCallback(size_t robot_index, const ros::Time &stamp)
{
    if (!render_depth_ || robot_index >= robots_.size())
        return;

    auto &robot = robots_[robot_index];
    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(robot.quat_wc.w(), robot.quat_wc.x(), robot.quat_wc.y(), robot.quat_wc.z(),
                             robot.pos.x(), robot.pos.y(), robot.pos.z());
    const auto dynamic_obstacles = getDynamicObstacles(robot_index);
    cv::Mat depth_image;
    renderDepthImage(grid_map_, camera_, T_wc, depth_image, dynamic_obstacles);

    cv::Mat target_mask;
    if (render_target_mask_)
    {
        target_mask = cv::Mat::zeros(camera_->image_height, camera_->image_width, CV_8UC1);
        if (!swarm_enabled_)
        {
            overlaySingleTargetAndMask(robot, depth_image, target_mask);
        }
        else if (render_swarm_detector_mask_)
        {
            overlaySwarmDetectorMasks(robot_index, depth_image, target_mask);
        }
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
    robot.image_pub.publish(ros_image);

    if (!swarm_enabled_ && target_odom_init_ && !robot.target_depth_topic.empty())
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
        robot.target_depth_pub.publish(target_depth_msg);
    }

    if (render_target_mask_)
    {
        sensor_msgs::Image mask_msg;
        cv_bridge::CvImage mask_bridge;
        mask_bridge.header.stamp = stamp;
        mask_bridge.encoding = sensor_msgs::image_encodings::MONO8;
        mask_bridge.image = target_mask;
        mask_bridge.toImageMsg(mask_msg);
        robot.target_mask_pub.publish(mask_msg);
    }
}

void SensorSimulator::renderLidarCallback(size_t robot_index, const ros::Time &stamp)
{
    if (!render_lidar_ || robot_index >= robots_.size())
        return;

    const auto dynamic_obstacles = getDynamicObstacles(robot_index);
    auto start = std::chrono::high_resolution_clock::now();

    auto &robot = robots_[robot_index];
    cudaMat::SE3<float> T_wc(robot.quat.w(), robot.quat.x(), robot.quat.y(), robot.quat.z(),
                             robot.pos.x(), robot.pos.y(), robot.pos.z());
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    renderLidarPointcloud(grid_map_, lidar_, T_wc, lidar_points, dynamic_obstacles);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    lidar_time_ += elapsed.count();
    lidar_count_++;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = stamp;
    output.header.frame_id = "odom";
    robot.point_cloud_pub.publish(output);

    if (!visualize_local_map_)
    {
        robot.local_map_world.clear();
        return;
    }

    robot.local_map_world.clear();
    robot.local_map_world.points.reserve(lidar_points.points.size());
    for (const auto &point_local : lidar_points.points)
    {
        const float3 point_world = T_wc * make_float3(point_local.x, point_local.y, point_local.z);
        robot.local_map_world.points.emplace_back(point_world.x, point_world.y, point_world.z);
    }
    robot.local_map_world.width = robot.local_map_world.points.size();
    robot.local_map_world.height = 1;
    robot.local_map_world.is_dense = true;
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

    pcl::PointCloud<pcl::PointXYZ> merged_cloud;
    for (const auto &robot : robots_)
        merged_cloud += robot.local_map_world;
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(merged_cloud, output);
    output.header.stamp = stamp;
    output.header.frame_id = "world";
    local_map_visual_pub_.publish(output);
}

void SensorSimulator::publishCollisionCounter(size_t robot_index)
{
    if (robot_index >= robots_.size())
        return;

    std_msgs::Int32 msg;
    msg.data = robots_[robot_index].collision_counter;
    robots_[robot_index].collision_pub.publish(msg);

    std_msgs::Int32 uav_msg;
    uav_msg.data = robots_[robot_index].uav_collision_counter;
    robots_[robot_index].uav_collision_pub.publish(uav_msg);
}

void SensorSimulator::publishCollisionCounterTotals()
{
    std_msgs::Int32 total_msg;
    total_msg.data = collision_counter_total_;
    collision_counter_total_pub_.publish(total_msg);

    std_msgs::Int32 target_msg;
    target_msg.data = target_collision_counter_total_;
    target_collision_counter_total_pub_.publish(target_msg);

    std_msgs::Int32 uav_total_msg;
    uav_total_msg.data = uav_collision_counter_total_;
    uav_collision_counter_total_pub_.publish(uav_total_msg);
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

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr &msg, size_t robot_index)
{
    if (robot_index >= robots_.size())
        return;
    auto &robot = robots_[robot_index];

    robot.quat.x() = msg->pose.pose.orientation.x;
    robot.quat.y() = msg->pose.pose.orientation.y;
    robot.quat.z() = msg->pose.pose.orientation.z;
    robot.quat.w() = msg->pose.pose.orientation.w;
    robot.quat_wc = robot.quat * quat_bc_;

    robot.pos.x() = msg->pose.pose.position.x;
    robot.pos.y() = msg->pose.pose.position.y;
    robot.pos.z() = msg->pose.pose.position.z;
    robot.odom_init = true;

    const bool static_collision = inStaticCollision(robot);
    const bool dynamic_collision = inDynamicCollision(robot_index);
    if (static_collision && !robot.in_static_collision)
    {
        robot.collision_counter += 1;
        collision_counter_total_ += 1;
        ROS_WARN_THROTTLE(1.0, "[%s] occupied-voxel collision detected. robot_total=%d all_total=%d",
                          robot.name.c_str(), robot.collision_counter, collision_counter_total_);
    }
    if (dynamic_collision && !robot.in_uav_collision)
    {
        robot.uav_collision_counter += 1;
        uav_collision_counter_total_ += 1;
        ROS_WARN_THROTTLE(1.0, "[%s] UAV-UAV collision detected. robot_total=%d all_total=%d",
                          robot.name.c_str(), robot.uav_collision_counter, uav_collision_counter_total_);
    }
    robot.in_static_collision = static_collision;
    robot.in_uav_collision = dynamic_collision;

    if (!swarm_enabled_)
    {
        const bool target_static_collision = inTargetStaticCollision();
        if (target_static_collision && !target_in_static_collision_)
        {
            target_collision_counter_total_ += 1;
            ROS_WARN_THROTTLE(1.0, "Target occupied-voxel collision detected. total=%d", target_collision_counter_total_);
        }
        target_in_static_collision_ = target_static_collision;
    }

    publishCollisionCounter(robot_index);
    publishCollisionCounterTotals();

    const ros::Time tnow = ros::Time::now();
    if (fabs((tnow - robot.next_depth_pub_time).toSec()) > 10 * depth_pub_duration_.toSec())
        robot.next_depth_pub_time = tnow;
    if (fabs((tnow - robot.next_lidar_pub_time).toSec()) > 10 * lidar_pub_duration_.toSec())
        robot.next_lidar_pub_time = tnow;

    if (tnow >= robot.next_depth_pub_time)
    {
        robot.next_depth_pub_time += depth_pub_duration_;
        renderDepthCallback(robot_index, msg->header.stamp);
    }
    if (tnow >= robot.next_lidar_pub_time)
    {
        robot.next_lidar_pub_time += lidar_pub_duration_;
        renderLidarCallback(robot_index, msg->header.stamp);
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
