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
#include <array>
#include <chrono>
#include <iostream>
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
        visualize_local_map_ = config["visualize_local_map"] ? config["visualize_local_map"].as<bool>() : true;
        depth_pub_duration_ = ros::Duration(1.0 / config["depth_fps"].as<float>());
        lidar_pub_duration_ = ros::Duration(1.0 / config["lidar_fps"].as<float>());

        swarm_enabled_ = config["swarm"]["enabled"].as<bool>();
        swarm_uav_num_ = std::max(1, config["swarm"]["uav_num"].as<int>());
        swarm_namespace_prefix_ = config["swarm"]["namespace_prefix"].as<std::string>();
        collision_radius_ = config["swarm"]["collision_radius"].as<float>();

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
        timer_map_ = nh_.createTimer(ros::Duration(1.0), &SensorSimulator::timerMapCallback, this);

        printf("3.Simulation Ready! robots=%zu swarm=%s\n", robots_.size(), swarm_enabled_ ? "true" : "false");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg, size_t robot_index);
    void renderDepthCallback(size_t robot_index, const ros::Time &stamp);
    void renderLidarCallback(size_t robot_index, const ros::Time &stamp);
    void timerMapCallback(const ros::TimerEvent &);
    void publishCollisionCounterTotal();

private:
    struct RobotChannels
    {
        std::string name;
        std::string odom_topic;
        std::string depth_topic;
        std::string lidar_topic;

        ros::Publisher image_pub;
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
    std::vector<SphereObstacle> getDynamicObstacles(size_t robot_index) const;
    bool inStaticCollision(const RobotChannels &robot) const;
    bool inDynamicCollision(size_t robot_index) const;
    void publishCollisionCounter(size_t robot_index);
    void publishLocalMapVisual(const ros::Time &stamp);

    bool render_depth_{false};
    bool render_lidar_{false};
    bool visualize_local_map_{true};
    bool swarm_enabled_{false};
    int swarm_uav_num_{1};
    float collision_radius_{0.45f};
    std::string swarm_namespace_prefix_{"uav"};

    Eigen::Quaternionf quat_bc_{Eigen::Quaternionf::Identity()};
    CameraParams *camera_{nullptr};
    LidarParams *lidar_{nullptr};
    GridMap *grid_map_{nullptr};

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher pcl_pub_;
    ros::Publisher local_map_visual_pub_;
    ros::Publisher collision_counter_total_pub_;
    ros::Publisher uav_collision_counter_total_pub_;
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
    int uav_collision_counter_total_{0};
};

void SensorSimulator::applyRosParamOverrides(YAML::Node &config)
{
    if (!config["swarm"])
        config["swarm"] = YAML::Node(YAML::NodeType::Map);

    bool swarm_enabled = config["swarm"]["enabled"] ? config["swarm"]["enabled"].as<bool>() : false;
    int swarm_uav_num = config["swarm"]["uav_num"] ? config["swarm"]["uav_num"].as<int>() : 1;
    std::string namespace_prefix = config["swarm"]["namespace_prefix"] ? config["swarm"]["namespace_prefix"].as<std::string>() : "uav";
    double ring_radius = config["swarm"]["ring_radius"] ? config["swarm"]["ring_radius"].as<double>() : 8.0;
    double altitude = config["swarm"]["altitude"] ? config["swarm"]["altitude"].as<double>() : 2.0;
    double spawn_clear_radius = config["swarm"]["spawn_clear_radius"] ? config["swarm"]["spawn_clear_radius"].as<double>() : 2.2;
    double collision_radius = config["swarm"]["collision_radius"] ? config["swarm"]["collision_radius"].as<double>() : 0.45;

    pnh_.param("swarm_enabled", swarm_enabled, swarm_enabled);
    pnh_.param("swarm_uav_num", swarm_uav_num, swarm_uav_num);
    pnh_.param("swarm_namespace_prefix", namespace_prefix, namespace_prefix);
    pnh_.param("swarm_ring_radius", ring_radius, ring_radius);
    pnh_.param("swarm_altitude", altitude, altitude);
    pnh_.param("swarm_spawn_clear_radius", spawn_clear_radius, spawn_clear_radius);
    pnh_.param("swarm_collision_radius", collision_radius, collision_radius);

    bool visualize_local_map = config["visualize_local_map"] ? config["visualize_local_map"].as<bool>() : true;
    pnh_.param("visualize_local_map", visualize_local_map, visualize_local_map);

    config["swarm"]["enabled"] = swarm_enabled;
    config["swarm"]["uav_num"] = swarm_uav_num;
    config["swarm"]["namespace_prefix"] = namespace_prefix;
    config["swarm"]["ring_radius"] = ring_radius;
    config["swarm"]["altitude"] = altitude;
    config["swarm"]["spawn_clear_radius"] = spawn_clear_radius;
    config["swarm"]["collision_radius"] = collision_radius;
    config["visualize_local_map"] = visualize_local_map;
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
        robot.lidar_topic = config["lidar_topic"].as<std::string>();
        robots_.push_back(robot);
    }

    const ros::Time now = ros::Time::now();
    for (size_t i = 0; i < robots_.size(); ++i)
    {
        robots_[i].image_pub = nh_.advertise<sensor_msgs::Image>(robots_[i].depth_topic, 1);
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

std::vector<SphereObstacle> SensorSimulator::getDynamicObstacles(size_t robot_index) const
{
    std::vector<SphereObstacle> obstacles;
    if (!swarm_enabled_ || robots_.size() <= 1)
        return obstacles;

    obstacles.reserve(robots_.size() - 1);
    for (size_t i = 0; i < robots_.size(); ++i)
    {
        if (i == robot_index || !robots_[i].odom_init)
            continue;

        SphereObstacle obstacle;
        obstacle.center = Vector3f(robots_[i].pos.x(), robots_[i].pos.y(), robots_[i].pos.z());
        obstacle.radius = collision_radius_;
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

void SensorSimulator::renderDepthCallback(size_t robot_index, const ros::Time &stamp)
{
    if (!render_depth_)
        return;

    const auto dynamic_obstacles = getDynamicObstacles(robot_index);
    auto start = std::chrono::high_resolution_clock::now();

    auto &robot = robots_[robot_index];
    cudaMat::SE3<float> T_wc(robot.quat_wc.w(), robot.quat_wc.x(), robot.quat_wc.y(), robot.quat_wc.z(),
                             robot.pos.x(), robot.pos.y(), robot.pos.z());
    cv::Mat depth_image;
    renderDepthImage(grid_map_, camera_, T_wc, depth_image, dynamic_obstacles);

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
}

void SensorSimulator::renderLidarCallback(size_t robot_index, const ros::Time &stamp)
{
    if (!render_lidar_)
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

    pcl::PointCloud<pcl::PointXYZ> combined_cloud;
    size_t total_points = 0;
    for (const auto &robot : robots_)
        total_points += robot.local_map_world.points.size();
    combined_cloud.points.reserve(total_points);

    for (const auto &robot : robots_)
        combined_cloud += robot.local_map_world;

    combined_cloud.width = combined_cloud.points.size();
    combined_cloud.height = 1;
    combined_cloud.is_dense = true;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(combined_cloud, output);
    output.header.stamp = stamp;
    output.header.frame_id = "world";
    local_map_visual_pub_.publish(output);
}

void SensorSimulator::publishCollisionCounter(size_t robot_index)
{
    std_msgs::Int32 msg;
    msg.data = robots_[robot_index].collision_counter;
    robots_[robot_index].collision_pub.publish(msg);

    std_msgs::Int32 uav_msg;
    uav_msg.data = robots_[robot_index].uav_collision_counter;
    robots_[robot_index].uav_collision_pub.publish(uav_msg);
}

void SensorSimulator::publishCollisionCounterTotal()
{
    std_msgs::Int32 total_msg;
    total_msg.data = collision_counter_total_;
    collision_counter_total_pub_.publish(total_msg);

    std_msgs::Int32 uav_total_msg;
    uav_total_msg.data = uav_collision_counter_total_;
    uav_collision_counter_total_pub_.publish(uav_total_msg);
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr &msg, size_t robot_index)
{
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
    publishCollisionCounter(robot_index);
    publishCollisionCounterTotal();

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
