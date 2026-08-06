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
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "sensor_simulator.cuh"
#include <chrono>
#include "maps.hpp"

using namespace raycast;

float weightedQuantileDepth(std::vector<std::pair<float, float>> depth_weights, float quantile)
{
    if (depth_weights.empty())
        return 0.0f;
    std::sort(depth_weights.begin(), depth_weights.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    float total_weight = 0.0f;
    for (const auto &item : depth_weights)
        total_weight += item.second;
    if (total_weight <= 1e-6f)
        return depth_weights.front().first;

    float target = std::clamp(quantile, 0.0f, 1.0f) * total_weight;
    float accum = 0.0f;
    for (const auto &item : depth_weights)
    {
        accum += item.second;
        if (accum >= target)
            return item.first;
    }
    return depth_weights.back().first;
}

void renderToFSenseMImage(GridMap *grid_map,
                          const CameraParams &tof_camera,
                          cudaMat::SE3<float> &T_wc,
                          int zone_subsample,
                          float depth_quantile,
                          float noise_std,
                          float far_noise_std,
                          float signal_floor,
                          float min_depth,
                          std::default_random_engine &generator,
                          cv::Mat &tof_image)
{
    const int subsample = std::max(1, zone_subsample);
    CameraParams sub_camera = tof_camera;
    const float tan_half_x = (0.5f * tof_camera.image_width) / tof_camera.fx;
    const float tan_half_y = (0.5f * tof_camera.image_height) / tof_camera.fy;
    sub_camera.image_width = tof_camera.image_width * subsample;
    sub_camera.image_height = tof_camera.image_height * subsample;
    sub_camera.fx = (0.5f * sub_camera.image_width) / tan_half_x;
    sub_camera.fy = (0.5f * sub_camera.image_height) / tan_half_y;
    sub_camera.cx = 0.5f * (sub_camera.image_width - 1);
    sub_camera.cy = 0.5f * (sub_camera.image_height - 1);

    cv::Mat sub_depth;
    renderDepthImage(grid_map, &sub_camera, T_wc, sub_depth);

    tof_image.create(tof_camera.image_height, tof_camera.image_width, CV_32FC1);
    std::normal_distribution<float> normal_distribution(0.0f, 1.0f);
    const float max_depth = tof_camera.max_depth_dist;

    for (int zone_v = 0; zone_v < tof_camera.image_height; ++zone_v)
        for (int zone_u = 0; zone_u < tof_camera.image_width; ++zone_u)
        {
            std::vector<std::pair<float, float>> depth_weights;
            depth_weights.reserve(subsample * subsample);
            for (int sv = 0; sv < subsample; ++sv)
                for (int su = 0; su < subsample; ++su)
                {
                    float depth = sub_depth.at<float>(zone_v * subsample + sv, zone_u * subsample + su);
                    depth = std::clamp(depth, min_depth, max_depth);
                    float normalized = depth / max_depth;
                    float signal = depth >= max_depth - 1e-4f
                                       ? signal_floor
                                       : std::max(signal_floor, 1.0f / (depth * depth + 0.05f));
                    signal *= std::max(0.05f, 1.0f - 0.35f * normalized * normalized);
                    depth_weights.emplace_back(depth, signal);
                }

            float depth = weightedQuantileDepth(depth_weights, depth_quantile);
            if (depth < max_depth - 1e-4f)
            {
                float normalized = depth / max_depth;
                float sigma = noise_std + far_noise_std * normalized * normalized;
                depth += sigma * normal_distribution(generator);
            }
            depth = std::clamp(depth, min_depth, max_depth);
            depth = std::round(depth * 1000.0f) / 1000.0f;
            tof_image.at<float>(zone_v, zone_u) = depth;
        }
}

class SensorSimulator {
public:
    SensorSimulator(ros::NodeHandle &nh) : nh_(nh) {
        YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
        // 读取前向高清debug相机参数
        debug_camera = new CameraParams();
        debug_camera->fx = config["camera"]["fx"].as<float>();
        debug_camera->fy = config["camera"]["fy"].as<float>();
        debug_camera->cx = config["camera"]["cx"].as<float>();
        debug_camera->cy = config["camera"]["cy"].as<float>();
        debug_camera->image_width = config["camera"]["image_width"].as<int>();
        debug_camera->image_height = config["camera"]["image_height"].as<int>();
        debug_camera->max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
        debug_camera->normalize_depth = config["camera"]["normalize_depth"].as<bool>();
        debug_camera_pitch_rad = config["camera"]["pitch"].as<float>() * M_PI / 180.0f;

        // 读取Nooploop TOFSense-M等效ToF参数，四向8x8 depth pixels作为网络输入
        YAML::Node tof_config = config["tof"] ? config["tof"] : config["camera"];
        tof_camera = new CameraParams();
        tof_camera->fx = tof_config["fx"].as<float>();
        tof_camera->fy = tof_config["fy"].as<float>();
        tof_camera->cx = tof_config["cx"].as<float>();
        tof_camera->cy = tof_config["cy"].as<float>();
        tof_camera->image_width = tof_config["image_width"].as<int>();
        tof_camera->image_height = tof_config["image_height"].as<int>();
        tof_camera->max_depth_dist = tof_config["max_depth_dist"].as<float>();
        tof_camera->normalize_depth = tof_config["normalize_depth"].as<bool>();
        tof_pitch_rad = tof_config["pitch"].as<float>() * M_PI / 180.0f;
        tof_zone_subsample_ = tof_config["zone_subsample"] ? tof_config["zone_subsample"].as<int>() : 4;
        tof_depth_quantile_ = tof_config["depth_quantile"] ? tof_config["depth_quantile"].as<float>() : 0.35f;
        tof_noise_std_ = tof_config["noise_std"] ? tof_config["noise_std"].as<float>() : 0.015f;
        tof_far_noise_std_ = tof_config["far_noise_std"] ? tof_config["far_noise_std"].as<float>() : 0.08f;
        tof_signal_floor_ = tof_config["signal_floor"] ? tof_config["signal_floor"].as<float>() : 0.08f;
        tof_min_depth_ = tof_config["min_depth_dist"] ? tof_config["min_depth_dist"].as<float>() : 0.015f;

        // 读取lidar参数
        lidar = new LidarParams();
        lidar->vertical_lines = config["lidar"]["vertical_lines"].as<int>();
        lidar->vertical_angle_start = config["lidar"]["vertical_angle_start"].as<float>();
        lidar->vertical_angle_end = config["lidar"]["vertical_angle_end"].as<float>();
        lidar->horizontal_num = config["lidar"]["horizontal_num"].as<int>();
        lidar->horizontal_resolution = config["lidar"]["horizontal_resolution"].as<float>();
        lidar->max_lidar_dist = config["lidar"]["max_lidar_dist"].as<float>();

        render_lidar = config["render_lidar"].as<bool>();
        render_depth = config["render_depth"].as<bool>();
        float depth_fps = config["depth_fps"].as<float>();
        float lidar_fps = config["lidar_fps"].as<float>();
        depth_pub_duration = ros::Duration(1 / depth_fps);
        lidar_pub_duration = ros::Duration(1 / lidar_fps);
        
        std::string ply_file = config["ply_file"].as<std::string>();
        std::string odom_topic = config["odom_topic"].as<std::string>();
        std::string depth_topic = config["depth_topic"].as<std::string>();
        std::string lidar_topic = config["lidar_topic"].as<std::string>();

        // 读取地图参数
        bool use_random_map = config["random_map"].as<bool>();
        float resolution = config["resolution"].as<float>();
        int occupy_threshold = config["occupy_threshold"].as<int>();
        pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("mock_map", 1);
        int seed = config["seed"].as<int>();
        int sizeX = config["x_length"].as<int>();
        int sizeY = config["y_length"].as<int>();
        int sizeZ = config["z_length"].as<int>();
        int type = config["maze_type"].as<int>();
        double scale = 1 / resolution;
        sizeX = sizeX * scale;
        sizeY = sizeY * scale;
        sizeZ = sizeZ * scale;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (use_random_map) {
            printf("1.Generate Random Map... \n");
            mocka::Maps::BasicInfo info;
            info.sizeX      = sizeX;
            info.sizeY      = sizeY;
            info.sizeZ      = sizeZ;
            info.seed       = seed;
            info.scale      = scale;
            info.cloud      = cloud;

            mocka::Maps map;
            map.setParam(config);
            map.setInfo(info);
            map.generate(type);
        }
        else {
            printf("1.Reading Point Cloud %s... \n", ply_file.c_str());
            if (pcl::io::loadPLYFile(ply_file, *cloud) == -1) {
                PCL_ERROR("Couldn't read PLY file \n");
            }
        }
        float map_viz_resolution = config["map_viz_resolution"] ? config["map_viz_resolution"].as<float>() : 0.2f;
        pcl::PointCloud<pcl::PointXYZ>::Ptr viz_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(map_viz_resolution, map_viz_resolution, map_viz_resolution);
        voxel_filter.filter(*viz_cloud);
        pcl::toROSMsg(*viz_cloud, output);
        output.header.frame_id = "world";

        std::cout<<"Pointloud size:"<<cloud->points.size()<<std::endl;
        std::cout<<"Map visualization pointcloud size:"<<viz_cloud->points.size()<<std::endl;
        printf("2.Mapping... \n");
        grid_map = new GridMap(cloud, resolution, occupy_threshold);
        
        ros::Time next_depth_pub_time = ros::Time::now();
        ros::Time next_lidar_pub_time = ros::Time::now();

        // ROS
        image_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic, 1);
        const std::array<std::string, 4> view_names = {"front", "left", "right", "back"};
        for (const auto &view_name : view_names)
            image_pubs_.push_back(nh_.advertise<sensor_msgs::Image>(depth_topic + "_" + view_name, 1));
        point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_topic, 1);
        collision_counter_total_pub_ = nh_.advertise<std_msgs::Int32>("/yopo/collision_counter_total", 1);
        odom_sub_ = nh_.subscribe(odom_topic, 1, &SensorSimulator::odomCallback, this, ros::TransportHints().tcpNoDelay());
        timer_map_   = nh_.createTimer(ros::Duration(1), &SensorSimulator::timerMapCallback, this);

        printf("3.Simulation Ready! \n");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);

    void renderDepthCallback(const ros::Time stamp);

    void renderLidarCallback(const ros::Time stamp);

    void timerMapCallback(const ros::TimerEvent &);

    void publishCollisionCounterTotal();

private:
    bool render_depth{false};
    bool render_lidar{false};
    Eigen::Quaternionf quat;
    Eigen::Quaternionf quat_bc, quat_wc;
    float debug_camera_pitch_rad{0.0f};
    float tof_pitch_rad{0.0f};
    Eigen::Vector3f pos;

    CameraParams* debug_camera;
    CameraParams* tof_camera;
    LidarParams* lidar;
    GridMap* grid_map;
    sensor_msgs::PointCloud2 output;
    
    ros::NodeHandle nh_;
    ros::Publisher image_pub_, point_cloud_pub_;
    std::vector<ros::Publisher> image_pubs_;
    ros::Publisher pcl_pub;
    ros::Publisher collision_counter_total_pub_;
    ros::Subscriber odom_sub_;
    ros::Timer timer_depth_, timer_lidar_, timer_map_;

    ros::Time next_depth_pub_time, next_lidar_pub_time;
    ros::Duration depth_pub_duration, lidar_pub_duration;
    double depth_time{0.0}, lidar_time{0.0};
    int depth_count{0}, lidar_count{0};
    int collision_counter_total_{0};
    int tof_zone_subsample_{4};
    float tof_depth_quantile_{0.35f};
    float tof_noise_std_{0.015f};
    float tof_far_noise_std_{0.08f};
    float tof_signal_floor_{0.08f};
    float tof_min_depth_{0.015f};
    std::default_random_engine tof_noise_generator_{3};
    // mocka::Maps map;
};



void SensorSimulator::renderDepthCallback(const ros::Time stamp) {
    if (!render_depth)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    const std::array<float, 4> view_yaws = {0.0f, M_PI / 2.0f, -M_PI / 2.0f, M_PI};
    for (size_t i = 0; i < view_yaws.size(); ++i) {
        Eigen::AngleAxisf yaw_view(view_yaws[i], Eigen::Vector3f::UnitZ());
        Eigen::AngleAxisf pitch_view(tof_pitch_rad, Eigen::Vector3f::UnitY());
        Eigen::Quaternionf quat_wc_view = quat * Eigen::Quaternionf(yaw_view * pitch_view);
        cudaMat::SE3<float> T_wc(quat_wc_view.w(), quat_wc_view.x(), quat_wc_view.y(), quat_wc_view.z(),
                                  pos.x(), pos.y(), pos.z());
        cv::Mat depth_image;
        renderToFSenseMImage(grid_map, *tof_camera, T_wc, tof_zone_subsample_,
                             tof_depth_quantile_, tof_noise_std_, tof_far_noise_std_,
                             tof_signal_floor_, tof_min_depth_, tof_noise_generator_, depth_image);

        sensor_msgs::Image ros_image;
        cv_bridge::CvImage cv_image;
        cv_image.header.stamp = stamp;
        cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        cv_image.image = depth_image;
        cv_image.toImageMsg(ros_image);
        image_pubs_[i].publish(ros_image);
    }

    Eigen::AngleAxisf debug_pitch_view(debug_camera_pitch_rad, Eigen::Vector3f::UnitY());
    Eigen::Quaternionf quat_wc_debug = quat * Eigen::Quaternionf(debug_pitch_view);
    cudaMat::SE3<float> T_wc_debug(quat_wc_debug.w(), quat_wc_debug.x(), quat_wc_debug.y(), quat_wc_debug.z(),
                                   pos.x(), pos.y(), pos.z());
    cv::Mat debug_depth_image;
    renderDepthImage(grid_map, debug_camera, T_wc_debug, debug_depth_image);

    sensor_msgs::Image debug_ros_image;
    cv_bridge::CvImage debug_cv_image;
    debug_cv_image.header.stamp = stamp;
    debug_cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    debug_cv_image.image = debug_depth_image;
    debug_cv_image.toImageMsg(debug_ros_image);
    image_pub_.publish(debug_ros_image);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    depth_time += elapsed.count();
    depth_count++;
    // std::cout << "生成图像耗时: " << elapsed.count() << " 秒" << std::endl;

}

void SensorSimulator::timerMapCallback(const ros::TimerEvent&) {
    if (pcl_pub.getNumSubscribers() > 0) {
        output.header.stamp = ros::Time::now();
        pcl_pub.publish(output);    
    }
}

void SensorSimulator::publishCollisionCounterTotal() {
    std_msgs::Int32 total_msg;
    total_msg.data = collision_counter_total_;
    collision_counter_total_pub_.publish(total_msg);
}

void SensorSimulator::renderLidarCallback(const ros::Time stamp) {
    if (!render_lidar)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat.w(), quat.x(), quat.y(), quat.z(), pos.x(), pos.y(), pos.z());
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    renderLidarPointcloud(grid_map, lidar, T_wc, lidar_points);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    lidar_time += elapsed.count();
    lidar_count++;
    // std::cout << "生成雷达耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = stamp;
    output.header.frame_id = "world";
    point_cloud_pub_.publish(output);
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    quat.x() = msg->pose.pose.orientation.x;
    quat.y() = msg->pose.pose.orientation.y;
    quat.z() = msg->pose.pose.orientation.z;
    quat.w() = msg->pose.pose.orientation.w;
    quat_wc = quat * quat_bc;

    pos.x() = msg->pose.pose.position.x;
    pos.y() = msg->pose.pose.position.y;
    pos.z() = msg->pose.pose.position.z;

    const int occupied = grid_map->mapQueryHost(Vector3f(pos.x(), pos.y(), pos.z()));
    if (occupied == 1) {
        collision_counter_total_ += 1;
        ROS_WARN_THROTTLE(1.0, "UAV is inside occupied voxel. total=%d", collision_counter_total_);
    }
    publishCollisionCounterTotal();

    ros::Time tnow = ros::Time::now();

    // 避免仿真odom消息中断，导致时间差太大
    if (fabs((tnow - next_depth_pub_time).toSec()) > 10 * depth_pub_duration.toSec())
        next_depth_pub_time = tnow;
    if (fabs((tnow - next_lidar_pub_time).toSec()) > 10 * lidar_pub_duration.toSec())
        next_lidar_pub_time = tnow;

    if (tnow >= next_depth_pub_time){
        next_depth_pub_time += depth_pub_duration;
        renderDepthCallback(msg->header.stamp);
    }
    if (tnow >= next_lidar_pub_time){
        next_lidar_pub_time += lidar_pub_duration;
        renderLidarCallback(msg->header.stamp);
    }
    ros::Duration render_duration = ros::Time::now() - tnow;
    if (render_duration > depth_pub_duration || render_duration > lidar_pub_duration){
        // Performance reference: should take < 1 ms on 3060 GPU & Ubuntu 20.04
        ROS_WARN("Current Rendering time: %.2f ms, delay too much!", 1000 * render_duration.toSec());
        std::cout << "Average Depth Rendering time: " << (depth_time / (depth_count + 1e-8)) * 1000 << " ms" << std::endl;
        std::cout << "Average Lidar Rendering time: " << (lidar_time / (lidar_count + 1e-8)) * 1000 << " ms" << std::endl;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator_node");
    ros::NodeHandle nh;

    SensorSimulator sensor_simulator(nh);
    return 0;
}
