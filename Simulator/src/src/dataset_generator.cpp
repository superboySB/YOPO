#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include "sensor_simulator.cuh"
#include "maps.hpp"

using namespace raycast;
namespace fs = std::filesystem;

template <typename T>
T yamlValueOrDefault(const YAML::Node &config, const std::string &key, const T &default_value)
{
    if (config[key])
    {
        return config[key].as<T>();
    }
    return default_value;
}

int intEnvOrDefault(const char *name, const int default_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return default_value;
    }
    return std::max(1, std::atoi(value));
}

std::string stringEnvOrDefault(const char *name, const std::string &default_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return default_value;
    }
    return std::string(value);
}

void prepareSavePath(const std::string &path, bool print=false)
{
    if (fs::exists(path))
    {
        if (print)
            std::cout << "Directory exists. Removing: " << path << std::endl;
        fs::remove_all(path);
    }
    fs::create_directories(path);
    if (print)
        std::cout << "Created new dataset directory: " << path << std::endl;
}

void savePointCloudAsPLY(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std::string &path)
{
    if (pcl::io::savePLYFileBinary(path, *cloud) == -1)
        std::cerr << "Failed to save ply file to " << path << std::endl;
}

void saveDepthAs16BitPNG(const cv::Mat &depth_float, float max_depth_dist, const std::string &filepath)
{
    cv::Mat depth_scaled;
    depth_scaled = depth_float / max_depth_dist; // 归一化0~1

    // clip [0,1]
    cv::threshold(depth_scaled, depth_scaled, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(depth_scaled, depth_scaled, 0.0, 0.0, cv::THRESH_TOZERO);

    // 转成uint16
    depth_scaled.convertTo(depth_scaled, CV_16UC1, 65535.0);

    cv::imwrite(filepath, depth_scaled);
}

Eigen::Quaternionf RPY2Quat(float roll_deg, float pitch_deg, float yaw_deg)
{
    float roll = roll_deg * M_PI / 180.0f;
    float pitch = pitch_deg * M_PI / 180.0f;
    float yaw = yaw_deg * M_PI / 180.0f;
    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());
    return yawAngle * pitchAngle * rollAngle;
}

void printProgressBar(int current, int total, int bar_width = 50)
{
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0f) << "%";
    std::cout.flush();
}

int main(int argc, char **argv)
{
    YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);

    // 1. 相机参数
    CameraParams camera;
    camera.fx = config["camera"]["fx"].as<float>();
    camera.fy = config["camera"]["fy"].as<float>();
    camera.cx = config["camera"]["cx"].as<float>();
    camera.cy = config["camera"]["cy"].as<float>();
    camera.image_width = config["camera"]["image_width"].as<int>();
    camera.image_height = config["camera"]["image_height"].as<int>();
    camera.max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
    camera.normalize_depth = config["camera"]["normalize_depth"].as<bool>();
    float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0;
    Eigen::AngleAxisf angle_axis(pitch, Eigen::Vector3f::UnitY());
    Eigen::Quaternionf quat_bc(angle_axis);

    // 2. 地图参数
    float resolution = config["resolution"].as<float>();
    int occupy_threshold = config["occupy_threshold"].as<int>();
    bool mirror_xy = yamlValueOrDefault<bool>(config, "map_mirror_xy", true);
    bool occupy_below_ground = yamlValueOrDefault<bool>(config, "occupy_below_ground", true);
    int seed = config["seed"].as<int>();
    int sizeX = config["x_length"].as<int>();
    int sizeY = config["y_length"].as<int>();
    int sizeZ = config["z_length"].as<int>();
    double scale = 1 / resolution;
    sizeX *= scale;
    sizeY *= scale;
    sizeZ *= scale;

    // 3. 数据集参数
    std::string save_path = stringEnvOrDefault("YOPO_DATASET_SAVE_PATH", config["save_path"].as<std::string>());
    int env_num = intEnvOrDefault("YOPO_DATASET_ENV_NUM", config["env_num"].as<int>());
    int image_num = intEnvOrDefault("YOPO_DATASET_IMAGE_NUM", config["image_num"].as<int>());
    float roll_range = config["roll_range"].as<float>();
    float pitch_range = config["pitch_range"].as<float>();
    float x_range = config["x_range"].as<float>();
    float y_range = config["y_range"].as<float>();
    float z_min = config["z_range"][0].as<float>();
    float z_max = config["z_range"][1].as<float>();
    float safe_dist = config["safe_dist"].as<float>();
    float ply_res = config["ply_res"].as<float>();

    bool gate_enabled = yamlValueOrDefault<bool>(config, "gate_enabled", false);
    float gate_pose_sample_ratio = gate_enabled ? yamlValueOrDefault<float>(config, "gate_pose_sample_ratio", 0.0f) : 0.0f;
    gate_pose_sample_ratio = std::max(0.0f, std::min(1.0f, gate_pose_sample_ratio));
    float gate_sample_x_min = yamlValueOrDefault<float>(config, "gate_sample_x_min", 0.9f);
    float gate_sample_x_max = yamlValueOrDefault<float>(config, "gate_sample_x_max", 3.0f);
    float gate_sample_y_range = yamlValueOrDefault<float>(config, "gate_sample_y_range", 0.9f);
    float gate_sample_z_range = yamlValueOrDefault<float>(config, "gate_sample_z_range", 0.45f);
    float gate_sample_center_ratio = yamlValueOrDefault<float>(config, "gate_sample_center_ratio", 0.7f);
    gate_sample_center_ratio = std::max(0.0f, std::min(1.0f, gate_sample_center_ratio));
    float gate_sample_y_std = yamlValueOrDefault<float>(config, "gate_sample_y_std", 0.22f);
    float gate_sample_z_std = yamlValueOrDefault<float>(config, "gate_sample_z_std", 0.14f);
    float gate_pose_yaw_noise_deg = yamlValueOrDefault<float>(config, "gate_pose_yaw_noise_deg", 15.0f);
    Eigen::Vector3f gate_center(yamlValueOrDefault<float>(config, "gate_x", 0.0f),
                                yamlValueOrDefault<float>(config, "gate_y", 0.0f),
                                yamlValueOrDefault<float>(config, "gate_z", 1.2f));
    float gate_roll_deg = yamlValueOrDefault<float>(config, "gate_roll_deg", 0.0f);
    gate_roll_deg = yamlValueOrDefault<float>(config, "gate_slit_roll_deg", gate_roll_deg);
    const float gate_pitch_deg = yamlValueOrDefault<float>(config, "gate_pitch_deg", 0.0f);
    const float gate_yaw_deg = yamlValueOrDefault<float>(config, "gate_yaw_deg", 0.0f);
    const bool gate_random_slit_roll = gate_enabled && yamlValueOrDefault<bool>(config, "gate_random_slit_roll", false);
    float gate_slit_roll_min_deg = yamlValueOrDefault<float>(config, "gate_slit_roll_min_deg", 0.0f);
    float gate_slit_roll_max_deg = yamlValueOrDefault<float>(config, "gate_slit_roll_max_deg", 90.0f);
    if (gate_slit_roll_min_deg > gate_slit_roll_max_deg)
        std::swap(gate_slit_roll_min_deg, gate_slit_roll_max_deg);
    int gate_count = std::max(1, yamlValueOrDefault<int>(config, "gate_count", 1));
    float gate_spacing = std::max(0.0f, yamlValueOrDefault<float>(config, "gate_spacing", 3.0f));

    // 中心对齐，计算偏移量
    int dataset_num = env_num * image_num;
    float x_min = -x_range / 2.0f;
    float y_min = -y_range / 2.0f;

    std::cout << "地图范围 (m): "
              << "X: [" << -sizeX * resolution / 2.0 << ", " << sizeX * resolution / 2.0 << "], "
              << "Y: [" << -sizeY * resolution / 2.0 << ", " << sizeY * resolution / 2.0 << "], "
              << "Z: [" << 0 << ", " << sizeZ * resolution << "]" << std::endl;

    std::cout << "采集范围 (m): "
              << "X: [" << x_min << ", " << x_min + x_range << "], "
              << "Y: [" << y_min << ", " << y_min + y_range << "], "
              << "Z: [" << z_min << ", " << z_max << "]" << std::endl;

    std::cout << "角度范围 (度): "
              << "Roll: [" << -roll_range << ", " << roll_range << "], "
              << "Pitch: [" << -pitch_range << ", " << pitch_range << "], "
              << "Yaw: [0, 360]" << std::endl;

    // 收集所有数据
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> normal_distribution(0.0f, 1.0f); // 均值0，标准差1
    std::uniform_real_distribution<float> uniform_uniform(0.0f, 1.0f);
    std::uniform_int_distribution<int> gate_index_distribution(0, gate_count - 1);
    prepareSavePath(save_path, true);
    for (int map_i = 0; map_i < env_num; ++map_i)
    {
        const float active_gate_roll_deg = gate_random_slit_roll
                                               ? gate_slit_roll_min_deg + uniform_uniform(generator) * (gate_slit_roll_max_deg - gate_slit_roll_min_deg)
                                               : gate_roll_deg;
        const Eigen::Matrix3f active_gate_rot = RPY2Quat(active_gate_roll_deg, gate_pitch_deg, gate_yaw_deg).toRotationMatrix();

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        mocka::Maps::BasicInfo info;
        info.sizeX = sizeX;
        info.sizeY = sizeY;
        info.sizeZ = sizeZ;
        info.seed = seed + map_i; // 每个环境使用不同的随机种子
        info.scale = scale;
        info.cloud = cloud;

        mocka::Maps map;
        map.setParam(config);
        map.setGateRollDeg(active_gate_roll_deg);
        map.setInfo(info);
        map.generate(config["maze_type"].as<int>());

        // 构建 GridMap
        GridMap grid_map(cloud, resolution, occupy_threshold, mirror_xy, occupy_below_ground);

        // 保存地图 (先滤波)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(ply_res, ply_res, ply_res);
        sor.filter(*filtered_cloud);
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);

        std::string image_path = save_path + std::to_string(map_i) + "/";
        prepareSavePath(image_path);

        savePointCloudAsPLY(filtered_cloud, save_path + "pointcloud-" + std::to_string(map_i) + ".ply");

        std::ofstream gate_file(save_path + "gate-" + std::to_string(map_i) + ".csv");
        gate_file << "roll,pitch,yaw,x,y,z\n";
        gate_file << std::fixed << std::setprecision(6)
                  << active_gate_roll_deg << "," << gate_pitch_deg << "," << gate_yaw_deg << ","
                  << gate_center.x() << "," << gate_center.y() << "," << gate_center.z() << "\n";
        gate_file.close();

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(filtered_cloud);

        // 收集当前环境的数据
        std::ofstream pose_file(save_path + "pose-" + std::to_string(map_i) + ".csv");
        pose_file << "px,py,pz,qw,qx,qy,qz\n";
        for (int image_i = 0; image_i < image_num; ++image_i)
        {
            Eigen::Vector3f pos;
            float dist;
            bool sample_gate_pose = gate_enabled && uniform_uniform(generator) < gate_pose_sample_ratio;
            int sample_attempts = 0;
            do{
                ++sample_attempts;
                if (sample_gate_pose && sample_attempts > 100)
                {
                    sample_gate_pose = false;
                }
                if (sample_gate_pose)
                {
                    const int gate_idx = gate_index_distribution(generator);
                    const Eigen::Vector3f active_gate_center =
                        gate_center + active_gate_rot * Eigen::Vector3f(gate_idx * gate_spacing, 0.0f, 0.0f);
                    const float side = uniform_uniform(generator) < 0.5f ? -1.0f : 1.0f;
                    const float local_x = side * (gate_sample_x_min + uniform_uniform(generator) * (gate_sample_x_max - gate_sample_x_min));
                    float local_y;
                    float local_z;
                    if (uniform_uniform(generator) < gate_sample_center_ratio)
                    {
                        local_y = std::max(-gate_sample_y_range,
                                           std::min(gate_sample_y_range, normal_distribution(generator) * gate_sample_y_std));
                        local_z = std::max(-gate_sample_z_range,
                                           std::min(gate_sample_z_range, normal_distribution(generator) * gate_sample_z_std));
                    }
                    else
                    {
                        local_y = (2.0f * uniform_uniform(generator) - 1.0f) * gate_sample_y_range;
                        local_z = (2.0f * uniform_uniform(generator) - 1.0f) * gate_sample_z_range;
                    }
                    pos = active_gate_center + active_gate_rot * Eigen::Vector3f(local_x, local_y, local_z);
                }
                else
                {
                    pos.x() = x_min + uniform_uniform(generator) * x_range;
                    pos.y() = y_min + uniform_uniform(generator) * y_range;
                    pos.z() = z_min + uniform_uniform(generator) * (z_max - z_min);
                }
                pcl::PointXYZ searchPoint(pos.x(), pos.y(), pos.z());
                std::vector<int> pointIdxNKNSearch(1);
                std::vector<float> pointNKNSquaredDistance(1);
                int found_num = kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
                if (found_num <= 0)
                {
                    dist = 0.0f;
                }
                else
                {
                    dist = sqrt(pointNKNSquaredDistance[0]);
                }
            } while (dist < safe_dist || pos.z() < z_min || pos.z() > z_max);

            float roll = normal_distribution(generator) * roll_range / 3.0f;   // 3 * sigmoid = range
            float pitch = normal_distribution(generator) * pitch_range / 3.0f; // 3 * sigmoid = range
            float yaw = uniform_uniform(generator) * 360.0f;
            if (sample_gate_pose)
            {
                Eigen::Vector3f local_to_first = active_gate_rot.transpose() * (pos - gate_center);
                int gate_idx = static_cast<int>(std::round(local_to_first.x() / std::max(1.0e-3f, gate_spacing)));
                gate_idx = std::max(0, std::min(gate_count - 1, gate_idx));
                const Eigen::Vector3f active_gate_center =
                    gate_center + active_gate_rot * Eigen::Vector3f(gate_idx * gate_spacing, 0.0f, 0.0f);
                Eigen::Vector3f look_dir = active_gate_center - pos;
                const float horiz_norm = std::max(1.0e-3f, std::sqrt(look_dir.x() * look_dir.x() + look_dir.y() * look_dir.y()));
                yaw = std::atan2(look_dir.y(), look_dir.x()) * 180.0f / M_PI +
                      normal_distribution(generator) * gate_pose_yaw_noise_deg / 3.0f;
                pitch = -std::atan2(look_dir.z(), horiz_norm) * 180.0f / M_PI +
                        normal_distribution(generator) * pitch_range / 6.0f;
                roll = normal_distribution(generator) * roll_range / 6.0f;
            }

            Eigen::Quaternionf quat = RPY2Quat(roll, pitch, yaw);
            Eigen::Quaternionf quat_wc = quat * quat_bc;

            cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(),
                                     pos.x(), pos.y(), pos.z());

            cv::Mat depth_image;
            renderDepthImage(&grid_map, &camera, T_wc, depth_image);

            std::string filename = image_path + "/img_" + std::to_string(image_i) + ".png";
            saveDepthAs16BitPNG(depth_image, camera.max_depth_dist, filename);

            pose_file << std::fixed << std::setprecision(6)
                      << pos.x() << "," << pos.y() << "," << pos.z() << ","
                      << quat_wc.w() << "," << quat_wc.x() << ","
                      << quat_wc.y() << "," << quat_wc.z() << "\n";

            printProgressBar(map_i * image_num + image_i + 1, dataset_num);
        }
        pose_file.close();
        grid_map.freeGridMap();
    }

    std::cout << "\nDataset generation completed!" << std::endl;

    return 0;
}
