#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include "sensor_simulator.cuh"
#include "maps.hpp"

using namespace raycast;
namespace fs = std::filesystem;

struct DatasetCliOptions
{
    std::string config_path{CONFIG_FILE_PATH};
    std::string save_path_override{};
    int env_num_override{-1};
    int image_num_override{-1};
};

Eigen::Quaternionf RPY2Quat(float roll_deg, float pitch_deg, float yaw_deg);

DatasetCliOptions parseCliOptions(int argc, char **argv)
{
    DatasetCliOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
        {
            options.config_path = argv[++i];
        }
        else if (arg == "--save-path" && i + 1 < argc)
        {
            options.save_path_override = argv[++i];
        }
        else if (arg == "--env-num" && i + 1 < argc)
        {
            options.env_num_override = std::stoi(argv[++i]);
        }
        else if (arg == "--image-num" && i + 1 < argc)
        {
            options.image_num_override = std::stoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help")
        {
            std::cout << "Usage: dataset_generator [--config PATH] [--save-path DIR] [--env-num N] [--image-num N]" << std::endl;
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown arg: " << arg << std::endl;
            std::exit(1);
        }
    }
    return options;
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

bool projectTarget(const Eigen::Vector3f &target_c,
                   const CameraParams &camera,
                   float &u,
                   float &v)
{
    if (target_c.x() <= 0.1f || target_c.x() > camera.max_depth_dist)
        return false;
    u = camera.cx - camera.fx * target_c.y() / target_c.x();
    v = camera.cy - camera.fy * target_c.z() / target_c.x();
    return u >= 0.0f && u < camera.image_width && v >= 0.0f && v < camera.image_height;
}

void overlayTargetAndMask(cv::Mat &depth_image,
                   cv::Mat &target_mask,
                   const Eigen::Vector3f &target_c,
                   const CameraParams &camera,
                   float target_radius,
                   float occlusion_margin,
                   float mask_bbox_scale)
{
    float u_center = 0.0f;
    float v_center = 0.0f;
    if (!projectTarget(target_c, camera, u_center, v_center))
        return;

    const int radius_px = std::max(2, std::min(24, static_cast<int>(std::ceil(camera.fx * target_radius / target_c.x()))));
    const int u0 = std::max(0, static_cast<int>(std::floor(u_center)) - radius_px);
    const int u1 = std::min(camera.image_width - 1, static_cast<int>(std::ceil(u_center)) + radius_px);
    const int v0 = std::max(0, static_cast<int>(std::floor(v_center)) - radius_px);
    const int v1 = std::min(camera.image_height - 1, static_cast<int>(std::ceil(v_center)) + radius_px);

    const int bbox_half = std::max(2, static_cast<int>(std::ceil(radius_px * mask_bbox_scale)));
    const int bu0 = std::max(0, static_cast<int>(std::floor(u_center)) - bbox_half);
    const int bu1 = std::min(camera.image_width - 1, static_cast<int>(std::ceil(u_center)) + bbox_half);
    const int bv0 = std::max(0, static_cast<int>(std::floor(v_center)) - bbox_half);
    const int bv1 = std::min(camera.image_height - 1, static_cast<int>(std::ceil(v_center)) + bbox_half);
    cv::rectangle(target_mask, cv::Point(bu0, bv0), cv::Point(bu1, bv1), cv::Scalar(255), cv::FILLED);

    for (int py = v0; py <= v1; ++py)
    {
        for (int px = u0; px <= u1; ++px)
        {
            const float y_at_target_x = -(px - camera.cx) / camera.fx * target_c.x();
            const float z_at_target_x = -(py - camera.cy) / camera.fy * target_c.x();
            const float dy = y_at_target_x - target_c.y();
            const float dz = z_at_target_x - target_c.z();
            const float lateral_sq = dy * dy + dz * dz;
            const float radius_sq = target_radius * target_radius;
            if (lateral_sq > radius_sq)
                continue;

            const float surface_depth = target_c.x() - std::sqrt(std::max(0.0f, radius_sq - lateral_sq));
            float &depth_ref = depth_image.at<float>(py, px);
            depth_ref = surface_depth;
        }
    }
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
    const DatasetCliOptions cli_options = parseCliOptions(argc, argv);
    YAML::Node config = YAML::LoadFile(cli_options.config_path);

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
    int seed = config["seed"].as<int>();
    int sizeX = config["x_length"].as<int>();
    int sizeY = config["y_length"].as<int>();
    int sizeZ = config["z_length"].as<int>();
    double scale = 1 / resolution;
    sizeX *= scale;
    sizeY *= scale;
    sizeZ *= scale;

    // 3. 数据集参数
    std::string save_path = config["save_path"].as<std::string>();
    if (!cli_options.save_path_override.empty())
        save_path = cli_options.save_path_override;
    int env_num = config["env_num"].as<int>();
    int image_num = config["image_num"].as<int>();
    if (cli_options.env_num_override > 0)
        env_num = cli_options.env_num_override;
    if (cli_options.image_num_override > 0)
        image_num = cli_options.image_num_override;
    float roll_range = config["roll_range"].as<float>();
    float pitch_range = config["pitch_range"].as<float>();
    float x_range = config["x_range"].as<float>();
    float y_range = config["y_range"].as<float>();
    float z_min = config["z_range"][0].as<float>();
    float z_max = config["z_range"][1].as<float>();
    float safe_dist = config["safe_dist"].as<float>();
    float ply_res = config["ply_res"].as<float>();
    float target_radius = config["target"] && config["target"]["radius"] ? config["target"]["radius"].as<float>() : 0.35f;
    float target_min_depth = config["target"] && config["target"]["min_depth"] ? config["target"]["min_depth"].as<float>() : 2.5f;
    float target_max_depth = config["target"] && config["target"]["max_depth"] ? config["target"]["max_depth"].as<float>() : 14.0f;
    float target_max_yaw = config["target"] && config["target"]["max_yaw_deg"] ? config["target"]["max_yaw_deg"].as<float>() : 35.0f;
    float target_max_pitch = config["target"] && config["target"]["max_pitch_deg"] ? config["target"]["max_pitch_deg"].as<float>() : 22.0f;
    float target_occlusion_margin = config["target"] && config["target"]["occlusion_margin"] ? config["target"]["occlusion_margin"].as<float>() : 0.3f;
    float target_mask_bbox_scale = config["target"] && config["target"]["mask_bbox_scale"] ? config["target"]["mask_bbox_scale"].as<float>() : 1.2f;

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
    std::mt19937 generator(std::random_device{}());
    std::normal_distribution<float> normal_distribution(0.0f, 1.0f); // 均值0，标准差1
    std::uniform_real_distribution<float> uniform_uniform(0.0f, 1.0f);
    prepareSavePath(save_path, true);
    for (int map_i = 0; map_i < env_num; ++map_i)
    {
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
        map.setInfo(info);
        map.generate(config["maze_type"].as<int>());

        // 构建 GridMap
        GridMap grid_map(cloud, resolution, occupy_threshold);

        // 保存地图 (先滤波)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(ply_res, ply_res, ply_res);
        sor.filter(*filtered_cloud);

        std::string image_path = save_path + std::to_string(map_i) + "/";
        prepareSavePath(image_path);

        savePointCloudAsPLY(filtered_cloud, save_path + "pointcloud-" + std::to_string(map_i) + ".ply");

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(filtered_cloud);

        // 收集当前环境的数据
        std::ofstream pose_file(save_path + "pose-" + std::to_string(map_i) + ".csv");
        pose_file << "px,py,pz,qw,qx,qy,qz\n";
        std::ofstream target_file(save_path + "target-" + std::to_string(map_i) + ".csv");
        target_file << "tx,ty,tz,tvx,tvy,tvz,visible,u,v,depth\n";
        for (int image_i = 0; image_i < image_num; ++image_i)
        {
            Eigen::Vector3f pos;
            float dist;
            do{
                pos.x() = x_min + uniform_uniform(generator) * x_range;
                pos.y() = y_min + uniform_uniform(generator) * y_range;
                pos.z() = z_min + uniform_uniform(generator) * (z_max - z_min);
                pcl::PointXYZ searchPoint(pos.x(), pos.y(), pos.z());
                std::vector<int> pointIdxNKNSearch(1);
                std::vector<float> pointNKNSquaredDistance(1);
                int found_num = kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
                dist = sqrt(pointNKNSquaredDistance[0]);
            } while (dist < safe_dist);

            float roll = normal_distribution(generator) * roll_range / 3.0f;   // 3 * sigmoid = range
            float pitch = normal_distribution(generator) * pitch_range / 3.0f; // 3 * sigmoid = range
            float yaw = uniform_uniform(generator) * 360.0f;

            Eigen::Quaternionf quat = RPY2Quat(roll, pitch, yaw);
            Eigen::Quaternionf quat_wc = quat * quat_bc;

            cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(),
                                     pos.x(), pos.y(), pos.z());

            cv::Mat depth_image;
            renderDepthImage(&grid_map, &camera, T_wc, depth_image);
            cv::Mat target_mask = cv::Mat::zeros(camera.image_height, camera.image_width, CV_8UC1);

            Eigen::Matrix3f R_wc = quat_wc.toRotationMatrix();
            Eigen::Vector3f target_c(target_min_depth, 0.0f, 0.0f);
            Eigen::Vector3f target_w = pos + R_wc * target_c;
            float target_u = -1.0f;
            float target_v = -1.0f;
            bool target_visible = false;
            for (int attempt = 0; attempt < 120; ++attempt)
            {
                const float depth = target_min_depth + uniform_uniform(generator) * (target_max_depth - target_min_depth);
                const float yaw_rad = (2.0f * uniform_uniform(generator) - 1.0f) * target_max_yaw * M_PI / 180.0f;
                const float pitch_rad = (2.0f * uniform_uniform(generator) - 1.0f) * target_max_pitch * M_PI / 180.0f;
                target_c = Eigen::Vector3f(
                    depth * std::cos(pitch_rad) * std::cos(yaw_rad),
                    -depth * std::cos(pitch_rad) * std::sin(yaw_rad),
                    -depth * std::sin(pitch_rad));

                if (!projectTarget(target_c, camera, target_u, target_v))
                    continue;
                target_w = pos + R_wc * target_c;
                if (grid_map.mapQueryHost(Vector3f(target_w.x(), target_w.y(), target_w.z())) == 1)
                    continue;

                const float map_depth = depth_image.at<float>(
                    std::max(0, std::min(camera.image_height - 1, static_cast<int>(std::round(target_v)))),
                    std::max(0, std::min(camera.image_width - 1, static_cast<int>(std::round(target_u)))));
                if (target_c.x() - target_radius < map_depth + target_occlusion_margin)
                {
                    target_visible = true;
                    break;
                }
            }

            if (target_visible)
                overlayTargetAndMask(depth_image, target_mask, target_c, camera, target_radius, target_occlusion_margin, target_mask_bbox_scale);

            const std::string depth_filename = image_path + "/depth_" + std::to_string(image_i) + ".png";
            const std::string mask_filename = image_path + "/mask_" + std::to_string(image_i) + ".png";
            saveDepthAs16BitPNG(depth_image, camera.max_depth_dist, depth_filename);
            cv::imwrite(mask_filename, target_mask);

            pose_file << std::fixed << std::setprecision(6)
                      << pos.x() << "," << pos.y() << "," << pos.z() << ","
                      << quat_wc.w() << "," << quat_wc.x() << ","
                      << quat_wc.y() << "," << quat_wc.z() << "\n";
            const Eigen::Vector3f target_v_w(
                normal_distribution(generator) * 0.5f,
                normal_distribution(generator) * 0.5f,
                normal_distribution(generator) * 0.2f);
            target_file << std::fixed << std::setprecision(6)
                        << target_w.x() << "," << target_w.y() << "," << target_w.z() << ","
                        << target_v_w.x() << "," << target_v_w.y() << "," << target_v_w.z() << ","
                        << (target_visible ? 1 : 0) << ","
                        << target_u << "," << target_v << "," << target_c.x() << "\n";

            printProgressBar(map_i * image_num + image_i + 1, dataset_num);
        }
        pose_file.close();
        target_file.close();
        grid_map.freeGridMap();
    }

    std::cout << "\nDataset generation completed!" << std::endl;

    return 0;
}
