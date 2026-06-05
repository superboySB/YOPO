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

struct VisibleTarget
{
    Eigen::Vector3f target_c;
    Eigen::Vector3f target_w;
    float u{-1.0f};
    float v{-1.0f};
    float distance{0.0f};
};

struct TargetEllipsoid
{
    Eigen::Vector3f axes{0.22f, 0.22f, 0.12f};
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

void drawTargetMask(cv::Mat &target_mask,
                   const Eigen::Vector3f &target_c,
                   const CameraParams &camera,
                   const TargetEllipsoid &target_shape)
{
    float u_center = 0.0f;
    float v_center = 0.0f;
    if (!projectTarget(target_c, camera, u_center, v_center))
        return;

    const int radius_u = std::max(2, static_cast<int>(std::ceil(camera.fx * target_shape.axes.y() / target_c.x())));
    const int radius_v = std::max(2, static_cast<int>(std::ceil(camera.fy * target_shape.axes.z() / target_c.x())));
    cv::ellipse(
        target_mask,
        cv::Point(static_cast<int>(std::round(u_center)), static_cast<int>(std::round(v_center))),
        cv::Size(radius_u, radius_v),
        0.0,
        0.0,
        360.0,
        cv::Scalar(255),
        cv::FILLED);
}

bool targetVisibleInDepth(const cv::Mat &depth_image,
                          const Eigen::Vector3f &target_c,
                          const CameraParams &camera,
                          const TargetEllipsoid &target_shape,
                          float occlusion_margin,
                          int min_visible_pixels,
                          float min_visible_ratio)
{
    float u_center = 0.0f;
    float v_center = 0.0f;
    if (!projectTarget(target_c, camera, u_center, v_center))
        return false;

    const int radius_u = std::max(2, static_cast<int>(std::ceil(camera.fx * target_shape.axes.y() / target_c.x())));
    const int radius_v = std::max(2, static_cast<int>(std::ceil(camera.fy * target_shape.axes.z() / target_c.x())));
    const int u0 = std::max(0, static_cast<int>(std::floor(u_center)) - radius_u);
    const int u1 = std::min(camera.image_width - 1, static_cast<int>(std::ceil(u_center)) + radius_u);
    const int v0 = std::max(0, static_cast<int>(std::floor(v_center)) - radius_v);
    const int v1 = std::min(camera.image_height - 1, static_cast<int>(std::ceil(v_center)) + radius_v);

    int projected_pixels = 0;
    int visible_pixels = 0;
    for (int v = v0; v <= v1; ++v)
    {
        for (int u = u0; u <= u1; ++u)
        {
            const float y_at_target_x = -(u - camera.cx) / camera.fx * target_c.x();
            const float z_at_target_x = -(v - camera.cy) / camera.fy * target_c.x();
            const float y_norm = (y_at_target_x - target_c.y()) / target_shape.axes.y();
            const float z_norm = (z_at_target_x - target_c.z()) / target_shape.axes.z();
            const float lateral_norm_sq = y_norm * y_norm + z_norm * z_norm;
            if (lateral_norm_sq > 1.0f)
                continue;

            ++projected_pixels;
            const float surface_depth = target_c.x() - target_shape.axes.x() *
                std::sqrt(std::max(0.0f, 1.0f - lateral_norm_sq));
            const float depth = depth_image.at<float>(v, u);
            if (std::isfinite(depth) && depth >= surface_depth - occlusion_margin)
                ++visible_pixels;
        }
    }

    if (projected_pixels <= 0 || visible_pixels < min_visible_pixels)
        return false;
    return static_cast<float>(visible_pixels) / static_cast<float>(projected_pixels) >= min_visible_ratio;
}

bool targetInStaticCollision(GridMap &grid_map,
                             const Eigen::Vector3f &target_w,
                             const TargetEllipsoid &target_shape)
{
    const Eigen::Vector3f offsets[] = {
        Eigen::Vector3f::Zero(),
        Eigen::Vector3f(target_shape.axes.x(), 0.0f, 0.0f),
        Eigen::Vector3f(-target_shape.axes.x(), 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, target_shape.axes.y(), 0.0f),
        Eigen::Vector3f(0.0f, -target_shape.axes.y(), 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, target_shape.axes.z()),
        Eigen::Vector3f(0.0f, 0.0f, -target_shape.axes.z()),
    };

    for (const auto &offset_c : offsets)
    {
        const Eigen::Vector3f point_w = target_w + offset_c;
        if (grid_map.mapQueryHost(Vector3f(point_w.x(), point_w.y(), point_w.z())) == 1)
            return true;
    }
    return false;
}

std::vector<EllipsoidObstacle> makeDynamicObstacles(const std::vector<VisibleTarget> &visible_targets,
                                                    const TargetEllipsoid &target_shape)
{
    std::vector<EllipsoidObstacle> obstacles;
    obstacles.reserve(visible_targets.size());
    for (const auto &target : visible_targets)
    {
        EllipsoidObstacle obstacle;
        obstacle.center = Vector3f(target.target_w.x(), target.target_w.y(), target.target_w.z());
        obstacle.radii = Vector3f(target_shape.axes.x(), target_shape.axes.y(), target_shape.axes.z());
        obstacles.push_back(obstacle);
    }
    return obstacles;
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

int sampleDynamicTargetCount(int min_count,
                             int max_count,
                             std::uniform_real_distribution<float> &uniform_distribution,
                             std::mt19937 &generator)
{
    int count = 0;
    while (count < max_count && uniform_distribution(generator) >= 0.5f)
        ++count;
    return std::max(min_count, count);
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
    const YAML::Node target_size = config["target"]["ellipsoid_size"];
    TargetEllipsoid target_shape;
    target_shape.axes = Eigen::Vector3f(
        target_size[0].as<float>() * 0.5f,
        target_size[1].as<float>() * 0.5f,
        target_size[2].as<float>() * 0.5f);
    float target_min_depth = config["target"] && config["target"]["min_depth"] ? config["target"]["min_depth"].as<float>() : 2.5f;
    float target_max_depth = config["target"] && config["target"]["max_depth"] ? config["target"]["max_depth"].as<float>() : 14.0f;
    float target_max_yaw = config["target"] && config["target"]["max_yaw_deg"] ? config["target"]["max_yaw_deg"].as<float>() : 35.0f;
    float target_max_pitch = config["target"] && config["target"]["max_pitch_deg"] ? config["target"]["max_pitch_deg"].as<float>() : 22.0f;
    float target_occlusion_margin = config["target"] && config["target"]["occlusion_margin"] ? config["target"]["occlusion_margin"].as<float>() : 0.3f;
    int target_mask_min_visible_pixels = config["target"]["mask_min_visible_pixels"].as<int>();
    float target_mask_min_visible_ratio = config["target"]["mask_min_visible_ratio"].as<float>();
    float target_min_center_distance = config["target"] && config["target"]["min_center_distance"]
                                           ? config["target"]["min_center_distance"].as<float>()
                                           : ((config["swarm"] && config["swarm"]["formation_lateral_spacing"])
                                                  ? config["swarm"]["formation_lateral_spacing"].as<float>()
                                                  : 2.0f);
    int target_mask_dynamic_min_count = config["target"] && config["target"]["mask_dynamic_min_count"]
                                            ? std::max(0, config["target"]["mask_dynamic_min_count"].as<int>())
                                            : 0;
    int target_mask_dynamic_max_count = config["target"] && config["target"]["mask_dynamic_max_count"]
                                            ? std::max(target_mask_dynamic_min_count, config["target"]["mask_dynamic_max_count"].as<int>())
                                            : 3;

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
        target_file << "target_count";
        for (int target_i = 0; target_i < target_mask_dynamic_max_count; ++target_i)
            target_file << ",tx" << target_i << ",ty" << target_i << ",tz" << target_i
                        << ",range" << target_i << ",u" << target_i << ",v" << target_i
                        << ",depth" << target_i;
        target_file << "\n";
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

            cv::Mat static_depth_image;
            renderDepthImage(&grid_map, &camera, T_wc, static_depth_image);
            cv::Mat target_mask = cv::Mat::zeros(camera.image_height, camera.image_width, CV_8UC1);

            Eigen::Matrix3f R_wc = quat_wc.toRotationMatrix();
            std::vector<VisibleTarget> visible_targets;
            const int target_count = sampleDynamicTargetCount(
                target_mask_dynamic_min_count,
                target_mask_dynamic_max_count,
                uniform_uniform,
                generator);
            visible_targets.reserve(target_count);
            for (int target_i = 0; target_i < target_count; ++target_i)
            {
                for (int attempt = 0; attempt < 120; ++attempt)
                {
                    const float depth = target_min_depth + uniform_uniform(generator) * (target_max_depth - target_min_depth);
                    const float yaw_rad = (2.0f * uniform_uniform(generator) - 1.0f) * target_max_yaw * M_PI / 180.0f;
                    const float pitch_rad = (2.0f * uniform_uniform(generator) - 1.0f) * target_max_pitch * M_PI / 180.0f;
                    Eigen::Vector3f candidate_c(
                        depth * std::cos(pitch_rad) * std::cos(yaw_rad),
                        -depth * std::cos(pitch_rad) * std::sin(yaw_rad),
                        -depth * std::sin(pitch_rad));

                    float candidate_u = -1.0f;
                    float candidate_v = -1.0f;
                    if (!projectTarget(candidate_c, camera, candidate_u, candidate_v))
                        continue;

                    Eigen::Vector3f candidate_w = pos + R_wc * candidate_c;
                    if (targetInStaticCollision(grid_map, candidate_w, target_shape))
                        continue;

                    bool too_close_to_existing = false;
                    for (const auto &existing_target : visible_targets)
                    {
                        if ((candidate_w - existing_target.target_w).norm() < target_min_center_distance)
                        {
                            too_close_to_existing = true;
                            break;
                        }
                    }
                    if (too_close_to_existing)
                        continue;

                    if (targetVisibleInDepth(static_depth_image,
                                             candidate_c,
                                             camera,
                                             target_shape,
                                             target_occlusion_margin,
                                             target_mask_min_visible_pixels,
                                             target_mask_min_visible_ratio))
                    {
                        visible_targets.push_back(VisibleTarget{
                            candidate_c,
                            candidate_w,
                            candidate_u,
                            candidate_v,
                            candidate_c.norm()});
                        break;
                    }
                }
            }

            if (!visible_targets.empty())
            {
                std::sort(
                    visible_targets.begin(),
                    visible_targets.end(),
                    [](const VisibleTarget &lhs, const VisibleTarget &rhs) {
                        return lhs.distance < rhs.distance;
                    });

                for (const auto &visible_target : visible_targets)
                    drawTargetMask(target_mask, visible_target.target_c, camera, target_shape);
            }
            const std::vector<EllipsoidObstacle> dynamic_obstacles = makeDynamicObstacles(visible_targets, target_shape);
            cv::Mat depth_image;
            renderDepthImage(&grid_map, &camera, T_wc, depth_image, dynamic_obstacles);

            const std::string depth_filename = image_path + "/depth_" + std::to_string(image_i) + ".png";
            const std::string mask_filename = image_path + "/mask_" + std::to_string(image_i) + ".png";
            saveDepthAs16BitPNG(depth_image, camera.max_depth_dist, depth_filename);
            cv::imwrite(mask_filename, target_mask);

            pose_file << std::fixed << std::setprecision(6)
                      << pos.x() << "," << pos.y() << "," << pos.z() << ","
                      << quat_wc.w() << "," << quat_wc.x() << ","
                      << quat_wc.y() << "," << quat_wc.z() << "\n";
            target_file << std::fixed << std::setprecision(6)
                        << visible_targets.size();
            for (int target_i = 0; target_i < target_mask_dynamic_max_count; ++target_i)
            {
                if (target_i < static_cast<int>(visible_targets.size()))
                {
                    const auto &target = visible_targets[target_i];
                    target_file << "," << target.target_w.x()
                                << "," << target.target_w.y()
                                << "," << target.target_w.z()
                                << "," << target.distance
                                << "," << target.u
                                << "," << target.v
                                << "," << target.target_c.x();
                }
                else
                {
                    target_file << ",0,0,0,-1,-1,-1,-1";
                }
            }
            target_file << "\n";

            printProgressBar(map_i * image_num + image_i + 1, dataset_num);
        }
        pose_file.close();
        target_file.close();
        grid_map.freeGridMap();
    }

    std::cout << "\nDataset generation completed!" << std::endl;

    return 0;
}
