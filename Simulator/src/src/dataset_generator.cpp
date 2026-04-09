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

struct SwarmDatasetOptions
{
    bool enabled{false};
    std::string save_path_override{};
    std::string placement_mode{"per_image"};
    int sphere_count_min{0};
    int sphere_count_max{0};
    float sphere_radius{0.25f};
    float sample_step{0.0f};
    float center_margin_xy{4.0f};
    float center_z_min{1.0f};
    float center_z_max{4.0f};
    float visible_depth_min{2.0f};
    float visible_depth_max{12.0f};
    float depth_margin{0.8f};
    int image_border{8};
    float camera_clearance{1.0f};
    float static_clearance{0.8f};
    float min_center_distance{3.0f};
    int max_attempts_per_obstacle{50};
};

struct SphereObstacleSpec
{
    Eigen::Vector3f center{Eigen::Vector3f::Zero()};
    float radius{0.25f};
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

SwarmDatasetOptions loadSwarmDatasetOptions(const YAML::Node &config)
{
    SwarmDatasetOptions options;
    const YAML::Node node = config["swarm_dataset"];
    if (!node)
        return options;

    options.enabled = node["enabled"] ? node["enabled"].as<bool>() : false;
    options.save_path_override = node["save_path"] ? node["save_path"].as<std::string>() : "";
    options.placement_mode = node["placement_mode"] ? node["placement_mode"].as<std::string>() : options.placement_mode;
    options.sphere_count_min = node["sphere_count_min"] ? node["sphere_count_min"].as<int>() : options.sphere_count_min;
    options.sphere_count_max = node["sphere_count_max"] ? node["sphere_count_max"].as<int>() : options.sphere_count_max;
    options.sphere_radius = node["sphere_radius"] ? node["sphere_radius"].as<float>()
                         : ((config["swarm"] && config["swarm"]["collision_radius"])
                            ? config["swarm"]["collision_radius"].as<float>()
                            : options.sphere_radius);
    options.sample_step = node["sample_step"] ? node["sample_step"].as<float>() : options.sample_step;
    options.center_margin_xy = node["center_margin_xy"] ? node["center_margin_xy"].as<float>() : options.center_margin_xy;
    options.visible_depth_min = node["visible_depth_min"] ? node["visible_depth_min"].as<float>() : options.visible_depth_min;
    options.visible_depth_max = node["visible_depth_max"] ? node["visible_depth_max"].as<float>() : options.visible_depth_max;
    options.depth_margin = node["depth_margin"] ? node["depth_margin"].as<float>() : options.depth_margin;
    options.image_border = node["image_border"] ? node["image_border"].as<int>() : options.image_border;
    options.camera_clearance = node["camera_clearance"] ? node["camera_clearance"].as<float>() : options.camera_clearance;
    options.static_clearance = node["static_clearance"] ? node["static_clearance"].as<float>() : options.static_clearance;
    options.min_center_distance = node["min_center_distance"] ? node["min_center_distance"].as<float>() : options.min_center_distance;
    options.max_attempts_per_obstacle = node["max_attempts_per_obstacle"] ? node["max_attempts_per_obstacle"].as<int>() : options.max_attempts_per_obstacle;
    if (node["center_z_range"] && node["center_z_range"].IsSequence() && node["center_z_range"].size() == 2)
    {
        options.center_z_min = node["center_z_range"][0].as<float>();
        options.center_z_max = node["center_z_range"][1].as<float>();
    }
    options.sphere_count_max = std::max(options.sphere_count_min, options.sphere_count_max);
    return options;
}

bool useMapSwarmDatasetSpheres(const SwarmDatasetOptions &options)
{
    return options.enabled && options.placement_mode == "map";
}

bool usePerImageSwarmDatasetSpheres(const SwarmDatasetOptions &options)
{
    return options.enabled && options.placement_mode != "map" && options.sphere_count_max > 0 && options.sphere_radius > 0.0f;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr createFilledSphereCloud(const SphereObstacleSpec &spec, float sample_step)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for (float x = -spec.radius; x <= spec.radius; x += sample_step)
    {
        for (float y = -spec.radius; y <= spec.radius; y += sample_step)
        {
            for (float z = -spec.radius; z <= spec.radius; z += sample_step)
            {
                if (x * x + y * y + z * z > spec.radius * spec.radius)
                    continue;

                pcl::PointXYZ point;
                point.x = spec.center.x() + x;
                point.y = spec.center.y() + y;
                point.z = spec.center.z() + z;
                cloud->points.push_back(point);
            }
        }
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

void appendSwarmDatasetSpheres(const SwarmDatasetOptions &options,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                               float resolution,
                               float x_min,
                               float x_max,
                               float y_min,
                               float y_max,
                               int seed)
{
    if (!options.enabled || options.sphere_count_max <= 0 || options.sphere_radius <= 0.0f)
        return;

    const float sample_step = options.sample_step > 0.0f ? options.sample_step : resolution;
    const float x_low = x_min + options.center_margin_xy;
    const float x_high = x_max - options.center_margin_xy;
    const float y_low = y_min + options.center_margin_xy;
    const float y_high = y_max - options.center_margin_xy;
    if (x_low >= x_high || y_low >= y_high)
    {
        std::cerr << "[swarm_dataset] invalid XY range for sphere placement." << std::endl;
        return;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> static_kdtree;
    static_kdtree.setInputCloud(cloud);

    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> count_dist(options.sphere_count_min, options.sphere_count_max);
    std::uniform_real_distribution<float> x_dist(x_low, x_high);
    std::uniform_real_distribution<float> y_dist(y_low, y_high);
    std::uniform_real_distribution<float> z_dist(options.center_z_min, options.center_z_max);

    const int target_count = count_dist(generator);
    std::vector<SphereObstacleSpec> spheres;
    spheres.reserve(target_count);

    for (int target_idx = 0; target_idx < target_count; ++target_idx)
    {
        bool placed = false;
        for (int attempt = 0; attempt < options.max_attempts_per_obstacle; ++attempt)
        {
            SphereObstacleSpec spec;
            spec.center = Eigen::Vector3f(x_dist(generator), y_dist(generator), z_dist(generator));
            spec.radius = options.sphere_radius;

            const float required_static_clearance = spec.radius + options.static_clearance;
            pcl::PointXYZ search_point(spec.center.x(), spec.center.y(), spec.center.z());
            std::vector<int> point_indices(1);
            std::vector<float> point_dist_sq(1);
            const int found_num = static_kdtree.nearestKSearch(search_point, 1, point_indices, point_dist_sq);
            if (found_num > 0 && std::sqrt(point_dist_sq[0]) < required_static_clearance)
                continue;

            bool overlaps_existing = false;
            for (const auto &existing : spheres)
            {
                const float required_center_distance =
                    existing.radius + spec.radius + options.min_center_distance;
                if ((existing.center - spec.center).norm() < required_center_distance)
                {
                    overlaps_existing = true;
                    break;
                }
            }
            if (overlaps_existing)
                continue;

            spheres.push_back(spec);
            placed = true;
            break;
        }
        if (!placed)
        {
            std::cerr << "[swarm_dataset] failed to place sphere " << target_idx
                      << " after " << options.max_attempts_per_obstacle << " attempts." << std::endl;
        }
    }

    for (const auto &spec : spheres)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_cloud = createFilledSphereCloud(spec, sample_step);
        *cloud += *sphere_cloud;
    }
    std::cout << "[swarm_dataset] appended " << spheres.size()
              << " sphere obstacles to the dataset map. radius=" << options.sphere_radius << std::endl;
}

Eigen::Vector3f pixelDepthToCameraPoint(int u, int v, float depth, const CameraParams &camera)
{
    const float y = -(static_cast<float>(u) - camera.cx) / camera.fx * depth;
    const float z = -(static_cast<float>(v) - camera.cy) / camera.fy * depth;
    return Eigen::Vector3f(depth, y, z);
}

Eigen::Vector3f cameraPointToWorld(const cudaMat::SE3<float> &T_wc, const Eigen::Vector3f &point_c)
{
    const float3 point_w = T_wc * make_float3(point_c.x(), point_c.y(), point_c.z());
    return Eigen::Vector3f(point_w.x, point_w.y, point_w.z);
}

std::vector<SphereObstacle> samplePerImageSwarmDatasetSpheres(
    const SwarmDatasetOptions &options,
    const cv::Mat &static_depth_image,
    const CameraParams &camera,
    const cudaMat::SE3<float> &T_wc,
    const Eigen::Vector3f &camera_pos,
    pcl::KdTreeFLANN<pcl::PointXYZ> &static_kdtree,
    const Eigen::Vector3f &world_min,
    const Eigen::Vector3f &world_max,
    std::mt19937 &generator)
{
    std::vector<SphereObstacle> obstacles;
    if (!usePerImageSwarmDatasetSpheres(options))
        return obstacles;

    const int x_low = std::max(0, options.image_border);
    const int x_high = std::min(camera.image_width - 1, camera.image_width - 1 - options.image_border);
    const int y_low = std::max(0, options.image_border);
    const int y_high = std::min(camera.image_height - 1, camera.image_height - 1 - options.image_border);
    if (x_low > x_high || y_low > y_high)
        return obstacles;

    std::uniform_int_distribution<int> count_dist(options.sphere_count_min, options.sphere_count_max);
    std::vector<cv::Point> candidate_pixels;
    candidate_pixels.reserve((x_high - x_low + 1) * (y_high - y_low + 1));
    for (int v = y_low; v <= y_high; ++v)
    {
        for (int u = x_low; u <= x_high; ++u)
        {
            const float static_depth = static_depth_image.at<float>(v, u);
            if (static_depth <= options.visible_depth_min + options.depth_margin)
                continue;
            if (static_depth >= camera.max_depth_dist - options.depth_margin)
                continue;
            candidate_pixels.emplace_back(u, v);
        }
    }
    if (candidate_pixels.empty())
        return obstacles;

    std::uniform_int_distribution<int> pixel_idx_dist(0, candidate_pixels.size() - 1);

    const int target_count = count_dist(generator);
    obstacles.reserve(target_count);

    for (int target_idx = 0; target_idx < target_count; ++target_idx)
    {
        bool placed = false;
        for (int attempt = 0; attempt < options.max_attempts_per_obstacle; ++attempt)
        {
            const cv::Point pixel = candidate_pixels[pixel_idx_dist(generator)];
            const int u = pixel.x;
            const int v = pixel.y;
            const float static_depth = static_depth_image.at<float>(v, u);

            const float max_center_depth = std::min(options.visible_depth_max, static_depth - options.depth_margin);
            if (max_center_depth <= options.visible_depth_min)
                continue;

            std::uniform_real_distribution<float> center_depth_dist(options.visible_depth_min, max_center_depth);
            const float center_depth = center_depth_dist(generator);
            const Eigen::Vector3f center_c = pixelDepthToCameraPoint(u, v, center_depth, camera);
            const Eigen::Vector3f center_w = cameraPointToWorld(T_wc, center_c);

            if (center_w.x() < world_min.x() + options.sphere_radius ||
                center_w.x() > world_max.x() - options.sphere_radius ||
                center_w.y() < world_min.y() + options.sphere_radius ||
                center_w.y() > world_max.y() - options.sphere_radius)
                continue;

            if (center_w.z() < std::max(world_min.z() + options.sphere_radius, options.center_z_min) ||
                center_w.z() > std::min(world_max.z() - options.sphere_radius, options.center_z_max))
                continue;

            if ((center_w - camera_pos).norm() < options.camera_clearance + options.sphere_radius)
                continue;

            const float required_static_clearance = options.sphere_radius + options.static_clearance;
            pcl::PointXYZ search_point(center_w.x(), center_w.y(), center_w.z());
            std::vector<int> point_indices(1);
            std::vector<float> point_dist_sq(1);
            const int found_num = static_kdtree.nearestKSearch(search_point, 1, point_indices, point_dist_sq);
            if (found_num > 0 && std::sqrt(point_dist_sq[0]) < required_static_clearance)
                continue;

            bool overlaps_existing = false;
            for (const auto &existing : obstacles)
            {
                const Eigen::Vector3f existing_center(existing.center.x, existing.center.y, existing.center.z);
                const float required_center_distance = existing.radius + options.sphere_radius + options.min_center_distance;
                if ((existing_center - center_w).norm() < required_center_distance)
                {
                    overlaps_existing = true;
                    break;
                }
            }
            if (overlaps_existing)
                continue;

            SphereObstacle obstacle;
            obstacle.center = Vector3f(center_w.x(), center_w.y(), center_w.z());
            obstacle.radius = options.sphere_radius;
            obstacles.push_back(obstacle);
            placed = true;
            break;
        }

        if (!placed)
        {
            std::cerr << "[swarm_dataset] failed to place per-image sphere " << target_idx
                      << " after " << options.max_attempts_per_obstacle << " attempts." << std::endl;
        }
    }

    return obstacles;
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
    const DatasetCliOptions cli_options = parseCliOptions(argc, argv);
    YAML::Node config = YAML::LoadFile(cli_options.config_path);
    const SwarmDatasetOptions swarm_dataset_options = loadSwarmDatasetOptions(config);

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
    if (swarm_dataset_options.enabled && !swarm_dataset_options.save_path_override.empty())
        save_path = swarm_dataset_options.save_path_override;
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
        if (useMapSwarmDatasetSpheres(swarm_dataset_options))
            appendSwarmDatasetSpheres(swarm_dataset_options, cloud, resolution, x_min, x_min + x_range, y_min, y_min + y_range, seed + map_i);

        // 构建 GridMap
        GridMap grid_map(cloud, resolution, occupy_threshold);

        // 保存地图 (先滤波)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(ply_res, ply_res, ply_res);
        sor.filter(*filtered_cloud);
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);
        const Eigen::Vector3f world_min(min_pt.x, min_pt.y, min_pt.z);
        const Eigen::Vector3f world_max(max_pt.x, max_pt.y, max_pt.z);

        std::string image_path = save_path + std::to_string(map_i) + "/";
        prepareSavePath(image_path);

        savePointCloudAsPLY(filtered_cloud, save_path + "pointcloud-" + std::to_string(map_i) + ".ply");

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(filtered_cloud);

        // 收集当前环境的数据
        std::ofstream pose_file(save_path + "pose-" + std::to_string(map_i) + ".csv");
        pose_file << "px,py,pz,qw,qx,qy,qz\n";
        std::ofstream dynamic_obstacle_file;
        if (usePerImageSwarmDatasetSpheres(swarm_dataset_options))
        {
            dynamic_obstacle_file.open(save_path + "dynamic_obstacles-" + std::to_string(map_i) + ".csv");
            dynamic_obstacle_file << "count";
            for (int sphere_idx = 0; sphere_idx < swarm_dataset_options.sphere_count_max; ++sphere_idx)
                dynamic_obstacle_file << ",cx" << sphere_idx << ",cy" << sphere_idx << ",cz" << sphere_idx << ",r" << sphere_idx;
            dynamic_obstacle_file << "\n";
        }
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
            std::vector<SphereObstacle> dynamic_obstacles;
            if (usePerImageSwarmDatasetSpheres(swarm_dataset_options))
            {
                renderDepthImage(&grid_map, &camera, T_wc, depth_image);
                dynamic_obstacles = samplePerImageSwarmDatasetSpheres(
                    swarm_dataset_options,
                    depth_image,
                    camera,
                    T_wc,
                    pos,
                    kdtree,
                    world_min,
                    world_max,
                    generator);
                renderDepthImage(&grid_map, &camera, T_wc, depth_image, dynamic_obstacles);
            }
            else
            {
                renderDepthImage(&grid_map, &camera, T_wc, depth_image);
            }

            std::string filename = image_path + "/img_" + std::to_string(image_i) + ".png";
            saveDepthAs16BitPNG(depth_image, camera.max_depth_dist, filename);

            pose_file << std::fixed << std::setprecision(6)
                      << pos.x() << "," << pos.y() << "," << pos.z() << ","
                      << quat_wc.w() << "," << quat_wc.x() << ","
                      << quat_wc.y() << "," << quat_wc.z() << "\n";

            if (dynamic_obstacle_file.is_open())
            {
                dynamic_obstacle_file << dynamic_obstacles.size();
                for (int sphere_idx = 0; sphere_idx < swarm_dataset_options.sphere_count_max; ++sphere_idx)
                {
                    if (sphere_idx < static_cast<int>(dynamic_obstacles.size()))
                    {
                        const SphereObstacle &obstacle = dynamic_obstacles[sphere_idx];
                        dynamic_obstacle_file << "," << obstacle.center.x << "," << obstacle.center.y << ","
                                              << obstacle.center.z << "," << obstacle.radius;
                    }
                    else
                    {
                        dynamic_obstacle_file << ",0,0,0,0";
                    }
                }
                dynamic_obstacle_file << "\n";
            }

            printProgressBar(map_i * image_num + image_i + 1, dataset_num);
        }
        pose_file.close();
        if (dynamic_obstacle_file.is_open())
            dynamic_obstacle_file.close();
        grid_map.freeGridMap();
    }

    std::cout << "\nDataset generation completed!" << std::endl;

    return 0;
}
