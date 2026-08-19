#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <queue>
#include <vector>
#include "sensor_simulator.cuh"
#include "maps.hpp"

using namespace raycast;
namespace fs = std::filesystem;

float edgeFocalLengthPx(int pixels, float fov_deg)
{
    return (0.5f * static_cast<float>(pixels)) / std::tan(0.5f * fov_deg * M_PI / 180.0f);
}

CameraParams loadCameraParams(const YAML::Node &config)
{
    CameraParams camera;
    camera.image_width = config["image_width"].as<int>();
    camera.image_height = config["image_height"].as<int>();
    camera.cx = config["cx"] ? config["cx"].as<float>() : 0.5f * (camera.image_width - 1);
    camera.cy = config["cy"] ? config["cy"].as<float>() : 0.5f * (camera.image_height - 1);
    camera.fx = config["horizontal_fov_deg"] ? edgeFocalLengthPx(camera.image_width, config["horizontal_fov_deg"].as<float>())
                                             : config["fx"].as<float>();
    camera.fy = config["vertical_fov_deg"] ? edgeFocalLengthPx(camera.image_height, config["vertical_fov_deg"].as<float>())
                                           : config["fy"].as<float>();
    camera.max_depth_dist = config["max_depth_dist"].as<float>();
    camera.normalize_depth = config["normalize_depth"].as<bool>();
    return camera;
}

struct HostDijkstraGrid
{
    float resolution{0.5f};
    Eigen::Vector3f min_bound{0.0f, 0.0f, 0.0f};
    Eigen::Vector3i size{0, 0, 0};
    std::vector<uint8_t> occupied;

    HostDijkstraGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                     float grid_resolution,
                     float inflation_radius,
                     float z_min,
                     float z_max)
    {
        resolution = grid_resolution;
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        min_bound = Eigen::Vector3f(min_pt.x, min_pt.y, std::min(min_pt.z, z_min));
        Eigen::Vector3f max_bound(max_pt.x, max_pt.y, std::max(max_pt.z, z_max));
        size = ((max_bound - min_bound) / resolution).array().ceil().cast<int>().matrix();
        size += Eigen::Vector3i::Ones();
        occupied.assign(static_cast<size_t>(size.x()) * size.y() * size.z(), 0);

        for (const auto &point : cloud->points)
        {
            Eigen::Vector3i vox = posToVox(Eigen::Vector3f(point.x, point.y, point.z));
            if (inBounds(vox))
                occupied[index(vox)] = 1;
        }
        inflate(inflation_radius);
    }

    int index(const Eigen::Vector3i &vox) const
    {
        return (vox.x() * size.y() + vox.y()) * size.z() + vox.z();
    }

    bool inBounds(const Eigen::Vector3i &vox) const
    {
        return vox.x() >= 0 && vox.y() >= 0 && vox.z() >= 0 &&
               vox.x() < size.x() && vox.y() < size.y() && vox.z() < size.z();
    }

    Eigen::Vector3i posToVox(const Eigen::Vector3f &pos) const
    {
        return ((pos - min_bound) / resolution).array().floor().cast<int>();
    }

    Eigen::Vector3f voxToPos(const Eigen::Vector3i &vox) const
    {
        return min_bound + (vox.cast<float>() + Eigen::Vector3f::Constant(0.5f)) * resolution;
    }

    bool isFree(const Eigen::Vector3i &vox) const
    {
        return inBounds(vox) && occupied[index(vox)] == 0;
    }

    bool isFree2D(const Eigen::Vector2i &xy_vox, int z_vox) const
    {
        return isFree(Eigen::Vector3i(xy_vox.x(), xy_vox.y(), z_vox));
    }

    void inflate(float radius)
    {
        int r = static_cast<int>(std::ceil(radius / resolution));
        if (r <= 0)
            return;

        std::vector<uint8_t> inflated = occupied;
        int r2 = r * r;
        for (int x = 0; x < size.x(); ++x)
            for (int y = 0; y < size.y(); ++y)
                for (int z = 0; z < size.z(); ++z)
                {
                    Eigen::Vector3i center(x, y, z);
                    if (occupied[index(center)] == 0)
                        continue;

                    for (int dx = -r; dx <= r; ++dx)
                        for (int dy = -r; dy <= r; ++dy)
                            for (int dz = -r; dz <= r; ++dz)
                            {
                                if (dx * dx + dy * dy + dz * dz > r2)
                                    continue;
                                Eigen::Vector3i vox = center + Eigen::Vector3i(dx, dy, dz);
                                if (inBounds(vox))
                                    inflated[index(vox)] = 1;
                            }
                }
        occupied.swap(inflated);
    }
};

float sectorAngle(int direction_idx, int direction_num, std::default_random_engine &generator)
{
    const float sector_width = 2.0f * M_PI / static_cast<float>(direction_num);
    std::uniform_real_distribution<float> jitter(-0.35f * sector_width, 0.35f * sector_width);
    return direction_idx * sector_width + jitter(generator);
}

int directionSector(const Eigen::Vector3f &dir, int direction_num)
{
    float angle = std::atan2(dir.y(), dir.x());
    if (angle < 0.0f)
        angle += 2.0f * M_PI;
    int sector = static_cast<int>(std::floor((angle + M_PI / direction_num) / (2.0f * M_PI / direction_num)));
    return sector % direction_num;
}

float pathLength(const std::vector<Eigen::Vector3f> &path)
{
    float length = 0.0f;
    for (size_t i = 1; i < path.size(); ++i)
        length += (path[i] - path[i - 1]).norm();
    return length;
}

bool hasLineOfSight2D(const HostDijkstraGrid &grid,
                      const Eigen::Vector3f &start,
                      const Eigen::Vector3f &end,
                      const Eigen::Vector2i &start_xy,
                      int z_vox)
{
    Eigen::Vector2f start_xy_pos(start.x(), start.y());
    Eigen::Vector2f delta(end.x() - start.x(), end.y() - start.y());
    float length = delta.norm();
    if (length < 1e-5f)
        return true;

    int steps = std::max(1, static_cast<int>(std::ceil(length / (0.5f * grid.resolution))));
    for (int i = 0; i <= steps; ++i)
    {
        float ratio = static_cast<float>(i) / static_cast<float>(steps);
        Eigen::Vector2f point = start_xy_pos + ratio * delta;
        Eigen::Vector3i vox3 = grid.posToVox(Eigen::Vector3f(point.x(), point.y(), start.z()));
        Eigen::Vector2i xy_vox(vox3.x(), vox3.y());
        if (xy_vox != start_xy && !grid.isFree2D(xy_vox, z_vox))
            return false;
    }
    return true;
}

std::vector<Eigen::Vector3f> shortenPathByVisibility2D(const HostDijkstraGrid &grid,
                                                       const std::vector<Eigen::Vector3f> &path,
                                                       int z_vox)
{
    if (path.size() <= 2)
        return path;

    Eigen::Vector3i start_vox3 = grid.posToVox(path.front());
    Eigen::Vector2i start_xy(start_vox3.x(), start_vox3.y());
    std::vector<Eigen::Vector3f> shortened;
    shortened.push_back(path.front());
    size_t anchor = 0;
    while (anchor < path.size() - 1)
    {
        size_t best = anchor + 1;
        for (size_t candidate = path.size() - 1; candidate > anchor + 1; --candidate)
        {
            if (hasLineOfSight2D(grid, path[anchor], path[candidate], start_xy, z_vox))
            {
                best = candidate;
                break;
            }
        }
        shortened.push_back(path[best]);
        anchor = best;
    }
    return shortened;
}

bool projectGoalToFree2D(const HostDijkstraGrid &grid,
                         const Eigen::Vector3f &start,
                         const Eigen::Vector3f &goal,
                         float search_radius,
                         Eigen::Vector3f &free_goal)
{
    Eigen::Vector3i start_vox3 = grid.posToVox(start);
    Eigen::Vector3i goal_vox3 = grid.posToVox(goal);
    Eigen::Vector2i goal_xy(goal_vox3.x(), goal_vox3.y());
    const int z_vox = start_vox3.z();

    if (grid.isFree2D(goal_xy, z_vox))
    {
        free_goal = goal;
        return true;
    }

    int radius_vox = static_cast<int>(std::ceil(search_radius / grid.resolution));
    float best_dist2 = std::numeric_limits<float>::infinity();
    Eigen::Vector2i best_xy;
    bool found = false;
    for (int dx = -radius_vox; dx <= radius_vox; ++dx)
        for (int dy = -radius_vox; dy <= radius_vox; ++dy)
        {
            Eigen::Vector2i candidate_xy = goal_xy + Eigen::Vector2i(dx, dy);
            if (!grid.isFree2D(candidate_xy, z_vox))
                continue;
            Eigen::Vector3f candidate = grid.voxToPos(Eigen::Vector3i(candidate_xy.x(), candidate_xy.y(), z_vox));
            float dist2 = (candidate.head<2>() - goal.head<2>()).squaredNorm();
            if (dist2 < best_dist2)
            {
                best_dist2 = dist2;
                best_xy = candidate_xy;
                found = true;
            }
        }
    if (found)
    {
        free_goal = grid.voxToPos(Eigen::Vector3i(best_xy.x(), best_xy.y(), z_vox));
        free_goal.z() = goal.z();
        return true;
    }

    Eigen::Vector2f delta(goal.x() - start.x(), goal.y() - start.y());
    float length = delta.norm();
    if (length < 1e-5f)
        return false;

    int steps = std::max(1, static_cast<int>(std::ceil(length / grid.resolution)));
    for (int i = steps; i >= 1; --i)
    {
        float ratio = static_cast<float>(i) / static_cast<float>(steps);
        Eigen::Vector3f candidate = start + ratio * (goal - start);
        Eigen::Vector3i candidate_vox3 = grid.posToVox(candidate);
        Eigen::Vector2i candidate_xy(candidate_vox3.x(), candidate_vox3.y());
        if (grid.isFree2D(candidate_xy, z_vox))
        {
            free_goal = candidate;
            return true;
        }
    }
    return false;
}

bool runAStar(const HostDijkstraGrid &grid,
              const Eigen::Vector3f &start,
              const Eigen::Vector3f &requested_goal,
              float local_radius,
              float goal_search_radius,
              std::vector<Eigen::Vector3f> &path,
              float &path_cost,
              Eigen::Vector3f &near_field_dir)
{
    Eigen::Vector3f goal;
    if (!projectGoalToFree2D(grid, start, requested_goal, goal_search_radius, goal))
        return false;

    Eigen::Vector3i start_vox3 = grid.posToVox(start);
    Eigen::Vector3i goal_vox3 = grid.posToVox(goal);
    Eigen::Vector2i start_vox(start_vox3.x(), start_vox3.y());
    Eigen::Vector2i goal_vox(goal_vox3.x(), goal_vox3.y());
    const int z_vox = start_vox3.z();
    if (!grid.inBounds(start_vox3) || !grid.inBounds(Eigen::Vector3i(goal_vox.x(), goal_vox.y(), z_vox)) ||
        !grid.isFree2D(goal_vox, z_vox))
        return false;

    int radius_vox = static_cast<int>(std::ceil(local_radius / grid.resolution));
    Eigen::Vector2i roi_min = start_vox.cwiseMin(goal_vox) - Eigen::Vector2i::Constant(radius_vox);
    Eigen::Vector2i roi_max = start_vox.cwiseMax(goal_vox) + Eigen::Vector2i::Constant(radius_vox);
    roi_min = roi_min.cwiseMax(Eigen::Vector2i::Zero());
    roi_max = roi_max.cwiseMin(Eigen::Vector2i(grid.size.x() - 1, grid.size.y() - 1));
    Eigen::Vector2i roi_size = roi_max - roi_min + Eigen::Vector2i::Ones();
    int roi_total = roi_size.x() * roi_size.y();

    auto localIndex = [&](const Eigen::Vector2i &vox) {
        Eigen::Vector2i local = vox - roi_min;
        return local.x() * roi_size.y() + local.y();
    };
    auto localToGlobal = [&](int idx) -> Eigen::Vector2i {
        int y = idx % roi_size.y();
        int x = idx / roi_size.y();
        return roi_min + Eigen::Vector2i(x, y);
    };
    auto inRoi = [&](const Eigen::Vector2i &vox) {
        return (vox.array() >= roi_min.array()).all() && (vox.array() <= roi_max.array()).all();
    };
    auto isSearchFree = [&](const Eigen::Vector2i &vox) {
        return vox == start_vox || grid.isFree2D(vox, z_vox);
    };
    auto heuristic = [&](const Eigen::Vector2i &vox) {
        Eigen::Vector3f pos = grid.voxToPos(Eigen::Vector3i(vox.x(), vox.y(), z_vox));
        Eigen::Vector3f goal_pos = grid.voxToPos(Eigen::Vector3i(goal_vox.x(), goal_vox.y(), z_vox));
        return (pos.head<2>() - goal_pos.head<2>()).norm();
    };

    const int start_idx = localIndex(start_vox);
    const int goal_idx = localIndex(goal_vox);
    std::vector<float> dist(roi_total, std::numeric_limits<float>::infinity());
    std::vector<int> parent(roi_total, -1);
    std::priority_queue<std::pair<float, int>,
                        std::vector<std::pair<float, int>>,
                        std::greater<std::pair<float, int>>> queue;

    std::vector<Eigen::Vector2i> neighbors;
    std::vector<float> neighbor_costs;
    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            {
                if (dx == 0 && dy == 0)
                    continue;
                neighbors.emplace_back(dx, dy);
                neighbor_costs.push_back(grid.resolution * std::sqrt(static_cast<float>(dx * dx + dy * dy)));
            }

    dist[start_idx] = 0.0f;
    queue.emplace(heuristic(start_vox), start_idx);
    while (!queue.empty())
    {
        auto [cur_priority, cur_idx] = queue.top();
        queue.pop();
        if (cur_idx == goal_idx)
            break;

        Eigen::Vector2i cur_vox = localToGlobal(cur_idx);
        float cur_cost = dist[cur_idx];

        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            Eigen::Vector2i next_vox = cur_vox + neighbors[i];
            if (!inRoi(next_vox) || !isSearchFree(next_vox))
                continue;
            int next_idx = localIndex(next_vox);
            float next_cost = cur_cost + neighbor_costs[i];
            if (next_cost < dist[next_idx])
            {
                dist[next_idx] = next_cost;
                parent[next_idx] = cur_idx;
                queue.emplace(next_cost + heuristic(next_vox), next_idx);
            }
        }
    }

    if (!std::isfinite(dist[goal_idx]))
        return false;

    std::vector<Eigen::Vector3f> reversed;
    for (int idx = goal_idx; idx != -1; idx = parent[idx])
    {
        Eigen::Vector2i vox = localToGlobal(idx);
        Eigen::Vector3f point = grid.voxToPos(Eigen::Vector3i(vox.x(), vox.y(), z_vox));
        point.z() = start.z();
        reversed.push_back(point);
        if (idx == start_idx)
            break;
    }

    std::vector<Eigen::Vector3f> raw_path(reversed.rbegin(), reversed.rend());
    raw_path.front() = start;
    raw_path.back() = goal;
    near_field_dir = raw_path.size() > 1 ? raw_path[1] - raw_path[0] : goal - start;

    path.assign(reversed.rbegin(), reversed.rend());
    path.front() = start;
    path.back() = goal;
    path = shortenPathByVisibility2D(grid, path, z_vox);
    path_cost = pathLength(path);
    return true;
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

void renderInsight9DepthImage(GridMap *grid_map,
                              const CameraParams &camera,
                              cudaMat::SE3<float> &T_wc,
                              float min_depth,
                              float accuracy_ratio,
                              std::default_random_engine &generator,
                              cv::Mat &depth_image)
{
    renderDepthImage(grid_map, const_cast<CameraParams *>(&camera), T_wc, depth_image);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    for (int row = 0; row < depth_image.rows; ++row)
        for (int col = 0; col < depth_image.cols; ++col)
        {
            float depth = std::clamp(depth_image.at<float>(row, col), min_depth, camera.max_depth_dist);
            if (depth < camera.max_depth_dist - 1e-4f)
            {
                // The product sheet specifies <2% error at 3m.  Use one
                // quarter of that bound as Gaussian sigma so approximately
                // 95% of simulated samples remain inside the stated error.
                const float sigma = std::max(0.001f, 0.25f * accuracy_ratio * depth);
                depth += sigma * noise(generator);
            }
            depth = std::clamp(depth, min_depth, camera.max_depth_dist);
            depth_image.at<float>(row, col) = std::round(depth * 1000.0f) / 1000.0f;
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
    YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);

    // 1. Insight 9 learning-stereo depth model and two-axis mount.
    YAML::Node insight_config = config["insight9"];
    CameraParams insight_camera = loadCameraParams(insight_config);
    const float insight_min_depth = insight_config["min_depth_dist"].as<float>();
    const float insight_accuracy_ratio = insight_config["depth_accuracy_ratio"].as<float>();
    const float camera_pitch_limit_deg = insight_config["pitch_limit_deg"].as<float>();
    const float camera_yaw_limit_deg = insight_config["yaw_limit_deg"].as<float>();
    const float camera_mount_x = insight_config["mount_x"].as<float>();
    const float camera_mount_y = insight_config["mount_y"].as<float>();
    const float camera_mount_z = insight_config["mount_z"].as<float>();
    std::cout << "Depth model: " << insight_config["model"].as<std::string>()
              << ", native " << insight_config["native_image_width"].as<int>() << "x"
              << insight_config["native_image_height"].as<int>() << ", train "
              << insight_camera.image_width << "x" << insight_camera.image_height
              << ", FOV " << insight_config["horizontal_fov_deg"].as<float>()
              << "x" << insight_config["vertical_fov_deg"].as<float>()
              << " deg, range [" << insight_min_depth << ", " << insight_camera.max_depth_dist
              << "] m" << std::endl;

    // 3. 地图参数
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

    // 4. 数据集参数
    std::string save_path = config["save_path"].as<std::string>();
    if (!save_path.empty() && save_path.back() != '/')
        save_path += "/";
    int env_num = config["env_num"].as<int>();
    int image_num = config["image_num"].as<int>();
    float roll_range = config["roll_range"].as<float>();
    float pitch_range = config["pitch_range"].as<float>();
    float x_range = config["x_range"].as<float>();
    float y_range = config["y_range"].as<float>();
    float z_min = config["z_range"][0].as<float>();
    float z_max = config["z_range"][1].as<float>();
    float safe_dist = config["safe_dist"].as<float>();
    float ply_res = config["ply_res"].as<float>();
    int omni_direction_num = config["omni"]["direction_num"].as<int>();
    float omni_goal_length = config["omni"]["goal_length"].as<float>();
    float omni_goal_z_margin = config["omni"]["goal_z_margin"].as<float>();
    float goal_search_radius = config["omni"]["goal_search_radius"].as<float>();
    float dijkstra_resolution = config["omni"]["dijkstra_resolution"].as<float>();
    float dijkstra_inflation = config["omni"]["dijkstra_inflation"].as<float>();
    float astar_local_radius = config["omni"]["astar_local_radius"].as<float>();
    float camera_gaze_lookahead = config["omni"]["camera_gaze_lookahead_m"].as<float>();
    float omni_goal_z_min = z_min + omni_goal_z_margin;
    float omni_goal_z_max = z_max - omni_goal_z_margin;
    if (omni_goal_z_min > omni_goal_z_max)
    {
        omni_goal_z_min = z_min;
        omni_goal_z_max = z_max;
    }

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
    std::default_random_engine generator(seed);
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
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);

        std::string image_path = save_path + std::to_string(map_i) + "/";
        prepareSavePath(image_path);

        savePointCloudAsPLY(filtered_cloud, save_path + "pointcloud-" + std::to_string(map_i) + ".ply");
        HostDijkstraGrid dijkstra_grid(filtered_cloud, dijkstra_resolution, dijkstra_inflation, z_min, z_max);

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);

        // 收集当前环境的数据
        std::ofstream pose_file(save_path + "pose-" + std::to_string(map_i) + ".csv");
        pose_file << "px,py,pz,qw,qx,qy,qz\n";
        std::ofstream sample_file(save_path + "samples-" + std::to_string(map_i) + ".csv");
        sample_file << "sample_id,pose_id,dir_idx,px,py,pz,qw,qx,qy,qz,"
                    << "vdes_bx,vdes_by,vdes_bz,goal_wx,goal_wy,goal_wz,"
                    << "guide_offset,guide_len,guide_mask,guide_cost,selected_topology,"
                    << "camera_pitch,camera_yaw,camera_target_pitch,camera_target_yaw\n";
        std::ofstream guide_file(save_path + "guides-" + std::to_string(map_i) + ".csv");
        guide_file << "sample_id,point_idx,x,y,z\n";
        int guide_offset = 0;
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
                if (found_num <= 0)
                {
                    dist = 0.0f;
                    continue;
                }
                dist = sqrt(pointNKNSquaredDistance[0]);
            } while (dist < safe_dist || !dijkstra_grid.isFree(dijkstra_grid.posToVox(pos)));

            float roll = std::clamp(normal_distribution(generator) * roll_range / 3.0f, -roll_range, roll_range);
            float pitch = std::clamp(normal_distribution(generator) * pitch_range / 3.0f, -pitch_range, pitch_range);
            float yaw = uniform_uniform(generator) * 360.0f;
            const float camera_pitch_deg = (2.0f * uniform_uniform(generator) - 1.0f) * camera_pitch_limit_deg;
            const float camera_yaw_deg = (2.0f * uniform_uniform(generator) - 1.0f) * camera_yaw_limit_deg;
            const float camera_pitch_rad = camera_pitch_deg * M_PI / 180.0f;
            const float camera_yaw_rad = camera_yaw_deg * M_PI / 180.0f;

            Eigen::Quaternionf quat = RPY2Quat(roll, pitch, yaw);
            Eigen::Matrix3f R_yaw = RPY2Quat(0.0f, 0.0f, yaw).toRotationMatrix();
            cudaMat::SE3<float> T_wb(quat.w(), quat.x(), quat.y(), quat.z(),
                                     pos.x(), pos.y(), pos.z());

            Eigen::Quaternionf quat_bc_camera = RPY2Quat(0.0f, camera_pitch_deg, camera_yaw_deg);
            cudaMat::SE3<float> T_bc_camera(
                quat_bc_camera.w(), quat_bc_camera.x(), quat_bc_camera.y(), quat_bc_camera.z(),
                camera_mount_x, camera_mount_y, camera_mount_z);
            cudaMat::SE3<float> T_wc_camera = T_wb * T_bc_camera;
            cv::Mat depth_image;
            renderInsight9DepthImage(&grid_map, insight_camera, T_wc_camera,
                                     insight_min_depth, insight_accuracy_ratio,
                                     generator, depth_image);
            std::string filename = image_path + "/img_" + std::to_string(image_i) + "_depth.png";
            saveDepthAs16BitPNG(depth_image, insight_camera.max_depth_dist, filename);

            pose_file << std::fixed << std::setprecision(6)
                      << pos.x() << "," << pos.y() << "," << pos.z() << ","
                      << quat.w() << "," << quat.x() << ","
                      << quat.y() << "," << quat.z() << "\n";

            Eigen::Matrix3f R_wb = quat.toRotationMatrix();
            for (int dir_i = 0; dir_i < omni_direction_num; ++dir_i)
            {
                int sample_id = image_i * omni_direction_num + dir_i;
                float theta = sectorAngle(dir_i, omni_direction_num, generator);
                Eigen::Vector3f vdes_yaw(std::cos(theta), std::sin(theta), 0.0f);
                Eigen::Vector3f vdes_w = R_yaw * vdes_yaw;
                Eigen::Vector3f vdes_b = R_wb.transpose() * vdes_w;
                Eigen::Vector3f goal = pos + omni_goal_length * vdes_w.normalized();
                goal.z() = std::clamp(goal.z(), omni_goal_z_min, omni_goal_z_max);

                std::vector<Eigen::Vector3f> guide_path;
                Eigen::Vector3f near_field_dir = goal - pos;
                float guide_cost = 1e6f;
                bool guide_success = runAStar(dijkstra_grid, pos, goal, astar_local_radius, goal_search_radius,
                                              guide_path, guide_cost, near_field_dir);
                if (guide_success && !guide_path.empty())
                    goal = guide_path.back();
                int guide_len = guide_success ? static_cast<int>(guide_path.size()) : 0;
                int selected_topology = dir_i;
                if (guide_success && guide_path.size() > 1)
                {
                    Eigen::Vector3f init_dir_b = R_wb.transpose() * near_field_dir;
                    selected_topology = directionSector(init_dir_b, omni_direction_num);
                }

                Eigen::Vector3f gaze_point_w = goal;
                if (guide_success && guide_path.size() > 1)
                {
                    gaze_point_w = guide_path.back();
                    for (size_t path_i = 1; path_i < guide_path.size(); ++path_i)
                    {
                        gaze_point_w = guide_path[path_i];
                        if ((gaze_point_w - pos).norm() >= camera_gaze_lookahead)
                            break;
                    }
                }
                Eigen::Vector3f gaze_dir_b = R_wb.transpose() * (gaze_point_w - pos);
                const float gaze_horizontal = std::hypot(gaze_dir_b.x(), gaze_dir_b.y());
                float camera_target_pitch = std::atan2(-gaze_dir_b.z(), std::max(1e-6f, gaze_horizontal));
                float camera_target_yaw = std::atan2(gaze_dir_b.y(), gaze_dir_b.x());
                camera_target_pitch = std::clamp(
                    camera_target_pitch,
                    -camera_pitch_limit_deg * static_cast<float>(M_PI) / 180.0f,
                    camera_pitch_limit_deg * static_cast<float>(M_PI) / 180.0f);
                camera_target_yaw = std::clamp(
                    camera_target_yaw,
                    -camera_yaw_limit_deg * static_cast<float>(M_PI) / 180.0f,
                    camera_yaw_limit_deg * static_cast<float>(M_PI) / 180.0f);

                sample_file << std::fixed << std::setprecision(6)
                            << sample_id << "," << image_i << "," << dir_i << ","
                            << pos.x() << "," << pos.y() << "," << pos.z() << ","
                            << quat.w() << "," << quat.x() << "," << quat.y() << "," << quat.z() << ","
                            << vdes_b.x() << "," << vdes_b.y() << "," << vdes_b.z() << ","
                            << goal.x() << "," << goal.y() << "," << goal.z() << ","
                            << guide_offset << "," << guide_len << "," << (guide_success ? 1 : 0) << ","
                            << guide_cost << "," << selected_topology << ","
                            << camera_pitch_rad << "," << camera_yaw_rad << ","
                            << camera_target_pitch << "," << camera_target_yaw << "\n";

                for (int path_i = 0; path_i < guide_len; ++path_i)
                {
                    const auto &point = guide_path[path_i];
                    guide_file << std::fixed << std::setprecision(6)
                               << sample_id << "," << path_i << ","
                               << point.x() << "," << point.y() << "," << point.z() << "\n";
                }
                guide_offset += guide_len;
            }

            printProgressBar(map_i * image_num + image_i + 1, dataset_num);
        }
        pose_file.close();
        sample_file.close();
        guide_file.close();
        grid_map.freeGridMap();
    }

    std::cout << "\nDataset generation completed!" << std::endl;

    return 0;
}
