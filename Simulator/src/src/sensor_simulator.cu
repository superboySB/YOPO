#include "sensor_simulator.cuh"

namespace raycast
{
    __global__ void mapQueryKernel(GridMap grid_map, Vector3f pos, int *occupied)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            occupied[0] = grid_map.mapQuery(pos);
    }

    GridMap::GridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution, int occupy_threshold)
    {
        const float epsilon = 0.001f;
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        float length = max_pt(0) - min_pt(0) + 2 * epsilon;
        float width = max_pt(1) - min_pt(1) + 2 * epsilon;
        float height = max_pt(2) - min_pt(2) + 2 * epsilon;
        Vector3f origin(min_pt(0), min_pt(1), min_pt(2));
        Vector3f map_size(length, width, height);
        origin_x_ = origin.x;
        origin_y_ = origin.y;
        origin_z_ = origin.z;

        Vector3i grid_size;
        grid_size.x = ceil(map_size.x / resolution);
        grid_size.y = ceil(map_size.y / resolution);
        grid_size.z = ceil(map_size.z / resolution);
        int grid_total_size = grid_size.x * grid_size.y * grid_size.z;

        resolution_ = resolution;
        grid_size_x_ = grid_size.x;
        grid_size_y_ = grid_size.y;
        grid_size_z_ = grid_size.z;
        grid_size_yz_ = grid_size.y * grid_size.z;
        occupy_threshold_ = occupy_threshold;
        raycast_step_ = resolution;

        std::vector<int> h_map(grid_total_size, 0);
        for (size_t i = 0; i < cloud->points.size(); i++)
        {
            Vector3f point(cloud->points[i].x + epsilon, cloud->points[i].y + epsilon, cloud->points[i].z + epsilon);
            int idx = Vox2Idx(Pos2Vox(point));
            if (idx < grid_total_size)
                h_map[idx]++;
        }
        cudaMalloc((void **)&map_cuda_, grid_total_size * sizeof(int));
        cudaMemcpy(map_cuda_, h_map.data(), grid_total_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMallocManaged((void **)&query_cuda_, sizeof(int));
        query_cuda_[0] = 0;
    }

    void GridMap::freeGridMap()
    {
        if (map_cuda_ != nullptr)
        {
            cudaFree(map_cuda_);
            map_cuda_ = nullptr;
        }
        if (query_cuda_ != nullptr)
        {
            cudaFree(query_cuda_);
            query_cuda_ = nullptr;
        }
    }

    __host__ __device__ Vector3i GridMap::Pos2Vox(const Vector3f &pos)
    {
        Vector3i vox;
        vox.x = floor((pos.x - origin_x_) / resolution_);
        vox.y = floor((pos.y - origin_y_) / resolution_);
        vox.z = floor((pos.z - origin_z_) / resolution_);
        return vox;
    }

    __host__ __device__ Vector3f GridMap::Vox2Pos(const Vector3i &vox)
    {
        Vector3f pos;
        pos.x = (vox.x + 0.5f) * resolution_ + origin_x_;
        pos.y = (vox.y + 0.5f) * resolution_ + origin_y_;
        pos.z = (vox.z + 0.5f) * resolution_ + origin_z_;
        return pos;
    }

    __host__ __device__ int GridMap::Vox2Idx(const Vector3i &vox)
    {
        return vox.x * grid_size_yz_ + vox.y * grid_size_z_ + vox.z;
    }

    __host__ __device__ Vector3i GridMap::Idx2Vox(int idx)
    {
        return Vector3i(idx / grid_size_yz_, (idx % grid_size_yz_) / grid_size_z_, idx % grid_size_z_);
    }

    __device__ int GridMap::symmetricIndex(int index, int length)
    {
        index = index % (2 * length - 2);
        if (index < 0)
            index += (2 * length - 2);

        if (index >= length)
            index = 2 * length - 2 - index;
        return index;
    }

    __device__ int GridMap::mapQuery(const Vector3f &pos)
    {
        Vector3i vox = Pos2Vox(pos);
        vox.x = symmetricIndex(vox.x, grid_size_x_);
        vox.y = symmetricIndex(vox.y, grid_size_y_);

        if (vox.z >= grid_size_z_)
            return 0;
        if (vox.z <= 0)
            return 1;

        int idx = Vox2Idx(vox);
        if (map_cuda_[idx] > occupy_threshold_)
            return 1;
        return 0;
    }

    int GridMap::mapQueryHost(const Vector3f &pos)
    {
        mapQueryKernel<<<1, 1>>>(*this, pos, query_cuda_);
        cudaDeviceSynchronize();
        return query_cuda_[0];
    }

    __global__ void cameraRaycastKernel(float *depth_values,
                                        GridMap grid_map,
                                        CameraParams camera_param,
                                        cudaMat::SE3<float> T_wc)
    {
        int u = threadIdx.x;
        int v = blockIdx.x;

        if (u >= camera_param.image_width || v >= camera_param.image_height)
            return;

        float y = -(u - camera_param.cx) / camera_param.fx;
        float z = -(v - camera_param.cy) / camera_param.fy;
        float x = 1.0f;

        const float length = sqrtf(x * x + y * y + z * z);
        x /= length;
        y /= length;
        z /= length;

        const float dx = 0.5f * grid_map.raycast_step_;
        const float dy = (y / x) * dx;
        const float dz = (z / x) * dx;

        int scale = 0;
        float depth = camera_param.max_depth_dist;

        while (1)
        {
            scale += 1;

            const float point_x = scale * dx;
            const float point_y = scale * dy;
            const float point_z = scale * dz;

            if (point_x >= camera_param.max_depth_dist)
                break;

            const float3 point_c = make_float3(point_x, point_y, point_z);
            const float3 point_w = T_wc * point_c;
            const Vector3f point(point_w.x, point_w.y, point_w.z);

            if (grid_map.mapQuery(point) == 1)
            {
                const Vector3i occ_vox_w = grid_map.Pos2Vox(point);
                const Vector3f occ_point_w = grid_map.Vox2Pos(occ_vox_w);
                const float3 occ_point_c = T_wc.inv() * make_float3(occ_point_w.x, occ_point_w.y, occ_point_w.z);
                depth = occ_point_c.x;
                break;
            }
        }

        if (camera_param.normalize_depth)
            depth = depth / camera_param.max_depth_dist;
        depth_values[v * camera_param.image_width + u] = depth;
    }

    void renderDepthImage(GridMap *grid_map,
                          CameraParams *camera_param,
                          cudaMat::SE3<float> &T_wc,
                          cv::Mat &depth_image)
    {
        float *depth_values;
        size_t num_elements = camera_param->image_width * camera_param->image_height;
        cudaMallocManaged(&depth_values, num_elements * sizeof(float));

        cameraRaycastKernel<<<camera_param->image_height, camera_param->image_width>>>(
            depth_values, *grid_map, *camera_param, T_wc);
        cudaDeviceSynchronize();

        depth_image.create(camera_param->image_height, camera_param->image_width, CV_32FC1);
        cudaMemcpy(depth_image.data, depth_values, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(depth_values);
    }

    __global__ void lidarRaycastKernel(Vector3f *point_values,
                                       GridMap grid_map,
                                       LidarParams lidar_param,
                                       cudaMat::SE3<float> T_wc)
    {
        int h = threadIdx.x;
        int v = blockIdx.x;

        if (h >= lidar_param.horizontal_num || v >= lidar_param.vertical_lines)
            return;

        const float vertical_resolution = (lidar_param.vertical_angle_end - lidar_param.vertical_angle_start) / (lidar_param.vertical_lines - 1);
        const float vertical_angle = lidar_param.vertical_angle_start + v * vertical_resolution;
        const float sin_vert = std::sin(vertical_angle * M_PI / 180.0f);
        const float cos_vert = std::cos(vertical_angle * M_PI / 180.0f);
        const float horizontal_angle = h * lidar_param.horizontal_resolution;
        const float sin_horz = std::sin(horizontal_angle * M_PI / 180.0f);
        const float cos_horz = std::cos(horizontal_angle * M_PI / 180.0f);
        const Vector3f ray_direction_local(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);

        const float dx = ray_direction_local.x * grid_map.raycast_step_;
        const float dy = ray_direction_local.y * grid_map.raycast_step_;
        const float dz = ray_direction_local.z * grid_map.raycast_step_;

        int scale = 0;
        Vector3f point_value(0, 0, 0);

        while (1)
        {
            scale += 1;

            const float point_x = scale * dx;
            const float point_y = scale * dy;
            const float point_z = scale * dz;
            const float ray_length = sqrtf(point_x * point_x + point_y * point_y + point_z * point_z);

            if (ray_length >= lidar_param.max_lidar_dist)
                break;

            const float3 point_c = make_float3(point_x, point_y, point_z);
            const float3 point_w = T_wc * point_c;
            const Vector3f point(point_w.x, point_w.y, point_w.z);

            if (grid_map.mapQuery(point) == 1)
            {
                const Vector3i occ_vox_w = grid_map.Pos2Vox(point);
                const Vector3f occ_point_w = grid_map.Vox2Pos(occ_vox_w);
                const float3 occ_point_c = T_wc.inv() * make_float3(occ_point_w.x, occ_point_w.y, occ_point_w.z);
                point_value = Vector3f(occ_point_c.x, occ_point_c.y, occ_point_c.z);
                break;
            }
        }

        point_values[v * lidar_param.horizontal_num + h] = point_value;
    }

    void renderLidarPointcloud(GridMap *grid_map,
                               LidarParams *lidar_param,
                               cudaMat::SE3<float> &T_wc,
                               pcl::PointCloud<pcl::PointXYZ> &lidar_points)
    {
        Vector3f *point_values;
        size_t num_elements = lidar_param->vertical_lines * lidar_param->horizontal_num;
        cudaMallocManaged(&point_values, num_elements * sizeof(Vector3f));

        lidarRaycastKernel<<<lidar_param->vertical_lines, lidar_param->horizontal_num>>>(
            point_values, *grid_map, *lidar_param, T_wc);
        cudaDeviceSynchronize();

        std::vector<Vector3f> cpu_points(num_elements);
        cudaMemcpy(cpu_points.data(), point_values, num_elements * sizeof(Vector3f), cudaMemcpyDeviceToHost);

        lidar_points.points.clear();
        lidar_points.points.reserve(num_elements);
        for (const auto &point : cpu_points)
        {
            if (point.x != 0 || point.y != 0 || point.z != 0)
                lidar_points.points.emplace_back(point.x, point.y, point.z);
        }

        cudaFree(point_values);
    }
}
