#ifndef MAPS_HPP
#define MAPS_HPP
#include <yaml-cpp/yaml.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Core>
#include "perlinnoise.hpp"

namespace mocka {

class Maps {
public:
  typedef struct BasicInfo {
    int sizeX;
    int sizeY;
    int sizeZ;
    int seed;
    double scale;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  } BasicInfo;

  BasicInfo getInfo() const;
  void setInfo(const BasicInfo &value);
  void setParam(const YAML::Node& config);
  void setGateRollDeg(double roll_deg) { gate_roll_deg = roll_deg; }
  Maps() {}
  void generate(int type);

private:
  BasicInfo info;
  // perlin3D
  double complexity;
  double fill;
  int    fractal;
  double attenuation;
  // randomMap
  double _w_l, _w_h;
  int    _ObsNum;
  // maze2D
  double width;
  int    addWallX;
  int    addWallY;
  // narrow gate / gate-wall scene
  bool gate_enabled = false;
  double gate_roll_deg = 0.0;
  double gate_pitch_deg = 0.0;
  double gate_yaw_deg = 0.0;
  double gate_x = 0.0;
  double gate_y = 0.0;
  double gate_z = 1.2;
  int gate_count = 1;
  double gate_spacing = 3.0;
  double gate_outer_width = 0.88;
  double gate_outer_length = 0.38;
  double gate_inner_width = 0.70;
  double gate_inner_length = 0.30;
  double gate_depth = 0.05;
  double gate_depth_margin = 0.01;
  double gate_point_resolution = 0.05;
  double gate_wall_width = 0.88;
  double gate_wall_length = 0.38;
  bool gate_wall_fill_boundary = false;
  bool add_ground_points = true;
  // room
  int room_number;
  int max_windows;
  int add_ceiling;
  double window_size_min, window_size_max;
  // wall
  double _wall_w_l, _wall_w_h;
  double _wall_thick;
  int    _wall_num;
  int    _wall_ceiling;

  std::uniform_real_distribution<double> dis_window_x, dis_window_z, dis_window_size;
  std::default_random_engine window_eng;

  void perlin3D();
  void maze2D();
  void randomMapGenerate();
  void Maze3DGen();
  void wall();
  void recursiveDivision(int xl, int xh, int yl, int yh, Eigen::MatrixXi &maze);
  void recursizeDivisionMaze(Eigen::MatrixXi &maze);
  void optimizeMap();

  void gateWallScene();
  void addMapBoundaryAnchors();
  void addGateWall();
  std::vector<Eigen::Vector3f> gateCenters() const;
  pcl::PointCloud<pcl::PointXYZ>::Ptr generateGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud, float grid_size, float hight = 0.0);

  void room();
  void transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud,
                           const Eigen::Matrix3f &rotation, const Eigen::Vector3f &translation);
  void generateWallWithWindows(pcl::PointCloud<pcl::PointXYZ>::Ptr wall, float L, float W, float H, int num_windows);
};

class MazePoint {
private:
  pcl::PointXYZ point;
  double dist1;
  double dist2;
  int point1;
  int point2;
  bool isdoor;

public:
  pcl::PointXYZ getPoint();
  int getPoint1();
  int getPoint2();
  double getDist1();
  double getDist2();
  void setPoint(pcl::PointXYZ p);
  void setPoint1(int p);
  void setPoint2(int p);
  void setDist1(double set);
  void setDist2(double set);
};

} // namespace mocka

#endif // MAPS_HPP
