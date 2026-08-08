#ifndef NETWORK_CONTROL_H_
#define NETWORK_CONTROL_H_

#include <Eigen/Eigen>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <quadrotor_msgs/SO3Command.h>
#include <tf/transform_datatypes.h>
#include <ros/ros.h>
#include <so3_control/SO3Control.h>
#include <so3_control/HGDO.h>
#include <so3_control/mavros_interface.h>
#include <quadrotor_msgs/SetTakeoffLand.h>
#include <string>
#include <iostream>
#include <boost/filesystem.hpp>
#include <regex>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <exception>

#define ONE_G 9.81

class NetworkControl
{
public:
    NetworkControl(ros::NodeHandle &node){
        nh_ = node;

        so3_controller_.setMass(mass_);
        disturbance_observer_ = HGDO(control_dt_);

        nh_.param("is_simulation", is_simulation_, false);
        nh_.param("use_disturbance_observer", use_disturbance_observer_, false);
        nh_.param("hover_thrust", hover_thrust_, 0.4);
        nh_.param("kx_xy", kx_xy, 5.7);
        nh_.param("kx_z", kx_z, 6.2);
        nh_.param("kv_xy", kv_xy, 3.4);
        nh_.param("kv_z", kv_z, 4.0);
        nh_.param("record_log", record_log_, false);
        nh_.param("logger_file_name", logger_file_name, std::string("/home/lu/"));
        nh_.param("position_cmd_timeout", position_cmd_timeout_, 0.30);
        nh_.param("simulation_takeoff_altitude", simulation_takeoff_altitude_, 2.0);
        nh_.param("takeoff_timeout", takeoff_timeout_, 8.0);
        nh_.param("takeoff_position_tolerance", takeoff_position_tolerance_, 0.15);
        nh_.param("takeoff_velocity_tolerance", takeoff_velocity_tolerance_, 0.20);
        nh_.param("takeoff_stable_time", takeoff_stable_time_, 0.20);
        if (!std::isfinite(position_cmd_timeout_) || position_cmd_timeout_ <= 0.0) {
            ROS_WARN("position_cmd_timeout must be positive; using 0.30 s");
            position_cmd_timeout_ = 0.30;
        }
        if (!std::isfinite(simulation_takeoff_altitude_)) {
            ROS_WARN("simulation_takeoff_altitude must be finite; using 2.0 m");
            simulation_takeoff_altitude_ = 2.0;
        }
        if (!std::isfinite(takeoff_timeout_) || takeoff_timeout_ <= 0.0) {
            ROS_WARN("takeoff_timeout must be positive; using 8.0 s");
            takeoff_timeout_ = 8.0;
        }
        takeoff_position_tolerance_ = std::max(0.01, takeoff_position_tolerance_);
        takeoff_velocity_tolerance_ = std::max(0.01, takeoff_velocity_tolerance_);
        takeoff_stable_time_ = std::max(control_dt_, takeoff_stable_time_);
        printf("kx: (%f, %f, %f), kv: (%f, %f, %f) \n", kx_xy, kx_xy, kx_z, kv_xy, kv_xy, kv_z);
        ROS_INFO("PositionCommand watchdog timeout: %.3f s", position_cmd_timeout_);

        so3_command_pub_ = nh_.advertise<quadrotor_msgs::SO3Command>("so3_cmd", 10);
        position_cmd_sub_ = nh_.subscribe("position_cmd", 1, &NetworkControl::network_cmd_callback, this, ros::TransportHints().tcpNoDelay());
        odom_sub_ = nh_.subscribe("odom", 1, &NetworkControl::odom_callback, this, ros::TransportHints().tcpNoDelay());
        imu_sub_ = nh_.subscribe("imu", 1, &NetworkControl::imu_callback, this, ros::TransportHints().tcpNoDelay());

        takeoff_land_control_timer = nh_.createTimer(ros::Duration(control_dt_), &NetworkControl::timerCallback, this);

        takeoff_land_srv = nh_.advertiseService("takeoff_land", &NetworkControl::takeoff_land_srv_handle, this);

        if (is_simulation_) {
            ROS_INFO("Automatic simulation takeoff will start once odometry is received");
        }
        
    };

    ~NetworkControl();

private:
    ros::NodeHandle nh_;
    ros::Publisher so3_command_pub_;
    ros::Subscriber position_cmd_sub_, odom_sub_, imu_sub_;
    ros::ServiceServer takeoff_land_srv;
    ros::Timer takeoff_land_control_timer;
    std::mutex mutex_;
    std::mutex logger_mutex_;
    std::mutex flight_task_mutex_;
    std::thread flight_task_thread_;

    double mass_ = 0.98;
    double control_dt_ = 0.02;
    double hover_thrust_ = 0.4;
    double kx_xy, kx_z, kv_xy, kv_z;
    double position_cmd_timeout_ = 0.30;
    double simulation_takeoff_altitude_ = 2.0;
    double takeoff_timeout_ = 8.0;
    double takeoff_position_tolerance_ = 0.15;
    double takeoff_velocity_tolerance_ = 0.20;
    double takeoff_stable_time_ = 0.20;
    
    double cur_yaw_ = 0;
    Eigen::Vector3d cur_pos_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d cur_vel_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d cur_acc_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Quaterniond cur_att_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d dis_acc_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d last_des_acc_ = Eigen::Vector3d(0, 0, 0);
    double last_thrust_ = 0;

    Eigen::Vector3d des_pos_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d des_vel_ = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d des_acc_ = Eigen::Vector3d(0, 0, 0);
    double des_yaw_ = 0;
    double des_yaw_dot_ = 0;

    bool is_simulation_ = false;
    std::atomic<bool> state_init_{false};
    std::atomic<bool> ref_valid_{false};
    std::atomic<bool> ctrl_valid_{false};
    std::atomic<bool> position_cmd_init_{false};
    std::atomic<bool> flight_task_running_{false};
    std::atomic<bool> shutdown_requested_{false};
    bool takeoff_cmd_init_ = false;
    bool use_disturbance_observer_ = false;
    bool record_log_ = false;
    bool watchdog_active_ = false;
    bool auto_takeoff_attempted_ = false;
    ros::WallTime last_position_cmd_time_;
    ros::WallTime last_odom_time_;
    
    SO3Control so3_controller_;
    HGDO disturbance_observer_;
    Mavros_Interface mavros_interface_;
    
    std::ofstream logger;
    std::string logger_file_name;

    void initLogRecorder();

    void recordLog(Eigen::Vector3d &cur_v, Eigen::Vector3d &cur_a, Eigen::Vector3d &des_a, Eigen::Vector3d &dis_a, double cur_yaw, double des_yaw);

    Eigen::Vector3d publishHoverSO3Command(Eigen::Vector3d des_pos, Eigen::Vector3d des_vel, Eigen::Vector3d des_acc, double des_yaw, double des_yaw_dot);

    Eigen::Vector3d get_Q_from_ACC(const Eigen::Vector3d &ref_acc, double ref_yaw, Eigen::Quaterniond &quat_des, Eigen::Vector3d &force_des);

    Eigen::Vector3d pub_SO3_command(Eigen::Vector3d ref_acc, double ref_yaw, double cur_yaw);

    void limite_acc(Eigen::Vector3d &acc);

    void network_cmd_callback(const quadrotor_msgs::PositionCommand::ConstPtr &cmd);

    void odom_callback(const nav_msgs::Odometry::ConstPtr &odom);

    void imu_callback(const sensor_msgs::Imu &imu);

    void timerCallback(const ros::TimerEvent&);

    // mavros interface
    bool takeoff_land_srv_handle(quadrotor_msgs::SetTakeoffLand::Request &req,
                                 quadrotor_msgs::SetTakeoffLand::Response &res);

    bool arm_disarm_vehicle(bool arm);

    bool start_flight_task(quadrotor_msgs::SetTakeoffLand::Request request,
                           const std::string &source);

    void maybe_start_simulation_takeoff();

    void takeoff_land_thread(quadrotor_msgs::SetTakeoffLand::Request request);
};

#endif
