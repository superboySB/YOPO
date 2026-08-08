#include "so3_control/NetworkControl.h"

NetworkControl::~NetworkControl()
{
    shutdown_requested_.store(true);
    std::lock_guard<std::mutex> task_lock(flight_task_mutex_);
    if (flight_task_thread_.joinable())
        flight_task_thread_.join();
}

void NetworkControl::initLogRecorder()
{
    std::lock_guard<std::mutex> logger_lock(logger_mutex_);
    // Use file count as name to avoid date confusion
    std::cout << "logger_file_name: " << logger_file_name << std::endl;
    int max_number = -1;
    boost::filesystem::path dir(logger_file_name);
    if (boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir)) {
        std::regex filename_pattern(R"(^(\d+)_.*$)"); // 匹配以数字开头，并以"_"分隔的文件名
        for (const auto& entry : boost::filesystem::directory_iterator(dir)) {
            if (boost::filesystem::is_regular_file(entry)) {
                std::string filename = entry.path().filename().string();
                std::smatch match;
                if (std::regex_match(filename, match, filename_pattern)) {
                    int number = std::stoi(match[1]); // 提取第一个"_"前的数字
                    max_number = std::max(max_number, number);
                }
            }
        }
    } else {
        std::cerr << "Error: invalid logger path" << std::endl;
    }
    std::string fileCountStr = std::to_string(max_number + 1);
    std::string temp_file_name = logger_file_name + fileCountStr + "_Net_logger_";
    time_t timep;
    timep = time(0);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y_%m_%d_%H_%M_%S", localtime(&timep));
    temp_file_name += tmp;
    temp_file_name += ".csv";
    if (logger.is_open())
    {
        logger.close();
    }
    logger.open(temp_file_name.c_str(), std::ios::out);
    std::cout << "logger: " << temp_file_name << std::endl;
    if (!logger.is_open())
    {
        std::cout << "cannot open the logger." << std::endl;
    }
    else
    {
        logger << "timestamp" << ',';
        logger << "cur_px" << ',';
        logger << "cur_py" << ',';
        logger << "cur_pz" << ',';
        logger << "cur_vx" << ',';
        logger << "cur_vy" << ',';
        logger << "cur_vz" << ',';
        logger << "cur_ax" << ',';
        logger << "cur_ay" << ',';
        logger << "cur_az" << ',';
        logger << "des_px" << ',';
        logger << "des_py" << ',';
        logger << "des_pz" << ',';
        logger << "des_vx" << ',';
        logger << "des_vy" << ',';
        logger << "des_vz" << ',';
        logger << "des_ax" << ',';
        logger << "des_ay" << ',';
        logger << "des_az" << ',';
        logger << "dis_ax" << ',';
        logger << "dis_ay" << ',';
        logger << "dis_az" << ',';
        logger << "px4_ax" << ',';
        logger << "px4_ay" << ',';
        logger << "px4_az" << ',';
        logger << "thrust" << ',';
        logger << "cur_yaw" << ',';
        logger << "des_yaw" << std::endl;
    }
}

void NetworkControl::recordLog(Eigen::Vector3d &cur_v, Eigen::Vector3d &cur_a, Eigen::Vector3d &des_a, Eigen::Vector3d &dis_a, double cur_yaw, double des_yaw)
{
    Eigen::Vector3d current_position;
    Eigen::Vector3d desired_position;
    Eigen::Vector3d desired_velocity;
    {
        std::lock_guard<std::mutex> state_lock(mutex_);
        current_position = cur_pos_;
        desired_position = des_pos_;
        desired_velocity = des_vel_;
    }

    std::lock_guard<std::mutex> logger_lock(logger_mutex_);
    if (logger.is_open())
    {
        logger << ros::Time::now().toNSec() << ',';
        logger << current_position(0) << ',';
        logger << current_position(1) << ',';
        logger << current_position(2) << ',';
        logger << cur_v(0) << ',';
        logger << cur_v(1) << ',';
        logger << cur_v(2) << ',';
        logger << cur_a(0) << ',';
        logger << cur_a(1) << ',';
        logger << cur_a(2) << ',';
        logger << desired_position(0) << ',';
        logger << desired_position(1) << ',';
        logger << desired_position(2) << ',';
        logger << desired_velocity(0) << ',';
        logger << desired_velocity(1) << ',';
        logger << desired_velocity(2) << ',';
        logger << des_a(0) << ',';
        logger << des_a(1) << ',';
        logger << des_a(2) << ',';
        logger << dis_a(0) << ',';
        logger << dis_a(1) << ',';
        logger << dis_a(2) << ',';
        logger << des_a(0) - dis_a(0) << ',';
        logger << des_a(1) - dis_a(1) << ',';
        logger << des_a(2) - dis_a(2) << ',';
        logger << last_thrust_ << ',';
        logger << cur_yaw << ',';
        logger << des_yaw << std::endl;
    }
}

Eigen::Vector3d NetworkControl::publishHoverSO3Command(Eigen::Vector3d des_pos, Eigen::Vector3d des_vel, 
                                                       Eigen::Vector3d des_acc, double des_yaw, double des_yaw_dot)
{
    Eigen::Vector3d kx(kx_xy, kx_xy, kx_z);
    Eigen::Vector3d kv(kv_xy, kv_xy, kv_z);
    so3_controller_.calculateControl(des_pos, des_vel, des_acc, des_yaw, des_yaw_dot, kx, kv);

    Eigen::Vector3d force = so3_controller_.getComputedForce();
    Eigen::Quaterniond orientation = so3_controller_.getComputedOrientation();

    quadrotor_msgs::SO3Command::Ptr so3_command(new quadrotor_msgs::SO3Command); //! @note memory leak?
    so3_command->header.stamp = ros::Time::now();
    so3_command->force.x = force(0);
    so3_command->force.y = force(1);
    so3_command->force.z = force(2);
    so3_command->orientation.x = orientation.x();
    so3_command->orientation.y = orientation.y();
    so3_command->orientation.z = orientation.z();
    so3_command->orientation.w = orientation.w();
    so3_command->kR[0] = 1.5;
    so3_command->kR[1] = 1.5;
    so3_command->kR[2] = 1.0;
    so3_command->kOm[0] = 0.13;
    so3_command->kOm[1] = 0.13;
    so3_command->kOm[2] = 0.1;
    so3_command->aux.current_yaw = cur_yaw_;
    so3_command->aux.enable_motors = true;
    so3_command_pub_.publish(so3_command);

    double thrust_norm = force.norm() / (mass_ * ONE_G) * hover_thrust_;
    mavros_interface_.pub_att_thrust_cmd(orientation, thrust_norm);
    last_thrust_ = thrust_norm;

    double thrust = force.norm() / mass_;
    Eigen::Matrix3d Cbn;    
    get_dcm_from_q(Cbn, orientation);
    Eigen::Vector3d att_acc = Eigen::Vector3d(0, 0, thrust);
    att_acc = Cbn * att_acc;
    att_acc(2) -= ONE_G;
    // std::cout<<"att_acc"<<att_acc.transpose()<<std::endl;
    return att_acc;
}

Eigen::Vector3d NetworkControl::get_Q_from_ACC(const Eigen::Vector3d &ref_acc, double ref_yaw, Eigen::Quaterniond &quat_des, Eigen::Vector3d &force_des)
{
    Eigen::Vector3d force_ = mass_ * ONE_G * Eigen::Vector3d(0, 0, 1);
    force_.noalias() += mass_ * ref_acc;

    // Limit control angle to theta degree
    double theta = M_PI / 4;
    double c = cos(theta);
    Eigen::Vector3d f;
    f.noalias() = force_ - mass_ * ONE_G * Eigen::Vector3d(0, 0, 1);
    if (Eigen::Vector3d(0, 0, 1).dot(force_ / force_.norm()) < c)
    {
        double nf = f.norm();
        double A = c * c * nf * nf - f(2) * f(2);
        double B = 2 * (c * c - 1) * f(2) * mass_ * ONE_G;
        double C = (c * c - 1) * mass_ * mass_ * ONE_G * ONE_G;
        double s = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
        force_.noalias() = s * f + mass_ * ONE_G * Eigen::Vector3d(0, 0, 1);
    }

    Eigen::Vector3d b1c, b2c, b3c;
    Eigen::Vector3d b1d(cos(ref_yaw), sin(ref_yaw), 0);

    if (force_.norm() > 1e-6)
        b3c.noalias() = force_.normalized();
    else
        b3c.noalias() = Eigen::Vector3d(0, 0, 1);

    b2c.noalias() = b3c.cross(b1d).normalized();
    b1c.noalias() = b2c.cross(b3c).normalized();

    Eigen::Matrix3d R;
    R << b1c, b2c, b3c;

    quat_des = Eigen::Quaterniond(R);
    force_des = force_;

    Eigen::Vector3d acc_actual = force_des / mass_ - Eigen::Vector3d(0, 0, ONE_G);
    return acc_actual;
}

// 世界系的期望加速度：ref_acc（加上g）、期望yaw：ref_yaw
Eigen::Vector3d NetworkControl::pub_SO3_command(Eigen::Vector3d ref_acc, double ref_yaw, double cur_yaw)
{
    Eigen::Vector3d force;
    Eigen::Quaterniond quat_des;
    Eigen::Vector3d acc_actual = get_Q_from_ACC(ref_acc, ref_yaw, quat_des, force);
    quadrotor_msgs::SO3Command::Ptr so3_command(new quadrotor_msgs::SO3Command);
    so3_command->header.stamp = ros::Time::now();
    so3_command->force.x = force(0);
    so3_command->force.y = force(1);
    so3_command->force.z = force(2);
    so3_command->orientation.x = quat_des.x();
    so3_command->orientation.y = quat_des.y();
    so3_command->orientation.z = quat_des.z();
    so3_command->orientation.w = quat_des.w();
    so3_command->kR[0] = 1.5;
    so3_command->kR[1] = 1.5;
    so3_command->kR[2] = 1.0;
    so3_command->kOm[0] = 0.13;
    so3_command->kOm[1] = 0.13;
    so3_command->kOm[2] = 0.1;
    so3_command->aux.current_yaw = cur_yaw;
    so3_command->aux.enable_motors = true;
    so3_command_pub_.publish(so3_command);

    double thrust_norm = force.norm() / (mass_ * ONE_G) * hover_thrust_;
    mavros_interface_.pub_att_thrust_cmd(quat_des, thrust_norm);
    last_thrust_ = thrust_norm;
    return acc_actual;
}

void NetworkControl::limite_acc(Eigen::Vector3d &acc){
    return;
    double max_norm = 10.0;
    double norm = acc.norm();
    if (norm > max_norm) {
        acc = acc / norm * max_norm;
    }
}

void NetworkControl::network_cmd_callback(const quadrotor_msgs::PositionCommand::ConstPtr &cmd)
{
    if (!ctrl_valid_.load())
        return;

    bool arm_state = false;
    bool ofb_enable = false;
    mavros_interface_.get_status(arm_state, ofb_enable);
    if (!arm_state || !ofb_enable)
        return;

    bool recovered_from_watchdog = false;
    {
        std::lock_guard<std::mutex> state_lock(mutex_);
        recovered_from_watchdog = watchdog_active_;
        watchdog_active_ = false;
        last_position_cmd_time_ = ros::WallTime::now();
        des_pos_ = Eigen::Vector3d(cmd->position.x, cmd->position.y, cmd->position.z);
        des_vel_ = Eigen::Vector3d(cmd->velocity.x, cmd->velocity.y, cmd->velocity.z);
    }
    position_cmd_init_.store(true);
    if (recovered_from_watchdog)
        ROS_INFO("PositionCommand stream recovered; planner control resumed");

    Eigen::Vector3d des_acc = Eigen::Vector3d(cmd->acceleration.x, cmd->acceleration.y, cmd->acceleration.z);
    limite_acc(des_acc);

    double des_yaw = cmd->yaw;
    
    disturbance_observer_.HGDO_ext_force_ob(last_des_acc_, cur_vel_, dis_acc_);
    // ROS_INFO_THROTTLE(0.5, "dis_acc: %.3f, %.3f, %.3f", dis_acc_.x(), dis_acc_.y(), dis_acc_.z());
    // std::cout << "dis_acc: " << dis_acc_.transpose() << std::endl;

    Eigen::Vector3d att_acc;
    if (cmd->trajectory_flag == quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY)
    {    
        if (use_disturbance_observer_)
            att_acc = des_acc - dis_acc_;
        else
            att_acc = des_acc;
        att_acc = pub_SO3_command(att_acc, des_yaw, cur_yaw_);
        // std::cout<<"acc: "<<des_acc.transpose()<<"   yaw:"<<des_yaw<<std::endl;
        if (record_log_)
            recordLog(cur_vel_, cur_acc_, des_acc, dis_acc_, cur_yaw_, des_yaw);
    }
    else
    {
        double des_yaw_dot = cmd->yaw_dot;
        att_acc = publishHoverSO3Command(des_pos_, des_vel_, des_acc, des_yaw, des_yaw_dot);
        if (record_log_)
            recordLog(cur_vel_, cur_acc_, att_acc, dis_acc_, cur_yaw_, des_yaw);
    }

    last_des_acc_ = att_acc;
}

void NetworkControl::odom_callback(const nav_msgs::Odometry::ConstPtr &odom)
{
    {
        std::lock_guard<std::mutex> state_lock(mutex_);
        cur_yaw_ = tf::getYaw(odom->pose.pose.orientation);
        cur_vel_ = Eigen::Vector3d(odom->twist.twist.linear.x, odom->twist.twist.linear.y, odom->twist.twist.linear.z);

        cur_pos_ = Eigen::Vector3d(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
        cur_att_.w() = odom->pose.pose.orientation.w;
        cur_att_.x() = odom->pose.pose.orientation.x;
        cur_att_.y() = odom->pose.pose.orientation.y;
        cur_att_.z() = odom->pose.pose.orientation.z;
        last_odom_time_ = ros::WallTime::now();
    }

    // if(!is_simulation_)
    //     cur_acc_ = Eigen::Vector3d(odom->twist.twist.angular.x, odom->twist.twist.angular.y, odom->twist.twist.angular.z);

    so3_controller_.setPosition(cur_pos_);
    so3_controller_.setVelocity(cur_vel_);
    if (!state_init_.exchange(true))
        ROS_INFO("Odom Recived! Ready to TakeOff...");
}

void NetworkControl::imu_callback(const sensor_msgs::Imu &imu)
{
    Eigen::Vector3d acc(imu.linear_acceleration.x,
                        imu.linear_acceleration.y,
                        imu.linear_acceleration.z);
    if (is_simulation_)
    {
        cur_acc_ = acc;
    }
    else
    {
        Eigen::Vector3d acc_world = cur_att_ * acc;
        acc_world(2) -= 9.8;
        cur_acc_ = acc_world;
    }

    // so3_controller_.setAcc(acc_world);
}

void NetworkControl::timerCallback(const ros::TimerEvent &)
{
    maybe_start_simulation_takeoff();

    if (!state_init_.load() || !ref_valid_.load())
        return;

    bool watchdog_just_triggered = false;
    double command_age = 0.0;
    Eigen::Vector3d des_pos_temp;
    Eigen::Vector3d des_vel_temp;
    Eigen::Vector3d des_acc_temp;
    double des_yaw_temp = 0.0;
    double des_yaw_dot_temp = 0.0;
    {
        std::lock_guard<std::mutex> state_lock(mutex_);
        if (position_cmd_init_.load() && ctrl_valid_.load() && !watchdog_active_)
        {
            command_age = (ros::WallTime::now() - last_position_cmd_time_).toSec();
            if (command_age <= position_cmd_timeout_)
                return;

            des_pos_ = cur_pos_;
            des_vel_.setZero();
            des_acc_.setZero();
            des_yaw_ = cur_yaw_;
            des_yaw_dot_ = 0.0;
            watchdog_active_ = true;
            watchdog_just_triggered = true;
        }

        des_pos_temp = des_pos_;
        des_vel_temp = des_vel_;
        des_acc_temp = des_acc_;
        des_yaw_temp = des_yaw_;
        des_yaw_dot_temp = des_yaw_dot_;
    }

    if (watchdog_just_triggered)
    {
        ROS_ERROR("PositionCommand watchdog timeout after %.3f s; holding world position "
                  "(%.3f, %.3f, %.3f)",
                  command_age, des_pos_temp.x(), des_pos_temp.y(), des_pos_temp.z());
    }

    Eigen::Vector3d att_acc = publishHoverSO3Command(des_pos_temp, des_vel_temp, des_acc_temp,
                                                     des_yaw_temp, des_yaw_dot_temp);

    if (takeoff_cmd_init_)
    {
        disturbance_observer_.HGDO_ext_force_ob(last_des_acc_, cur_vel_, dis_acc_);
        // ROS_INFO_THROTTLE(1.0, " dis_acc: (%f, %f, %f)", dis_acc_.x(), dis_acc_.y(), dis_acc_.z());
    }

    last_des_acc_ = att_acc;
    if (record_log_)
        recordLog(cur_vel_, cur_acc_, att_acc, dis_acc_, cur_yaw_, des_yaw_);
    takeoff_cmd_init_ = true;
}

bool NetworkControl::takeoff_land_srv_handle(quadrotor_msgs::SetTakeoffLand::Request &req,
                                              quadrotor_msgs::SetTakeoffLand::Response &res)
{
    if (!state_init_.load())
    {
        ROS_WARN("Rejecting takeoff/land request before first odometry");
        res.res = false;
        return true;
    }

    if (req.takeoff && !std::isfinite(req.takeoff_altitude))
    {
        ROS_WARN("Rejecting takeoff request with a non-finite target altitude");
        res.res = false;
        return true;
    }

    if (is_simulation_)
        auto_takeoff_attempted_ = true;

    // Copy the callback-local request into the worker. The service response only
    // reports whether the asynchronous task was accepted, not flight completion.
    res.res = start_flight_task(req, "service");
    return true;
}

bool NetworkControl::start_flight_task(quadrotor_msgs::SetTakeoffLand::Request request,
                                       const std::string &source)
{
    std::lock_guard<std::mutex> task_lock(flight_task_mutex_);
    if (flight_task_running_.load())
    {
        ROS_WARN("Rejecting %s takeoff/land request: another flight task is running", source.c_str());
        return false;
    }

    if (flight_task_thread_.joinable())
        flight_task_thread_.join();

    flight_task_running_.store(true);
    try
    {
        flight_task_thread_ = std::thread(&NetworkControl::takeoff_land_thread, this, request);
    }
    catch (const std::exception &error)
    {
        flight_task_running_.store(false);
        ROS_ERROR("Failed to start %s takeoff/land task: %s", source.c_str(), error.what());
        return false;
    }

    ROS_INFO("Accepted %s %s task", source.c_str(), request.takeoff ? "takeoff" : "landing");
    return true;
}

void NetworkControl::maybe_start_simulation_takeoff()
{
    if (!is_simulation_ || auto_takeoff_attempted_ || !state_init_.load())
        return;

    auto_takeoff_attempted_ = true;
    quadrotor_msgs::SetTakeoffLand::Request request;
    request.takeoff = true;
    request.takeoff_altitude = simulation_takeoff_altitude_;
    if (!start_flight_task(request, "automatic simulation"))
        ROS_ERROR("Automatic simulation takeoff could not be started");
}

void NetworkControl::takeoff_land_thread(quadrotor_msgs::SetTakeoffLand::Request request)
{
    const double target_altitude = request.takeoff_altitude;
    {
        std::lock_guard<std::mutex> state_lock(mutex_);
        des_pos_ = cur_pos_;
        des_vel_.setZero();
        des_acc_.setZero();
        des_yaw_ = cur_yaw_;
        des_yaw_dot_ = 0.0;
        watchdog_active_ = false;
    }
    position_cmd_init_.store(false);
    ctrl_valid_.store(false);
    ref_valid_.store(true);

    if (request.takeoff)
    {
        std::cout << "takeoff process start" << std::endl;
        if (!arm_disarm_vehicle(true))
        {
            std::cout << "Service failed because cannot Arm!" << std::endl;
            flight_task_running_.store(false);
            return;
        }
        ros::WallDuration(1.0).sleep();

        const double takeoff_vel = 0.8;
        const double takeoff_dz = takeoff_vel * control_dt_;
        const int required_stable_cycles = std::max(
            1, static_cast<int>(std::ceil(takeoff_stable_time_ / control_dt_)));
        int stable_cycles = 0;
        bool takeoff_succeeded = false;
        ros::WallRate takeoff_loop(1.0 / control_dt_);
        std::cout << "takeoff altitude: " << target_altitude << " m" << std::endl;
        std::cout << "takeoff velocity: " << takeoff_vel << " m/s" << std::endl;
        const ros::WallTime start_takeoff_task_time = ros::WallTime::now();
        while (ros::ok() && !shutdown_requested_.load() &&
               (ros::WallTime::now() - start_takeoff_task_time).toSec() < takeoff_timeout_)
        {
            double actual_z = 0.0;
            double actual_vz = 0.0;
            double desired_z = 0.0;
            bool odom_fresh = false;
            {
                std::lock_guard<std::mutex> state_lock(mutex_);
                const double altitude_error = target_altitude - des_pos_(2);
                if (fabs(altitude_error) <= takeoff_dz)
                    des_pos_(2) = target_altitude;
                else
                    des_pos_(2) += altitude_error > 0.0 ? takeoff_dz : -takeoff_dz;

                desired_z = des_pos_(2);
                actual_z = cur_pos_(2);
                actual_vz = cur_vel_(2);
                odom_fresh = !last_odom_time_.isZero() &&
                             (ros::WallTime::now() - last_odom_time_).toSec() <= 0.5;
            }

            const bool desired_at_target = fabs(desired_z - target_altitude) <= 1e-6;
            const bool actual_settled = fabs(actual_z - target_altitude) <= takeoff_position_tolerance_ &&
                                        fabs(actual_vz) <= takeoff_velocity_tolerance_;
            stable_cycles = desired_at_target && actual_settled && odom_fresh ? stable_cycles + 1 : 0;
            if (stable_cycles >= required_stable_cycles)
            {
                ROS_INFO("TakeOff Done! Ready to Flight... actual_z=%.3f m, vz=%.3f m/s",
                         actual_z, actual_vz);
                ctrl_valid_.store(true);
                takeoff_succeeded = true;
                break;
            }

            takeoff_loop.sleep();
        }

        if (!takeoff_succeeded)
        {
            double actual_z = 0.0;
            double actual_vz = 0.0;
            {
                std::lock_guard<std::mutex> state_lock(mutex_);
                actual_z = cur_pos_(2);
                actual_vz = cur_vel_(2);
            }
            ctrl_valid_.store(false);
            ROS_ERROR("Takeoff failed to settle within %.1f s: target_z=%.3f m, actual_z=%.3f m, "
                      "vz=%.3f m/s; external PositionCommand remains disabled",
                      takeoff_timeout_, target_altitude, actual_z, actual_vz);
        }
    }
    else
    {
        ctrl_valid_.store(false);
        const double land_vel = -0.4;
        const double land_ddz = land_vel * control_dt_;
        ros::WallRate land_loop(1.0 / control_dt_);
        const ros::WallTime start_land_task_time = ros::WallTime::now();
        while (ros::ok() && !shutdown_requested_.load() &&
               (ros::WallTime::now() - start_land_task_time).toSec() < 8.0)
        {
            double actual_z = 0.0;
            double actual_vz = 0.0;
            {
                std::lock_guard<std::mutex> state_lock(mutex_);
                des_pos_(2) += land_ddz;
                actual_z = cur_pos_(2);
                actual_vz = cur_vel_(2);
            }

            if (fabs(actual_z) < 0.1f && fabs(actual_vz) < 1.0f)
            {
                ROS_INFO("detect land: disarm");
                arm_disarm_vehicle(false);
                break;
            }
            land_loop.sleep();
        }
    }
    ROS_INFO("take off thread out");
    flight_task_running_.store(false);
    return;
}

bool NetworkControl::arm_disarm_vehicle(bool arm)
{
    if (arm)
    {   
        if (!state_init_.load()){
            ROS_WARN("State timeout, will not arm!");
            return false;
        }

        ROS_INFO("UAV will be armed!");
        if (is_simulation_)
            mavros_interface_.set_arm_and_offboard_manually();
        else if (mavros_interface_.set_arm_and_offboard())
            ROS_INFO("Arm done!");
        else{
            ROS_ERROR("Arm failure!");
            return false;
        }
        if (record_log_)
            initLogRecorder();
    }
    else
    {
        ROS_INFO("UAV will be disarmed!");
        if (is_simulation_)
            mavros_interface_.set_disarm_manually();
        else if (mavros_interface_.set_disarm())
            ROS_INFO("Disarm done!");
        else {
            ROS_ERROR("Disarm failure!");
            return false;
        }
        if (record_log_)
        {
            std::lock_guard<std::mutex> logger_lock(logger_mutex_);
            logger.close();
        }
    }
    return true;
}
