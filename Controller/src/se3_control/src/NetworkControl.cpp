#include "se3_control/NetworkControl.h"

void NetworkControl::initLogRecorder()
{   
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
    if (logger.is_open())
    {
        logger << ros::Time::now().toNSec() << ',';
        logger << cur_pos_(0) << ',';
        logger << cur_pos_(1) << ',';
        logger << cur_pos_(2) << ',';
        logger << cur_v(0) << ',';
        logger << cur_v(1) << ',';
        logger << cur_v(2) << ',';
        logger << cur_a(0) << ',';
        logger << cur_a(1) << ',';
        logger << cur_a(2) << ',';
        logger << des_pos_(0) << ',';
        logger << des_pos_(1) << ',';
        logger << des_pos_(2) << ',';
        logger << des_vel_(0) << ',';
        logger << des_vel_(1) << ',';
        logger << des_vel_(2) << ',';
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

Eigen::Vector3d NetworkControl::publishHoverSE3Command(Eigen::Vector3d des_pos, Eigen::Vector3d des_vel,
                                                       Eigen::Vector3d des_acc, double des_yaw, double des_yaw_dot)
{
    Eigen::Vector3d kx(kx_xy, kx_xy, kx_z);
    Eigen::Vector3d kv(kv_xy, kv_xy, kv_z);
    se3_controller_.calculateControl(des_pos, des_vel, des_acc, Eigen::Vector3d::Zero(), des_yaw, des_yaw_dot, kx, kv);

    Eigen::Vector3d force = se3_controller_.getComputedForce();
    Eigen::Quaterniond orientation = se3_controller_.getComputedOrientation();

    quadrotor_msgs::SO3Command::Ptr attitude_command(new quadrotor_msgs::SO3Command); //! @note memory leak?
    attitude_command->header.stamp = ros::Time::now();
    attitude_command->force.x = force(0);
    attitude_command->force.y = force(1);
    attitude_command->force.z = force(2);
    attitude_command->orientation.x = orientation.x();
    attitude_command->orientation.y = orientation.y();
    attitude_command->orientation.z = orientation.z();
    attitude_command->orientation.w = orientation.w();
    attitude_command->kR[0] = 1.5;
    attitude_command->kR[1] = 1.5;
    attitude_command->kR[2] = 1.0;
    attitude_command->kOm[0] = 0.13;
    attitude_command->kOm[1] = 0.13;
    attitude_command->kOm[2] = 0.1;
    attitude_command->aux.current_yaw = cur_yaw_;
    attitude_command->aux.enable_motors = true;
    attitude_command_pub_.publish(attitude_command);

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

Eigen::Vector3d NetworkControl::pub_SE3_command(const Eigen::Vector3d &des_pos, const Eigen::Vector3d &des_vel,
                                                const Eigen::Vector3d &des_acc, const Eigen::Vector3d &des_jerk,
                                                double des_yaw, double des_yaw_dot, double cur_yaw)
{
    Eigen::Vector3d kx = use_reference_feedback_
                             ? Eigen::Vector3d(kx_xy, kx_xy, kx_z)
                             : Eigen::Vector3d::Zero();
    Eigen::Vector3d kv = use_reference_feedback_
                             ? Eigen::Vector3d(kv_xy, kv_xy, kv_z)
                             : Eigen::Vector3d::Zero();
    se3_controller_.calculateControl(des_pos, des_vel, des_acc, des_jerk, des_yaw, des_yaw_dot, kx, kv);
    Eigen::Vector3d force = se3_controller_.getComputedForce();
    Eigen::Quaterniond quat_des = se3_controller_.getComputedOrientation();
    Eigen::Vector3d acc_actual = se3_controller_.getCommandAcceleration();
    quadrotor_msgs::SO3Command::Ptr attitude_command(new quadrotor_msgs::SO3Command);
    attitude_command->header.stamp = ros::Time::now();
    attitude_command->force.x = force(0);
    attitude_command->force.y = force(1);
    attitude_command->force.z = force(2);
    attitude_command->orientation.x = quat_des.x();
    attitude_command->orientation.y = quat_des.y();
    attitude_command->orientation.z = quat_des.z();
    attitude_command->orientation.w = quat_des.w();
    attitude_command->kR[0] = 1.5;
    attitude_command->kR[1] = 1.5;
    attitude_command->kR[2] = 1.0;
    attitude_command->kOm[0] = 0.13;
    attitude_command->kOm[1] = 0.13;
    attitude_command->kOm[2] = 0.1;
    attitude_command->aux.current_yaw = cur_yaw;
    attitude_command->aux.enable_motors = true;
    attitude_command_pub_.publish(attitude_command);

    double thrust_norm = force.norm() / (mass_ * ONE_G) * hover_thrust_;
    mavros_interface_.pub_att_thrust_cmd(quat_des, thrust_norm);
    last_thrust_ = thrust_norm;
    return acc_actual;
}

void NetworkControl::limite_acc(Eigen::Vector3d &acc){
    if (!std::isfinite(acc(0)) || !std::isfinite(acc(1)) || !std::isfinite(acc(2)))
    {
        acc.setZero();
        return;
    }
    double max_norm = max_ref_acc_;
    double norm = acc.norm();
    if (max_norm > 1.0e-6 && norm > max_norm) {
        acc = acc / norm * max_norm;
    }
}

void NetworkControl::limit_vec(Eigen::Vector3d &vec, double max_norm){
    if (!std::isfinite(vec(0)) || !std::isfinite(vec(1)) || !std::isfinite(vec(2)))
    {
        vec.setZero();
        return;
    }
    const double norm = vec.norm();
    if (max_norm > 1.0e-6 && norm > max_norm) {
        vec = vec / norm * max_norm;
    }
}

void NetworkControl::network_cmd_callback(const quadrotor_msgs::PositionCommand::ConstPtr &cmd)
{
    if (!ctrl_valid_)
        return;

    bool arm_state = false;
    bool ofb_enable = false;
    mavros_interface_.get_status(arm_state, ofb_enable);
    if (!arm_state || !ofb_enable)
        return;

    position_cmd_init_ = true;

    des_pos_ = Eigen::Vector3d(cmd->position.x, cmd->position.y, cmd->position.z);
    des_vel_ = Eigen::Vector3d(cmd->velocity.x, cmd->velocity.y, cmd->velocity.z);
    Eigen::Vector3d des_acc = Eigen::Vector3d(cmd->acceleration.x, cmd->acceleration.y, cmd->acceleration.z);
    Eigen::Vector3d des_jerk = Eigen::Vector3d(cmd->jerk.x, cmd->jerk.y, cmd->jerk.z);
    limit_vec(des_vel_, max_ref_vel_);
    limite_acc(des_acc);
    limit_vec(des_jerk, max_ref_jerk_);

    double des_yaw = cmd->yaw;
    double des_yaw_dot = cmd->yaw_dot;
    
    disturbance_observer_.HGDO_ext_force_ob(last_des_acc_, cur_vel_, dis_acc_);
    // ROS_INFO_THROTTLE(0.5, "dis_acc: %.3f, %.3f, %.3f", dis_acc_.x(), dis_acc_.y(), dis_acc_.z());
    // std::cout << "dis_acc: " << dis_acc_.transpose() << std::endl;

    Eigen::Vector3d att_acc;
    if (cmd->trajectory_flag == quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY)
    {
        if (use_disturbance_observer_)
            des_acc = des_acc - dis_acc_;
        att_acc = pub_SE3_command(des_pos_, des_vel_, des_acc, des_jerk, des_yaw, des_yaw_dot, cur_yaw_);
        if (record_log_)
            recordLog(cur_vel_, cur_acc_, des_acc, dis_acc_, cur_yaw_, des_yaw);
    }
    else
    {
        att_acc = publishHoverSE3Command(des_pos_, des_vel_, des_acc, des_yaw, des_yaw_dot);
        if (record_log_)
            recordLog(cur_vel_, cur_acc_, att_acc, dis_acc_, cur_yaw_, des_yaw);
    }

    last_des_acc_ = att_acc;
}

void NetworkControl::odom_callback(const nav_msgs::Odometry::ConstPtr &odom)
{
    cur_yaw_ = tf::getYaw(odom->pose.pose.orientation);
    cur_vel_ = Eigen::Vector3d(odom->twist.twist.linear.x, odom->twist.twist.linear.y, odom->twist.twist.linear.z);

    cur_pos_ = Eigen::Vector3d(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
    cur_att_.w() = odom->pose.pose.orientation.w;
    cur_att_.x() = odom->pose.pose.orientation.x;
    cur_att_.y() = odom->pose.pose.orientation.y;
    cur_att_.z() = odom->pose.pose.orientation.z;

    // if(!is_simulation_)
    //     cur_acc_ = Eigen::Vector3d(odom->twist.twist.angular.x, odom->twist.twist.angular.y, odom->twist.twist.angular.z);

    se3_controller_.setPosition(cur_pos_);
    se3_controller_.setVelocity(cur_vel_);
    if (!state_init_)
        ROS_INFO("Odom Recived! Ready to TakeOff...");
    state_init_ = true;
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

    // se3_controller_.setAcc(acc_world);
}

void NetworkControl::timerCallback(const ros::TimerEvent &)
{
    if (!state_init_ || !ref_valid_)
        return;
    if (position_cmd_init_ && ctrl_valid_)
        return;

    mutex_.lock();
    Eigen::Vector3d des_pos_temp = des_pos_;
    mutex_.unlock();

    Eigen::Vector3d att_acc = publishHoverSE3Command(des_pos_temp, des_vel_, des_acc_, des_yaw_, des_yaw_dot_);

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

void NetworkControl::takeoff_land_thread(quadrotor_msgs::SetTakeoffLand::Request &req)
{
    mutex_.lock();
    float takeoff_altitude = req.takeoff_altitude;
    des_pos_ = cur_pos_;
    des_pos_(2) -= 0.2;
    des_vel_ = Eigen::Vector3d(0, 0, 0);
    des_yaw_ = cur_yaw_;
    mutex_.unlock();
    ref_valid_ = true;

    if (req.takeoff)
    {
        std::cout << "takeoff process start" << std::endl;
        if (!arm_disarm_vehicle(true))
        {
            std::cout << "Service failed because cannot Arm!" << std::endl;
            return;
        }
        sleep(1);

        double takeoff_vel = 0.8;
        double takeoff_ddz = takeoff_vel * control_dt_;
        ros::Rate takeoff_loop(1 / control_dt_);
        std::cout << "takeoff altitude: " << takeoff_altitude << " m" << std::endl;
        std::cout << "takeoff velocity: " << takeoff_vel << " m/s" << std::endl;
        ros::Time start_takeoff_task_time = ros::Time::now();
        while (ros::ok() && ros::Time::now() - start_takeoff_task_time < ros::Duration(8.0))
        {       
            mutex_.lock();
            des_pos_(2) += takeoff_ddz;
            mutex_.unlock();

            if (des_pos_(2) > takeoff_altitude)
            {
                ROS_INFO("TakeOff Done! Ready to Flight...");
                ctrl_valid_ = true;
                break;
            }
            takeoff_loop.sleep();
        }
    }
    else
    {
        ctrl_valid_ = false;
        double land_vel = -0.4;
        double land_ddz = land_vel * control_dt_;
        ros::Rate land_loop(1 / control_dt_);
        ros::Time start_land_task_time = ros::Time::now();
        while (ros::ok() && ros::Time::now() - start_land_task_time < ros::Duration(8.0))
        {
            mutex_.lock();
            des_pos_(2) += land_ddz;
            mutex_.unlock();

            if (fabs(cur_pos_(2)) < 0.1f && fabs(cur_vel_(2)) < 1.0f)
            {
                ROS_INFO("detect land: disarm");
                arm_disarm_vehicle(false);
                break;
            }
            land_loop.sleep();
        }
    }
    ROS_INFO("take off thread out");
    return;
}

bool NetworkControl::arm_disarm_vehicle(bool arm)
{
    if (arm)
    {   
        if (!state_init_){
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
            logger.close();
    }
    return true;
}
