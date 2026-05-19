#include <se3_control/SE3Control.h>

#include <algorithm>

SE3Control::SE3Control() = default;

void SE3Control::setMass(double mass)
{
  mass_ = mass;
}

void SE3Control::setGravity(double g)
{
  g_ = g;
}

void SE3Control::setPosition(const Eigen::Vector3d& position)
{
  pos_ = position;
}

void SE3Control::setVelocity(const Eigen::Vector3d& velocity)
{
  vel_ = velocity;
}

void SE3Control::setDrag(double horizontal_drag, double vertical_drag,
                         double parasitic_drag, double speed_smooth_factor)
{
  horizontal_drag_ = horizontal_drag;
  vertical_drag_ = vertical_drag;
  parasitic_drag_ = parasitic_drag;
  speed_smooth_factor_ = std::max(speed_smooth_factor, 1.0e-9);
}

void SE3Control::setLimits(double max_tilt_angle, double min_thrust,
                           double max_thrust)
{
  max_tilt_angle_ = std::max(1.0e-3, std::min(max_tilt_angle, M_PI / 2.0 - 1.0e-3));
  min_thrust_ = std::max(0.0, min_thrust);
  max_thrust_ = std::max(min_thrust_, max_thrust);
}

bool SE3Control::finiteVector(const Eigen::Vector3d& vec)
{
  return std::isfinite(vec(0)) && std::isfinite(vec(1)) && std::isfinite(vec(2));
}

Eigen::Vector3d SE3Control::limitTilt(const Eigen::Vector3d& acc) const
{
  Eigen::Vector3d force = mass_ * (acc + g_ * Eigen::Vector3d::UnitZ());
  if (force.norm() < 1.0e-9)
  {
    return Eigen::Vector3d::Zero();
  }

  const double c = std::cos(max_tilt_angle_);
  if (Eigen::Vector3d::UnitZ().dot(force.normalized()) >= c)
  {
    return acc;
  }

  Eigen::Vector3d lateral_force(force(0), force(1), 0.0);
  const double lateral_norm = lateral_force.norm();
  if (lateral_norm < 1.0e-9)
  {
    return acc;
  }

  const double vertical_force = std::max(force(2), 1.0e-6);
  const double max_lateral = vertical_force * std::tan(max_tilt_angle_);
  lateral_force *= max_lateral / lateral_norm;
  force(0) = lateral_force(0);
  force(1) = lateral_force(1);
  return force / mass_ - g_ * Eigen::Vector3d::UnitZ();
}

void SE3Control::calculateControl(const Eigen::Vector3d& des_pos,
                                  const Eigen::Vector3d& des_vel,
                                  const Eigen::Vector3d& des_acc,
                                  const Eigen::Vector3d& des_jerk,
                                  double des_yaw,
                                  double des_yaw_dot,
                                  const Eigen::Vector3d& kx,
                                  const Eigen::Vector3d& kv)
{
  Eigen::Vector3d acc_cmd = finiteVector(des_acc) ? des_acc : Eigen::Vector3d::Zero();
  if (finiteVector(des_pos))
  {
    acc_cmd.noalias() += kx.asDiagonal() * (des_pos - pos_) / mass_;
  }
  if (finiteVector(des_vel))
  {
    acc_cmd.noalias() += kv.asDiagonal() * (des_vel - vel_) / mass_;
  }
  acc_cmd = limitTilt(acc_cmd);
  command_acc_ = acc_cmd;

  Eigen::Vector3d jerk_cmd = finiteVector(des_jerk) ? des_jerk : Eigen::Vector3d::Zero();
  Eigen::Vector3d vel_cmd = finiteVector(des_vel) ? des_vel : Eigen::Vector3d::Zero();
  if (!std::isfinite(des_yaw))
  {
    des_yaw = 0.0;
  }
  if (!std::isfinite(des_yaw_dot))
  {
    des_yaw_dot = 0.0;
  }

  se3_control::FlatnessMap flatmap;
  flatmap.reset(mass_, g_, horizontal_drag_, vertical_drag_,
                parasitic_drag_, speed_smooth_factor_);

  double thrust = 0.0;
  Eigen::Vector4d quat_wxyz;
  flatmap.forward(vel_cmd, acc_cmd, jerk_cmd, des_yaw, des_yaw_dot,
                  thrust, quat_wxyz, body_rate_);

  thrust = std::max(min_thrust_, std::min(thrust, max_thrust_));
  orientation_ = Eigen::Quaterniond(quat_wxyz(0), quat_wxyz(1),
                                    quat_wxyz(2), quat_wxyz(3));
  orientation_.normalize();
  force_ = orientation_ * Eigen::Vector3d::UnitZ() * thrust;
}

const Eigen::Vector3d& SE3Control::getComputedForce() const
{
  return force_;
}

const Eigen::Quaterniond& SE3Control::getComputedOrientation() const
{
  return orientation_;
}

const Eigen::Vector3d& SE3Control::getComputedBodyRate() const
{
  return body_rate_;
}

const Eigen::Vector3d& SE3Control::getCommandAcceleration() const
{
  return command_acc_;
}
