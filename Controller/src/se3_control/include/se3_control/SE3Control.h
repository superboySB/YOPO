#ifndef SE3_CONTROL_SE3_CONTROL_H_
#define SE3_CONTROL_SE3_CONTROL_H_

#include <Eigen/Geometry>
#include <se3_control/FlatnessMap.h>
#include <cmath>
#include <limits>

class SE3Control
{
public:
  SE3Control();

  void setMass(double mass);
  void setGravity(double g);
  void setPosition(const Eigen::Vector3d& position);
  void setVelocity(const Eigen::Vector3d& velocity);
  void setDrag(double horizontal_drag, double vertical_drag,
               double parasitic_drag, double speed_smooth_factor);
  void setLimits(double max_tilt_angle, double min_thrust, double max_thrust);

  void calculateControl(const Eigen::Vector3d& des_pos,
                        const Eigen::Vector3d& des_vel,
                        const Eigen::Vector3d& des_acc,
                        const Eigen::Vector3d& des_jerk,
                        double des_yaw,
                        double des_yaw_dot,
                        const Eigen::Vector3d& kx,
                        const Eigen::Vector3d& kv);

  const Eigen::Vector3d& getComputedForce() const;
  const Eigen::Quaterniond& getComputedOrientation() const;
  const Eigen::Vector3d& getComputedBodyRate() const;
  const Eigen::Vector3d& getCommandAcceleration() const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  Eigen::Vector3d limitTilt(const Eigen::Vector3d& acc) const;
  static bool finiteVector(const Eigen::Vector3d& vec);

  double mass_{0.98};
  double g_{9.81};
  double horizontal_drag_{0.70};
  double vertical_drag_{0.80};
  double parasitic_drag_{0.01};
  double speed_smooth_factor_{1.0e-4};
  double max_tilt_angle_{1.20};
  double min_thrust_{2.0};
  double max_thrust_{20.0};

  Eigen::Vector3d pos_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel_{Eigen::Vector3d::Zero()};

  Eigen::Vector3d force_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d body_rate_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d command_acc_{Eigen::Vector3d::Zero()};
};

#endif
