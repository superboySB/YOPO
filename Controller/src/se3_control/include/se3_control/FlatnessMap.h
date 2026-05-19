/*
 * MIT License
 *
 * This flatness map follows the GCOPTER SE(3) differential-flatness mapping.
 * It maps flat outputs (velocity, acceleration, jerk, yaw, yaw-rate) to total
 * thrust, attitude, and body rate. Only the forward map is needed by YOPO's
 * online controller.
 */

#ifndef SE3_CONTROL_FLATNESS_MAP_H_
#define SE3_CONTROL_FLATNESS_MAP_H_

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>

namespace se3_control
{
class FlatnessMap
{
public:
  inline void reset(const double vehicle_mass,
                    const double gravitational_acceleration,
                    const double horizontal_drag_coeff,
                    const double vertical_drag_coeff,
                    const double parasitic_drag_coeff,
                    const double speed_smooth_factor)
  {
    mass_ = vehicle_mass;
    grav_ = gravitational_acceleration;
    dh_ = horizontal_drag_coeff;
    dv_ = vertical_drag_coeff;
    cp_ = parasitic_drag_coeff;
    veps_ = speed_smooth_factor;
  }

  inline void forward(const Eigen::Vector3d& vel,
                      const Eigen::Vector3d& acc,
                      const Eigen::Vector3d& jer,
                      const double yaw,
                      const double yaw_dot,
                      double& thrust,
                      Eigen::Vector4d& quat,
                      Eigen::Vector3d& body_rate) const
  {
    const double v0 = vel(0);
    const double v1 = vel(1);
    const double v2 = vel(2);
    const double a0 = acc(0);
    const double a1 = acc(1);
    const double a2 = acc(2);

    const double cp_term = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2 + veps_);
    const double w_term = 1.0 + cp_ * cp_term;
    const double w0 = w_term * v0;
    const double w1 = w_term * v1;
    const double w2 = w_term * v2;
    const double dh_over_m = dh_ / mass_;

    const double zu0 = a0 + dh_over_m * w0;
    const double zu1 = a1 + dh_over_m * w1;
    const double zu2 = a2 + dh_over_m * w2 + grav_;
    const double zu_sqr0 = zu0 * zu0;
    const double zu_sqr1 = zu1 * zu1;
    const double zu_sqr2 = zu2 * zu2;
    const double zu01 = zu0 * zu1;
    const double zu12 = zu1 * zu2;
    const double zu02 = zu0 * zu2;
    const double zu_sqr_norm = zu_sqr0 + zu_sqr1 + zu_sqr2;
    const double zu_norm = std::sqrt(std::max(zu_sqr_norm, 1.0e-12));
    const double z0 = zu0 / zu_norm;
    const double z1 = zu1 / zu_norm;
    const double z2 = zu2 / zu_norm;

    const double ng_den = zu_sqr_norm * zu_norm;
    const double ng00 = (zu_sqr1 + zu_sqr2) / ng_den;
    const double ng01 = -zu01 / ng_den;
    const double ng02 = -zu02 / ng_den;
    const double ng11 = (zu_sqr0 + zu_sqr2) / ng_den;
    const double ng12 = -zu12 / ng_den;
    const double ng22 = (zu_sqr0 + zu_sqr1) / ng_den;

    const double v_dot_a = v0 * a0 + v1 * a1 + v2 * a2;
    const double dw_term = cp_ * v_dot_a / cp_term;
    const double dw0 = w_term * a0 + dw_term * v0;
    const double dw1 = w_term * a1 + dw_term * v1;
    const double dw2 = w_term * a2 + dw_term * v2;
    const double dz_term0 = jer(0) + dh_over_m * dw0;
    const double dz_term1 = jer(1) + dh_over_m * dw1;
    const double dz_term2 = jer(2) + dh_over_m * dw2;
    const double dz0 = ng00 * dz_term0 + ng01 * dz_term1 + ng02 * dz_term2;
    const double dz1 = ng01 * dz_term0 + ng11 * dz_term1 + ng12 * dz_term2;
    const double dz2 = ng02 * dz_term0 + ng12 * dz_term1 + ng22 * dz_term2;

    const double f_term0 = mass_ * a0 + dv_ * w0;
    const double f_term1 = mass_ * a1 + dv_ * w1;
    const double f_term2 = mass_ * (a2 + grav_) + dv_ * w2;
    thrust = z0 * f_term0 + z1 * f_term1 + z2 * f_term2;

    const double tilt_den = std::sqrt(std::max(2.0 * (1.0 + z2), 1.0e-12));
    const double tilt0 = 0.5 * tilt_den;
    const double tilt1 = -z1 / tilt_den;
    const double tilt2 = z0 / tilt_den;
    const double c_half_yaw = std::cos(0.5 * yaw);
    const double s_half_yaw = std::sin(0.5 * yaw);
    quat(0) = tilt0 * c_half_yaw;
    quat(1) = tilt1 * c_half_yaw + tilt2 * s_half_yaw;
    quat(2) = tilt2 * c_half_yaw - tilt1 * s_half_yaw;
    quat(3) = tilt0 * s_half_yaw;

    const double c_yaw = std::cos(yaw);
    const double s_yaw = std::sin(yaw);
    const double omg_den = std::max(z2 + 1.0, 1.0e-6);
    const double omg_term = dz2 / omg_den;
    body_rate(0) = dz0 * s_yaw - dz1 * c_yaw -
                   (z0 * s_yaw - z1 * c_yaw) * omg_term;
    body_rate(1) = dz0 * c_yaw + dz1 * s_yaw -
                   (z0 * c_yaw + z1 * s_yaw) * omg_term;
    body_rate(2) = (z1 * dz0 - z0 * dz1) / omg_den + yaw_dot;
  }

private:
  double mass_{0.98};
  double grav_{9.81};
  double dh_{0.70};
  double dv_{0.80};
  double cp_{0.01};
  double veps_{1.0e-4};
};
}  // namespace se3_control

#endif
