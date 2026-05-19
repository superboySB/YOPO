#include "quadrotor_msgs/encode_msgs.h"
#include <quadrotor_msgs/comm_types.h>

namespace quadrotor_msgs
{

void encodeSO3Command(const quadrotor_msgs::SO3Command &attitude_command,
                      std::vector<uint8_t> &output)
{
  struct SO3_CMD_INPUT attitude_cmd_input;

  attitude_cmd_input.force[0] = attitude_command.force.x*500;
  attitude_cmd_input.force[1] = attitude_command.force.y*500;
  attitude_cmd_input.force[2] = attitude_command.force.z*500;

  attitude_cmd_input.des_qx = attitude_command.orientation.x*125;
  attitude_cmd_input.des_qy = attitude_command.orientation.y*125;
  attitude_cmd_input.des_qz = attitude_command.orientation.z*125;
  attitude_cmd_input.des_qw = attitude_command.orientation.w*125;

  attitude_cmd_input.kR[0] = attitude_command.kR[0]*50;
  attitude_cmd_input.kR[1] = attitude_command.kR[1]*50;
  attitude_cmd_input.kR[2] = attitude_command.kR[2]*50;

  attitude_cmd_input.kOm[0] = attitude_command.kOm[0]*100;
  attitude_cmd_input.kOm[1] = attitude_command.kOm[1]*100;
  attitude_cmd_input.kOm[2] = attitude_command.kOm[2]*100;

  attitude_cmd_input.cur_yaw = attitude_command.aux.current_yaw*1e4;

  attitude_cmd_input.kf_correction = attitude_command.aux.kf_correction*1e11;
  attitude_cmd_input.angle_corrections[0] = attitude_command.aux.angle_corrections[0]*2500;
  attitude_cmd_input.angle_corrections[1] = attitude_command.aux.angle_corrections[1]*2500;

  attitude_cmd_input.enable_motors = attitude_command.aux.enable_motors;
  attitude_cmd_input.use_external_yaw = attitude_command.aux.use_external_yaw;

  attitude_cmd_input.seq = attitude_command.header.seq % 255;

  output.resize(sizeof(attitude_cmd_input));
  memcpy(&output[0], &attitude_cmd_input, sizeof(attitude_cmd_input));
}

void encodeTRPYCommand(const quadrotor_msgs::TRPYCommand &trpy_command,
                        std::vector<uint8_t> &output)
{
  struct TRPY_CMD trpy_cmd_input;
  trpy_cmd_input.thrust = trpy_command.thrust*1e4;
  trpy_cmd_input.roll = trpy_command.roll*1e4;
  trpy_cmd_input.pitch = trpy_command.pitch*1e4;
  trpy_cmd_input.yaw = trpy_command.yaw*1e4;
  trpy_cmd_input.current_yaw = trpy_command.aux.current_yaw*1e4;
  trpy_cmd_input.enable_motors = trpy_command.aux.enable_motors;
  trpy_cmd_input.use_external_yaw = trpy_command.aux.use_external_yaw;

  output.resize(sizeof(trpy_cmd_input));
  memcpy(&output[0], &trpy_cmd_input, sizeof(trpy_cmd_input));
}

void encodePPRGains(const quadrotor_msgs::Gains &gains,
                    std::vector<uint8_t> &output)
{
  struct PPR_GAINS ppr_gains;
  ppr_gains.Kp = gains.Kp;
  ppr_gains.Kd = gains.Kd;
  ppr_gains.Kp_yaw = gains.Kp_yaw;
  ppr_gains.Kd_yaw = gains.Kd_yaw;

  output.resize(sizeof(ppr_gains));
  memcpy(&output[0], &ppr_gains, sizeof(ppr_gains));
}

}
