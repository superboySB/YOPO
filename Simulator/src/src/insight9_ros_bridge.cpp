#include <linux/videodev2.h>

#include <Insight_9_receive.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>

namespace {

ros::Publisher depth_pub;
ros::Publisher imu_pub;
ros::Publisher vio_pub;

void imageCallback(int cam_id, uint8_t *data, size_t size, int width, int height,
                   unsigned int format, uint64_t, uint64_t, void *) {
    // SDK v1.1 uses cam_id=2 for depth. Some published guide revisions say 3;
    // pixel format is therefore the authoritative discriminator.
    if (format != V4L2_PIX_FMT_Z16 || (cam_id != 2 && cam_id != 3))
        return;
    const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uint16_t);
    if (data == nullptr || size < expected) {
        ROS_WARN_THROTTLE(1.0, "Insight 9 returned a truncated Z16 frame: %zu < %zu", size, expected);
        return;
    }
    sensor_msgs::Image message;
    message.header.stamp = ros::Time::now();
    message.header.frame_id = "insight9_optical_frame";
    message.height = height;
    message.width = width;
    message.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
    message.is_bigendian = false;
    message.step = width * sizeof(uint16_t);
    message.data.assign(data, data + expected);  // SDK buffer is reused after callback.
    depth_pub.publish(message);
}

void imuCallback(float ax, float ay, float az, float gx, float gy, float gz,
                 uint64_t, void *) {
    sensor_msgs::Imu message;
    message.header.stamp = ros::Time::now();
    message.header.frame_id = "insight9_imu_frame";
    message.orientation_covariance[0] = -1.0;
    message.linear_acceleration.x = ax;
    message.linear_acceleration.y = ay;
    message.linear_acceleration.z = az;
    message.angular_velocity.x = gx;
    message.angular_velocity.y = gy;
    message.angular_velocity.z = gz;
    imu_pub.publish(message);
}

void vioCallback(float px, float py, float pz, float qx, float qy, float qz,
                 float qw, uint64_t, void *) {
    nav_msgs::Odometry message;
    message.header.stamp = ros::Time::now();
    message.header.frame_id = "insight9_vio_world";
    message.child_frame_id = "insight9_vio_frame";
    message.pose.pose.position.x = px;
    message.pose.pose.position.y = py;
    message.pose.pose.position.z = pz;
    message.pose.pose.orientation.x = qx;
    message.pose.pose.orientation.y = qy;
    message.pose.pose.orientation.z = qz;
    message.pose.pose.orientation.w = qw;
    vio_pub.publish(message);
}

}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "insight9_ros_bridge");
    ros::NodeHandle node;
    depth_pub = node.advertise<sensor_msgs::Image>("/depth_image", 1);
    imu_pub = node.advertise<sensor_msgs::Imu>("/insight9/imu", 10);
    vio_pub = node.advertise<nav_msgs::Odometry>("/insight9/vio", 10);

    if (insight9_receive_init_default() != 0) {
        ROS_FATAL("Insight 9 SDK initialization failed; check /dev/video*, /dev/hidraw* and permissions");
        return 2;
    }
    if (insight9_receive_set_camera_fps(2, 15) != 0)
        ROS_WARN("Could not request the documented 15 Hz depth rate; continuing with the device rate");
    insight9_receive_register_image_callback(imageCallback, nullptr);
    insight9_receive_register_imu_callback(imuCallback, nullptr);
    insight9_receive_register_vio_callback(vioCallback, nullptr);
    if (insight9_receive_start() != 0) {
        ROS_FATAL("Insight 9 SDK failed to start acquisition");
        insight9_receive_cleanup();
        return 3;
    }

    ROS_INFO("Insight 9 bridge ready: Z16 /depth_image, /insight9/imu, /insight9/vio");
    ros::spin();
    insight9_receive_stop();
    insight9_receive_cleanup();
    return 0;
}
