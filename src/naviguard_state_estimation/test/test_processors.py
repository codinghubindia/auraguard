import pytest
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

from naviguard_state_estimation.imu_processor import ImuProcessor, ImuProcessorConfig
from naviguard_state_estimation.wheel_odom_processor import WheelOdomProcessor, WheelOdomProcessorConfig


def test_imu_processor_valid_data():
    processor = ImuProcessor()
    msg = Imu()
    msg.header.stamp.sec = 10
    msg.header.stamp.nanosec = 500000000  # 10.5 s
    msg.header.frame_id = "imu_link"

    msg.angular_velocity.x = 0.01
    msg.angular_velocity.y = -0.02
    msg.angular_velocity.z = 0.35
    msg.linear_acceleration.x = 0.5
    msg.linear_acceleration.y = 0.1
    msg.linear_acceleration.z = 9.81
    msg.angular_velocity_covariance[8] = 0.002
    msg.linear_acceleration_covariance[0] = 0.01
    msg.orientation_covariance[0] = -1.0  # unoriented 6-axis IMU

    meas = processor.process(msg)
    assert meas.is_valid is True
    assert pytest.approx(meas.stamp_sec, rel=1e-6) == 10.5
    assert meas.data['wz'] == 0.35
    assert meas.data['wz_cov'] == 0.002
    assert meas.data['has_orientation'] == False


def test_imu_processor_gating():
    processor = ImuProcessor(ImuProcessorConfig(max_angular_velocity=5.0, max_linear_acceleration=30.0))

    # Test angular velocity exceedance
    msg_fast = Imu()
    msg_fast.header.stamp.sec = 1
    msg_fast.angular_velocity.z = 10.0  # exceeds 5.0
    meas = processor.process(msg_fast)
    assert meas.is_valid is False
    assert "ANGULAR_VELOCITY_EXCEEDS_LIMIT" in meas.rejection_reason

    # Test linear acceleration exceedance
    msg_accel = Imu()
    msg_accel.header.stamp.sec = 2
    msg_accel.linear_acceleration.z = 50.0  # exceeds 30.0
    meas = processor.process(msg_accel)
    assert meas.is_valid is False
    assert "LINEAR_ACCELERATION_EXCEEDS_LIMIT" in meas.rejection_reason

    # Test NaN rejection
    msg_nan = Imu()
    msg_nan.header.stamp.sec = 3
    msg_nan.angular_velocity.x = float('nan')
    meas = processor.process(msg_nan)
    assert meas.is_valid is False
    assert "NAN" in meas.rejection_reason


def test_wheel_odom_processor_valid():
    processor = WheelOdomProcessor()
    msg = Odometry()
    msg.header.stamp.sec = 5
    msg.header.stamp.nanosec = 0
    msg.header.frame_id = "odom"
    msg.child_frame_id = "base_link"

    msg.pose.pose.position.x = 2.0
    msg.pose.pose.position.y = 1.0
    # 90 degrees yaw rotation: z = sin(pi/4) = 0.7071068, w = cos(pi/4) = 0.7071068
    msg.pose.pose.orientation.z = np.sin(np.pi / 4.0)
    msg.pose.pose.orientation.w = np.cos(np.pi / 4.0)

    msg.twist.twist.linear.x = 0.6
    msg.twist.twist.linear.y = 0.02
    msg.twist.twist.angular.z = 0.15
    msg.twist.covariance[0] = 0.01
    msg.twist.covariance[35] = 0.02

    meas = processor.process(msg)
    assert meas.is_valid is True
    assert meas.data['vx'] == 0.6
    assert pytest.approx(meas.data['yaw'], rel=1e-3) == np.pi / 2.0
    assert meas.data['pos_x'] == 2.0
    assert meas.data['pos_y'] == 1.0


def test_wheel_odom_frame_mismatch_and_speed_limits():
    processor = WheelOdomProcessor(WheelOdomProcessorConfig(max_linear_velocity=2.0, max_lateral_velocity=0.3))

    # Frame mismatch
    msg_bad_frame = Odometry()
    msg_bad_frame.header.stamp.sec = 1
    msg_bad_frame.header.frame_id = "map"
    msg_bad_frame.child_frame_id = "base_link"
    meas = processor.process(msg_bad_frame)
    assert meas.is_valid is False
    assert "FRAME_MISMATCH" in meas.rejection_reason

    # Excess speed
    msg_fast = Odometry()
    msg_fast.header.stamp.sec = 2
    msg_fast.header.frame_id = "odom"
    msg_fast.child_frame_id = "base_link"
    msg_fast.twist.twist.linear.x = 5.0
    meas = processor.process(msg_fast)
    assert meas.is_valid is False
    assert "LINEAR_VELOCITY_EXCEEDS_LIMIT" in meas.rejection_reason

    # Excess lateral velocity (non-holonomic violation)
    msg_lateral = Odometry()
    msg_lateral.header.stamp.sec = 3
    msg_lateral.header.frame_id = "odom"
    msg_lateral.child_frame_id = "base_link"
    msg_lateral.twist.twist.linear.x = 0.5
    msg_lateral.twist.twist.linear.y = 0.8
    meas = processor.process(msg_lateral)
    assert meas.is_valid is False
    assert "LATERAL_VELOCITY_EXCEEDS_NON_HOLONOMIC_LIMIT" in meas.rejection_reason
