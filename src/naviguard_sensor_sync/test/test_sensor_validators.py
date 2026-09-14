"""Unit tests for sensor data format and intrinsic validators."""

import pytest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, Imu

from naviguard_sensor_sync.sensor_validators import (
    validate_camera_info,
    validate_imu,
    validate_odometry,
)


def test_camera_info_validation_valid():
    """Verify valid CameraInfo message passes validation."""
    info = CameraInfo()
    info.header.frame_id = "camera_optical_link"
    info.width = 640
    info.height = 480
    info.k = [381.36, 0.0, 320.0, 0.0, 381.36, 240.0, 0.0, 0.0, 1.0]

    img = Image()
    img.width = 640
    img.height = 480

    report = validate_camera_info(info, img)
    assert report.is_valid is True
    assert report.status == "OK"
    assert report.fx == pytest.approx(381.36)
    assert report.dimension_match is True


def test_camera_info_validation_invalid_intrinsics():
    """Verify CameraInfo with invalid focal length or out-of-bounds center fails."""
    info = CameraInfo()
    info.width = 640
    info.height = 480
    # Negative fx and cx out of bounds
    info.k = [-10.0, 0.0, 999.0, 0.0, 381.36, 240.0, 0.0, 0.0, 1.0]

    report = validate_camera_info(info)
    assert report.is_valid is False
    assert report.status == "FAIL"


def test_camera_info_image_dimension_mismatch():
    """Verify dimension mismatch between image and camera_info is detected."""
    info = CameraInfo()
    info.width = 640
    info.height = 480
    info.k = [381.36, 0.0, 320.0, 0.0, 381.36, 240.0, 0.0, 0.0, 1.0]

    img = Image()
    img.width = 1280
    img.height = 720

    report = validate_camera_info(info, img)
    assert report.is_valid is False
    assert report.dimension_match is False
    assert report.status == "FAIL"


def test_imu_validation_orientation_unavailable():
    """Verify standard 6-axis IMU without orientation is flagged UNAVAILABLE but valid."""
    imu = Imu()
    imu.header.frame_id = "imu_link"
    imu.angular_velocity.x = 0.01
    imu.angular_velocity.y = -0.01
    imu.angular_velocity.z = 0.00
    imu.linear_acceleration.x = 0.0
    imu.linear_acceleration.y = 0.0
    imu.linear_acceleration.z = 9.81
    # Orientation unset (quaternion 0,0,0,0)
    imu.orientation_covariance[0] = -1.0

    report = validate_imu(imu)
    assert report.is_valid is True
    assert report.status == "OK"
    assert report.orientation_status == "UNAVAILABLE"
    assert report.linear_accel_norm == pytest.approx(9.81, abs=0.1)


def test_imu_validation_orientation_available():
    """Verify IMU with normalized orientation quaternion is marked AVAILABLE."""
    imu = Imu()
    imu.header.frame_id = "imu_link"
    imu.angular_velocity.z = 0.1
    imu.linear_acceleration.z = 9.81
    imu.orientation.w = 1.0  # Unit quaternion [0, 0, 0, 1]

    report = validate_imu(imu)
    assert report.is_valid is True
    assert report.status == "OK"
    assert report.orientation_status == "AVAILABLE"


def test_odometry_validation_valid():
    """Verify valid Odometry message with expected frames passes validation."""
    odom = Odometry()
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base_link"
    odom.pose.pose.orientation.w = 1.0
    odom.twist.twist.linear.x = 0.5

    report = validate_odometry(odom)
    assert report.is_valid is True
    assert report.status == "OK"
    assert report.header_frame_id == "odom"
    assert report.child_frame_id == "base_link"
    assert report.linear_speed == pytest.approx(0.5)


def test_odometry_validation_wrong_frames():
    """Verify Odometry with unexpected frame names fails validation."""
    odom = Odometry()
    odom.header.frame_id = "map"
    odom.child_frame_id = "chassis"

    report = validate_odometry(odom)
    assert report.is_valid is False
    assert report.status == "FAIL"
