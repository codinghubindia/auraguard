import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def launch_setup(context, *args, **kwargs):
    pkg_naviguard = get_package_share_directory('naviguard_description')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    headless = LaunchConfiguration('headless').perform(context).lower() == 'true'
    world_file = LaunchConfiguration('world').perform(context)
    x_pos = LaunchConfiguration('x').perform(context)
    y_pos = LaunchConfiguration('y').perform(context)
    z_pos = LaunchConfiguration('z').perform(context)
    yaw_pos = LaunchConfiguration('yaw').perform(context)

    # Process Xacro
    xacro_file = os.path.join(pkg_naviguard, 'urdf', 'naviguard.urdf.xacro')
    robot_description_raw = xacro.process_file(xacro_file).toxml()

    # Robot State Publisher Node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_raw,
            'use_sim_time': True
        }]
    )

    # Gazebo Sim Launch
    gz_args = f'-r {world_file}'
    if headless:
        gz_args = f'-r -s {world_file}'

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    # Spawn NAVIGUARD UGV in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', 'naviguard',
            '-topic', 'robot_description',
            '-x', x_pos,
            '-y', y_pos,
            '-z', z_pos,
            '-Y', yaw_pos
        ]
    )

    # ROS-GZ Bridge Node
    bridge_config = os.path.join(pkg_naviguard, 'config', 'bridge.yaml')
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{'config_file': bridge_config}]
    )

    return [
        robot_state_publisher,
        gz_sim,
        spawn_robot,
        bridge
    ]


def generate_launch_description():
    pkg_naviguard = get_package_share_directory('naviguard_description')
    default_world = os.path.join(pkg_naviguard, 'worlds', 'rellis_outdoor_world.sdf')

    return LaunchDescription([
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run Gazebo headless (server only without GUI)'
        ),
        DeclareLaunchArgument(
            'world',
            default_value=default_world,
            description='Path to SDF world file'
        ),
        DeclareLaunchArgument(
            'x',
            default_value='0.0',
            description='Initial spawn X coordinate (meters)'
        ),
        DeclareLaunchArgument(
            'y',
            default_value='0.0',
            description='Initial spawn Y coordinate (meters)'
        ),
        DeclareLaunchArgument(
            'z',
            default_value='0.15',
            description='Initial spawn Z coordinate (meters)'
        ),
        DeclareLaunchArgument(
            'yaw',
            default_value='0.0',
            description='Initial spawn Yaw orientation (radians)'
        ),
        OpaqueFunction(function=launch_setup)
    ])
