import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_rec = get_package_share_directory('naviguard_recovery')
    default_params_file = os.path.join(pkg_rec, 'config', 'recovery_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to YAML parameter file for recovery node',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true',
    )

    recovery_node = Node(
        package='naviguard_recovery',
        executable='recovery_node',
        name='recovery_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            },
        ],
    )

    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        recovery_node,
    ])
