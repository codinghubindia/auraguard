import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_nav = get_package_share_directory('naviguard_navigation')
    default_params_file = os.path.join(pkg_nav, 'config', 'navigation_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to YAML parameter file for navigation node',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true',
    )

    nav_node = Node(
        package='naviguard_navigation',
        executable='navigation_node',
        name='navigation_node',
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
        nav_node,
    ])
