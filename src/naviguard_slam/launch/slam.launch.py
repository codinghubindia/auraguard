import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_slam = get_package_share_directory('naviguard_slam')
    default_params_file = os.path.join(pkg_slam, 'config', 'slam_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to YAML parameter file for SLAM node',
    )

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='mapping',
        description='SLAM operating mode: mapping or localization',
    )

    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=os.path.join(pkg_slam, 'maps', 'naviguard_slam_map.json'),
        description='Full path to saved map file for save/load operations',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true',
    )

    slam_node = Node(
        package='naviguard_slam',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'mode': LaunchConfiguration('mode'),
                'map_file': LaunchConfiguration('map_file'),
            },
        ],
    )

    return LaunchDescription([
        params_file_arg,
        mode_arg,
        map_file_arg,
        use_sim_time_arg,
        slam_node,
    ])
