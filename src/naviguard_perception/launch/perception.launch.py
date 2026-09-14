import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_naviguard_perception = get_package_share_directory('naviguard_perception')
    default_params_file = os.path.join(pkg_naviguard_perception, 'config', 'perception_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to YAML parameter file for perception node'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    perception_node = Node(
        package='naviguard_perception',
        executable='naviguard_perception_node',
        name='naviguard_perception_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        perception_node
    ])
