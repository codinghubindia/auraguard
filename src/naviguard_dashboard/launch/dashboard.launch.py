import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dash = get_package_share_directory('naviguard_dashboard')
    default_params = os.path.join(pkg_dash, 'config', 'dashboard_params.yaml')

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Full path to YAML parameter file for dashboard node'
    )

    port_arg = DeclareLaunchArgument(
        'port',
        default_value='8080',
        description='HTTP Web Server port for dashboard'
    )

    dash_node = Node(
        package='naviguard_dashboard',
        executable='dashboard_node',
        name='dashboard_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'port': LaunchConfiguration('port')}
        ]
    )

    return LaunchDescription([
        params_arg,
        port_arg,
        dash_node
    ])
