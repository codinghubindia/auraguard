import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_state_estimation = get_package_share_directory('naviguard_state_estimation')
    default_params_file = os.path.join(pkg_state_estimation, 'config', 'state_estimation_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to YAML parameter file for state estimation node',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true',
    )

    publish_tf_arg = DeclareLaunchArgument(
        'publish_tf',
        default_value='false',
        description='Publish odom -> base_link TF (set false to avoid conflict with diff-drive plugin)',
    )

    state_estimation_node = Node(
        package='naviguard_state_estimation',
        executable='state_estimation_node',
        name='state_estimation_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'publish_tf': LaunchConfiguration('publish_tf'),
            },
        ],
    )

    return LaunchDescription([
        params_file_arg,
        use_sim_time_arg,
        publish_tf_arg,
        state_estimation_node,
    ])
