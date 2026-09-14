import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_rellis = get_package_share_directory('naviguard_rellis')
    default_params = os.path.join(pkg_rellis, 'config', 'rellis_params.yaml')

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to RELLIS parameter YAML'
    )

    data_root_arg = DeclareLaunchArgument(
        'data_root',
        default_value='~/datasets/rellis3d',
        description='Path to RELLIS-3D dataset root'
    )

    sequence_arg = DeclareLaunchArgument(
        'sequence',
        default_value='00000',
        description='Sequence ID to replay'
    )

    rate_arg = DeclareLaunchArgument(
        'replay_rate_hz',
        default_value='10.0',
        description='Replay publishing frequency in Hz'
    )

    replay_node = Node(
        package='naviguard_rellis',
        executable='rellis_replay_node',
        name='rellis_replay_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'data_root': LaunchConfiguration('data_root'),
                'sequence': LaunchConfiguration('sequence'),
                'replay_rate_hz': LaunchConfiguration('replay_rate_hz'),
            }
        ]
    )

    return LaunchDescription([
        params_arg,
        data_root_arg,
        sequence_arg,
        rate_arg,
        replay_node
    ])
