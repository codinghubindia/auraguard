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

    sequence_arg = DeclareLaunchArgument(
        'sequence',
        default_value='00000',
        description='Sequence ID under evaluation'
    )

    output_arg = DeclareLaunchArgument(
        'output_file',
        default_value='rellis_evaluation_results.json',
        description='Output path for evaluation JSON metrics'
    )

    replay_node = Node(
        package='naviguard_rellis',
        executable='rellis_replay_node',
        name='rellis_replay_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'sequence': LaunchConfiguration('sequence'),
            }
        ]
    )

    evaluator_node = Node(
        package='naviguard_rellis',
        executable='rellis_evaluator_node',
        name='rellis_evaluator_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'sequence': LaunchConfiguration('sequence'),
                'metrics_output_file': LaunchConfiguration('output_file'),
            }
        ]
    )

    return LaunchDescription([
        params_arg,
        sequence_arg,
        output_arg,
        replay_node,
        evaluator_node
    ])
