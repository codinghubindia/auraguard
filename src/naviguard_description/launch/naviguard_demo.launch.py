# Master integration launch file for NAVIGUARD demonstration
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_naviguard_desc = get_package_share_directory('naviguard_description')
    pkg_perception = get_package_share_directory('naviguard_perception')
    pkg_vo = get_package_share_directory('naviguard_visual_odometry')
    pkg_sync = get_package_share_directory('naviguard_sensor_sync')
    pkg_state_est = get_package_share_directory('naviguard_state_estimation')
    pkg_slam = get_package_share_directory('naviguard_slam')
    pkg_conf = get_package_share_directory('naviguard_confidence')
    pkg_rec = get_package_share_directory('naviguard_recovery')
    pkg_nav = get_package_share_directory('naviguard_navigation')
    pkg_dashboard = get_package_share_directory('naviguard_dashboard')

    # Enable D3D12 NVIDIA GPU acceleration in WSL2 if available
    if os.path.exists('/usr/lib/wsl/lib'):
        os.environ['GALLIUM_DRIVER'] = 'd3d12'
        os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME'] = 'NVIDIA'
        os.environ['LD_LIBRARY_PATH'] = f"/usr/lib/wsl/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '0'

    default_rviz_config = os.path.join(pkg_naviguard_desc, 'config', 'naviguard_final.rviz')
    default_world = os.path.join(pkg_naviguard_desc, 'worlds', 'rellis_outdoor_world.sdf')

    # Launch arguments
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo headless (server only without GUI)',
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='Path to SDF world file',
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock',
    )

    start_demo_motion_arg = DeclareLaunchArgument(
        'start_demo_motion',
        default_value='false',
        description='Start deterministic demo motion controller',
    )

    start_navigation_arg = DeclareLaunchArgument(
        'start_navigation',
        default_value='true',
        description='Start Phase 9 autonomous navigation stack',
    )

    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz',
        default_value='false',
        description='Launch RViz2 with NAVIGUARD demo display configuration',
    )

    start_dashboard_arg = DeclareLaunchArgument(
        'start_dashboard',
        default_value='true',
        description='Start NAVIGUARD operator web dashboard on port 8080',
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_config,
        description='Full path to RViz configuration file',
    )

    # 1. Gazebo Harmonic + Robot State Publisher + Bridge
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_naviguard_desc, 'launch', 'naviguard_sim.launch.py')
        ),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'world': LaunchConfiguration('world'),
        }.items(),
    )

    # 2. Perception
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_perception, 'launch', 'perception.launch.py')
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    # 3. Visual Odometry
    vo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_vo, 'launch', 'visual_odometry.launch.py')
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    # 4. Sensor Synchronization
    sensor_sync_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_sync, 'launch', 'sensor_sync.launch.py')
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    # 5. State Estimation
    state_est_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_state_est, 'launch', 'state_estimation.launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'publish_tf': 'false',
        }.items(),
    )

    # 6. Visual SLAM
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_slam, 'launch', 'slam.launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'mode': 'mapping',
        }.items(),
    )

    # 7. Confidence Engine
    confidence_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_conf, 'launch', 'confidence.launch.py')
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    # 8. Autonomous Recovery Node
    recovery_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_rec, 'launch', 'recovery.launch.py')
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    # 9. Phase 9 Mission & Global Navigation
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav, 'launch', 'navigation.launch.py')
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
        condition=IfCondition(LaunchConfiguration('start_navigation')),
    )

    # 10. Demo Motion Controller
    demo_motion_node = Node(
        package='naviguard_recovery',
        executable='naviguard_demo_motion',
        name='naviguard_demo_motion',
        output='screen',
        parameters=[
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'linear_velocity': 0.25,
                'angular_velocity': 0.25,
                'pattern': 'loop',
            }
        ],
        condition=IfCondition(LaunchConfiguration('start_demo_motion')),
    )

    # 11. Operator Web Dashboard
    dashboard_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dashboard, 'launch', 'dashboard.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('start_dashboard')),
    )

    # 12. RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('enable_rviz')),
    )

    return LaunchDescription([
        headless_arg,
        world_arg,
        use_sim_time_arg,
        start_demo_motion_arg,
        start_navigation_arg,
        start_dashboard_arg,
        enable_rviz_arg,
        rviz_config_arg,
        sim_launch,
        perception_launch,
        vo_launch,
        sensor_sync_launch,
        state_est_launch,
        slam_launch,
        confidence_launch,
        recovery_launch,
        navigation_launch,
        dashboard_launch,
        demo_motion_node,
        rviz_node,
    ])
