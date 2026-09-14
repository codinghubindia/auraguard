from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'naviguard_visual_odometry'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maxx',
    maintainer_email='maxx@todo.todo',
    description='Temporal feature tracking and visual motion estimation foundation for NAVIGUARD UGV',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visual_odometry_node = naviguard_visual_odometry.visual_odometry_node:main',
        ],
    },
)
