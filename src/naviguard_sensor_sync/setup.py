from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'naviguard_sensor_sync'

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
    description='Sensor timestamp synchronization, rate monitoring, and frame calibration validation for NAVIGUARD UGV',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_sync_node = naviguard_sensor_sync.sensor_sync_node:main',
        ],
    },
)
