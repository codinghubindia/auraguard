from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'naviguard_rellis'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz') + glob('config/*.json')),
        (os.path.join('share', package_name, 'config', 'scenarios'), glob('config/scenarios/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NAVIGUARD Team',
    maintainer_email='maxx@naviguard.org',
    description='NAVIGUARD RELLIS-3D Dataset Replay, Realistic Off-Road Environment, and Evaluation Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rellis_replay_node = naviguard_rellis.image_replayer:main',
            'rellis_evaluator_node = naviguard_rellis.rellis_evaluator_node:main',
            'dataset_validator = naviguard_rellis.dataset_validator:main',
        ],
    },
)
