from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'naviguard_confidence'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maxx',
    maintainer_email='maxx@naviguard.org',
    description='NAVIGUARD Confidence and Autonomous Decision Layer for Outdoor UGV',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'confidence_node = naviguard_confidence.confidence_node:main',
        ],
    },
)
