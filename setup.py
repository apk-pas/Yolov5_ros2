from setuptools import setup,find_packages
from glob import glob
import os

package_name = 'yolov5_ros2'

setup(
    name=package_name,
    version='0.0.0',
    # packages=[package_name],
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', 'ament_index', 'resource_index', 'packages'),
            [os.path.join('resource', package_name)]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/**')),
        (os.path.join('share', package_name, 'resource'), glob('resource/**')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='apk',
    maintainer_email='1991170159@qq.com',
    description='yolov5 with ros2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "yolo_detect_2d=yolov5_ros2.yolo_detect_2d:main"
        ],
    },
)
