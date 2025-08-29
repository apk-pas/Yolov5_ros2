from setuptools import setup,find_packages
from glob import glob
import os

package_name = 'yolov5_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
    entry_points={
        'console_scripts': [
            "yolo_detect=yolov5_ros2.yolo_detect:main",
            "tf_publish=yolov5_ros2.tf_publish:main"
        ],
    },
)
