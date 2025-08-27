from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_dir = get_package_share_directory('yolov5_ros2')
    camera_info_path = os.path.join(package_dir, 'config', 'camera_info.yaml')

    return LaunchDescription([
        # 图像处理节点
        Node(
            package='yolov5_ros2',
            executable='image_processor',
            name='image_processor',
            output='screen',
            parameters=[{
                'device': 'cuda',  # 可改为'cpu'
                'image_topic': '/image_raw',
                'camera_info_file': camera_info_path,
                'show_result': False,
                'pub_result_img': True
            }]
        ),
        # TF发布节点
        Node(
            package='yolov5_ros2',
            executable='tf_publisher',
            name='tf_publisher',
            output='screen',
            parameters=[{
                'base_frame': 'base_link'
            }]
        )
    ])