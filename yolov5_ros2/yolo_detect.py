from yolov5 import YOLOv5
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import yaml
import os
import numpy as np

# 获取包路径
package_share_directory = get_package_share_directory('yolov5_ros2')

class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')

        # 声明核心参数
        self.declare_parameter("device", "cpu")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("camera_info_file", f"{package_share_directory}/config/camera_info.yaml")

        # 加载模型
        model_path = os.path.join(package_share_directory, "config", "best.pt")
        self.yolov5 = YOLOv5(model_path=model_path, device=self.get_parameter('device').value)

        # 创建发布器
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()

        # 订阅图像和相机信息
        self.image_sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self.image_callback, 10)

        # 加载相机参数
        with open(self.get_parameter('camera_info_file').value) as f:
            self.camera_info = yaml.full_load(f.read())

        # 解析相机内参为矩阵
        self.camera_matrix = np.array(self.camera_info["k"]).reshape(3, 3)
        self.dist_coeffs = np.array(self.camera_info["d"])

        self.bridge = CvBridge()
        
    def image_callback(self, msg: Image):
        """处理图像并发布检测结果"""
        # 转换图像格式
        image = self.bridge.imgmsg_to_cv2(msg)
        
        # 模型推理
        detect_result = self.yolov5.predict(image)
        predictions = detect_result.pred[0]
        if len(predictions) == 0:
            return  # 无检测结果时不发布

        # 初始化检测消息
        self.result_msg.header = msg.header  # 使用图像的header确保时间同步
        self.result_msg.detections.clear()

        # 解析检测结果并填充消息
        boxes = predictions[:, :4]  # x1,y1,x2,y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        for idx in range(len(categories)):
            class_name = detect_result.names[int(categories[idx])]
            x1, y1, x2, y2 = map(int, boxes[idx])
            
            # 创建Detection2D消息
            detection = Detection2D()
            detection.id = class_name
            detection.header = self.result_msg.header  # 每个检测结果关联相同header

            # 设置边界框
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            detection.bbox.center.position.x = center_x
            detection.bbox.center.position.y = center_y
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)

            # 设置置信度和位姿
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = float(scores[idx])
            
            # 准备PNP所需的3D和2D点
            object_points = np.array([
                [-0.02, -0.04, 0.0],  # 左下
                [0.02, -0.04, 0.0],   # 右下
                [0.02, 0.04, 0.0],    # 右上
                [-0.02, 0.04, 0.0]    # 左上
            ], dtype=np.float32)
            
            image_points = np.array([
                [x1, y1],  # 左下
                [x2, y1],  # 右下
                [x2, y2],  # 右上
                [x1, y2]   # 左上
            ], dtype=np.float32)
            
            # 执行PNP求解
            ret, rvec, tvec = cv2.solvePnP(
                object_points, 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            # 获取3D坐标
            if ret:
                world_x, world_y, world_z = tvec.flatten()
            else:
                world_x, world_y, world_z = 0.0, 0.0, 0.0
                self.get_logger().warn(f"PNP failed for {class_name}")
            
            hypothesis.pose.pose.position.x = world_x
            hypothesis.pose.pose.position.y = world_y
            hypothesis.pose.pose.position.z = world_z
            detection.results.append(hypothesis)

            self.result_msg.detections.append(detection)

        # 发布检测结果
        self.yolo_result_pub.publish(self.result_msg)

def main():
    rclpy.init()
    node = YoloV5Ros2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()