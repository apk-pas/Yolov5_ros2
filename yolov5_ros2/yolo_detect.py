from math import frexp
from traceback import print_tb
from torch import imag
from yolov5 import YOLOv5
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import yaml
from yolov5_ros2.cv_tool import px2xy
import os

# 获取ROS发行版版本并设置YoloV5配置文件的共享目录
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')

# 创建ROS 2节点类YoloDetectorNode
class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detect')

        # 声明ROS参数
        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="计算设备选择，默认: cpu,可选: cuda:0"))

        self.declare_parameter("model", "yolov5s", ParameterDescriptor(
            name="model", description="默认模型选择: yolov5s"))

        self.declare_parameter("image_topic", "/image_raw", ParameterDescriptor(
            name="image_topic", description="图像话题，默认: /image_raw"))
        
        self.declare_parameter("camera_info_topic", "/camera/camera_info", ParameterDescriptor(
            name="camera_info_topic", description="相机信息话题，默认: /camera/camera_info"))

        # 从camera_info话题读取参数（如果可用），否则使用文件定义的参数
        self.declare_parameter("camera_info_file", f"{package_share_directory}/config/camera_info.yaml", ParameterDescriptor(
            name="camera_info", description=f"相机信息文件路径，默认: {package_share_directory}/config/camera_info.yaml"))

        # 默认不显示检测结果
        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="是否显示检测结果，默认: False"))

        # 默认不发布检测结果图像
        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="是否发布检测结果图像，默认: False"))

        # 1. 加载模型
        model_path = package_share_directory + "/config/" + "best.pt"
        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model_path, device=device)

        # 2. 创建发布器
        self.detection_pub = self.create_publisher(
            Detection2DArray, "detection_results", 10)
        self.detection_msg = Detection2DArray()

        self.result_img_pub = self.create_publisher(Image, "detection_image", 10)

        # 3. 创建图像订阅器（3D相机订阅深度信息，2D相机加载相机信息）
        image_topic = self.get_parameter('image_topic').value
        self.image_sub = self.create_subscription(
            Image, image_topic, self.process_image, 10)

        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.update_camera_info, 1)

        # 获取相机信息
        with open(self.get_parameter('camera_info_file').value) as f:
            self.camera_info = yaml.full_load(f.read())
            self.get_logger().info(f"默认相机参数: {self.camera_info['k']} \n {self.camera_info['d']}")

        # 4. 图像格式转换（使用cv_bridge）
        self.bridge = CvBridge()
        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

    def update_camera_info(self, msg: CameraInfo):
        """
        通过回调函数获取相机参数
        """
        self.camera_info['k'] = msg.k
        self.camera_info['p'] = msg.p
        self.camera_info['d'] = msg.d
        self.camera_info['r'] = msg.r
        self.camera_info['roi'] = msg.roi

        self.camera_info_sub.destroy()  # 只需要更新一次

    def process_image(self, msg: Image):
        # 5. 检测并发布结果
        image = self.bridge.imgmsg_to_cv2(msg)
        detection_result = self.yolov5.predict(image)
        self.get_logger().info(str(detection_result))

        self.detection_msg.detections.clear()
        self.detection_msg.header.frame_id = "camera"
        self.detection_msg.header.stamp = self.get_clock().now().to_msg()

        # 解析结果
        predictions = detection_result.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        for index in range(len(categories)):
            class_name = detection_result.names[int(categories[index])]
            detection_2d = Detection2D()
            detection_2d.id = class_name
            x1, y1, x2, y2 = boxes[index]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            if ros_distribution == 'galactic':
                detection_2d.bbox.center.x = center_x
                detection_2d.bbox.center.y = center_y
            else:
                detection_2d.bbox.center.position.x = center_x
                detection_2d.bbox.center.position.y = center_y

            detection_2d.bbox.size_x = float(x2 - x1)
            detection_2d.bbox.size_y = float(y2 - y1)

            obj_hypothesis = ObjectHypothesisWithPose()
            obj_hypothesis.hypothesis.class_id = class_name
            obj_hypothesis.hypothesis.score = float(scores[index])

            # 像素坐标转相机坐标
            world_x, world_y = px2xy(
                [center_x, center_y], self.camera_info["k"], self.camera_info["d"], 1)
            obj_hypothesis.pose.pose.position.x = world_x
            obj_hypothesis.pose.pose.position.y = world_y
            detection_2d.results.append(obj_hypothesis)
            self.detection_msg.detections.append(detection_2d)

            # 绘制结果
            if self.show_result or self.pub_result_img:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{class_name}({world_x:.2f},{world_y:.2f})", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.waitKey(1)

        # 显示结果（如果需要）
        if self.show_result:
            cv2.imshow('detection_result', image)
            cv2.waitKey(1)

        # 发布结果图像（如果需要）
        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            result_img_msg.header = msg.header
            self.result_img_pub.publish(result_img_msg)

        if len(categories) > 0:
            self.detection_pub.publish(self.detection_msg)

def main():
    rclpy.init()
    rclpy.spin(YoloDetectorNode())
    rclpy.shutdown()

if __name__ == "__main__":
    main()