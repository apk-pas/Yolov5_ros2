import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import yaml
import os
import numpy as np
from yolov5 import YOLOv5

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        
        # 参数声明
        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="计算设备: cpu/cuda:0"))
        self.declare_parameter("image_topic", "/image_raw", ParameterDescriptor(
            name="image_topic", description="图像话题"))
        self.declare_parameter("camera_info_topic", "/camera/camera_info", ParameterDescriptor(
            name="camera_info_topic", description="相机信息话题"))
        self.declare_parameter("camera_info_file", 
            f"{get_package_share_directory('yolov5_ros2')}/config/camera_info.yaml",
            ParameterDescriptor(name="camera_info_file", description="相机参数文件路径"))
        self.declare_parameter("show_result", False)
        self.declare_parameter("pub_result_img", False)

        # 加载模型
        model_path = os.path.join(
            get_package_share_directory('yolov5_ros2'), "config", "best.pt")
        self.yolov5 = YOLOv5(model_path, self.get_parameter('device').value)

        # 相机参数
        self.camera_info = self._load_camera_info()
        self.camera_frame = "camera"

        # 发布器
        self.detection_pub = self.create_publisher(Detection2DArray, "detections", 10)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        # 订阅器
        self.image_sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, 
            self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.camera_info_callback, 1)

        # 工具
        self.bridge = CvBridge()
        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value
        self.ros_distribution = os.environ.get("ROS_DISTRO", "humble")

    def _load_camera_info(self):
        """从文件加载默认相机参数"""
        with open(self.get_parameter('camera_info_file').value, 'r') as f:
            return yaml.full_load(f.read())

    def camera_info_callback(self, msg: CameraInfo):
        """从话题更新相机参数"""
        self.camera_info.update({
            'k': msg.k,
            'd': msg.d,
            'r': msg.r,
            'p': msg.p
        })
        self.get_logger().info("已从话题更新相机参数")
        self.camera_info_sub.destroy()  # 只需要一次更新

    def px2xy(self, point, camera_k, camera_d, z=1.0):
        """像素坐标转相机坐标（内置方法，替代cv_tool）"""
        MK = np.array(camera_k, dtype=float).reshape(3, 3)
        MD = np.array(camera_d, dtype=float)
        point = np.array(point, dtype=float).reshape(1, 1, 2)
        undistorted = cv2.undistortPoints(point, MK, MD, P=MK)
        return (undistorted[0][0] * z).tolist()

    def image_callback(self, msg: Image):
        """处理图像并发布检测结果"""
        try:
            image = self.bridge.imgmsg_to_cv2(msg)
            detect_result = self.yolov5.predict(image)
            detection_array = Detection2DArray()
            detection_array.header = msg.header
            detection_array.header.frame_id = self.camera_frame

            # 解析检测结果
            predictions = detect_result.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]

            for idx in range(len(categories)):
                cls_name = detect_result.names[int(categories[idx])]
                x1, y1, x2, y2 = map(int, boxes[idx])
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                # 转换为相机坐标系
                world_x, world_y = self.px2xy(
                    [center_x, center_y], 
                    self.camera_info["k"], 
                    self.camera_info["d"], 
                    z=1.0  # 假设深度为1米
                )

                # 创建检测消息
                detection = Detection2D()
                detection.id = f"{cls_name}_{idx}"
                if self.ros_distribution == 'galactic':
                    detection.bbox.center.x = center_x
                    detection.bbox.center.y = center_y
                else:
                    detection.bbox.center.position.x = center_x
                    detection.bbox.center.position.y = center_y
                detection.bbox.size_x = x2 - x1
                detection.bbox.size_y = y2 - y1

                # 添加位姿假设
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = cls_name
                hypothesis.hypothesis.score = float(scores[idx])
                hypothesis.pose.pose.position.x = world_x
                hypothesis.pose.pose.position.y = world_y
                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

                # 绘制检测框
                if self.show_result or self.pub_result_img:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{cls_name}: {scores[idx]:.2f}", 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 发布结果
            self.detection_pub.publish(detection_array)

            # 显示/发布结果图像
            if self.show_result:
                cv2.imshow("Detection Result", image)
                cv2.waitKey(1)
            if self.pub_result_img:
                self.result_img_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

        except Exception as e:
            self.get_logger().error(f"图像处理错误: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()