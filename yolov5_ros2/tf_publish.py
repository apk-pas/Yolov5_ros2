import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
import tf2_ros
from geometry_msgs.msg import TransformStamped

class TfPublisher(Node):
    def __init__(self):
        super().__init__('tf_publish')
        
        self.declare_parameter("base_frame", "base_link")
        self.base_frame = self.get_parameter('base_frame').value
        
        # 动态TF发布器
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 订阅检测结果
        self.detection_sub = self.create_subscription(
            Detection2DArray, "yolo_result", self.detection_callback, 10)

    def publish_camera_tf(self,msg: Detection2DArray):
        """发布base_link到camera_frame的动态变换"""
        camera_frame = msg.header.frame_id
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = self.base_frame
        transform.child_frame_id = camera_frame
        
        # 相机相对base_link的固定位置
        self.x = 0.1
        self.y = 0.0
        self.z = 0.1
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.translation.z = self.z
        
        # 无旋转
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(transform)

    def detection_callback(self, msg: Detection2DArray):
        """接收检测结果并发布动态TF"""
        self.publish_camera_tf(msg)
        camera_frame = msg.header.frame_id
        for idx, detection in enumerate(msg.detections):
            transform = TransformStamped()
            transform.header.stamp = msg.header.stamp
            transform.header.frame_id = camera_frame
            transform.child_frame_id = f"{detection.id}_{idx}"

            # 保持原始坐标赋值逻辑
            transform.transform.translation.x = detection.results[0].pose.pose.position.x
            transform.transform.translation.y = detection.results[0].pose.pose.position.y
            transform.transform.translation.z = detection.results[0].pose.pose.position.z

            # 保持原始旋转赋值
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    node = TfPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()