import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
import tf2_ros
from geometry_msgs.msg import TransformStamped

class TfPublisher(Node):
    def __init__(self):
        super().__init__('tf_publisher')
        
        # 参数
        self.declare_parameter("base_frame", "base_link", 
            description="基础坐标系")
        self.base_frame = self.get_parameter('base_frame').value

        # TF广播器
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 订阅检测结果
        self.detection_sub = self.create_subscription(
            Detection2DArray, "detections", self.detection_callback, 10)

        self.get_logger().info(f"TF发布器已启动，基础坐标系: {self.base_frame}")

    def detection_callback(self, msg: Detection2DArray):
        """将检测结果转换为TF发布"""
        camera_frame = msg.header.frame_id
        for detection in msg.detections:
            # 创建TF变换
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = self.base_frame
            transform.child_frame_id = detection.id  # 格式: 类别_索引

            # 从检测结果获取坐标
            transform.transform.translation.x = detection.results[0].pose.pose.position.x
            transform.transform.translation.y = detection.results[0].pose.pose.position.y
            transform.transform.translation.z = 0.0  # 简化为2D场景

            # 无旋转
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0

            # 发布TF
            self.tf_broadcaster.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    node = TfPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()