# import rclpy
# from rclpy.node import Node
# from pc_segmentation_interfaces.srv import PCSegmentation
# from sensor_msgs.msg import PointCloud2, Image
# import cv2
# from cv_bridge import CvBridge
# import numpy as np
# from pypcd4 import PointCloud

# class ServiceTestNode(Node):
#     def __init__(self):
#         super().__init__('service_test_node')
#         self.client = self.create_client(PCSegmentation, 'segment_pointcloud')
#         self.bridge = CvBridge()

#     def load_and_publish_data(self, image_path, pointcloud_path):
#         # Load image
#         img = cv2.imread(image_path)
#         img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        
#         # Load pointcloud
#         pc = PointCloud.from_path(pointcloud_path)
#         pc_msg = pc.to_msg()

#         # Create request
#         request = PCSegmentation.Request()
#         request.rgb_image = img_msg
#         request.pc = pc_msg
#         request.labels = "cup: Cup, bowl: Bowl"  # Modify based on your objects
        
#         # Call service
#         future = self.client.call_sync(request)
#         return future

# def main():
#     rclpy.init()
#     node = ServiceTestNode()
    
#     # Replace with your paths
#     future = node.load_and_publish_data(
#         '/home/demo/ros2/data_rosbag2_2024_11_23-15_37_20/color_226_1732376243177459245.jpg',
#         '/home/demo/ros2/data_rosbag2_2024_11_23-15_37_20/points_224_1732376243176561318.pcd'
#     )
    
#     rclpy.spin_until_future_complete(node, future)
    
#     if future.result() is not None:
#         print('Got response:', future.result().segmented_clouds)
#     else:
#         print('Service call failed')

#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from pc_segmentation_interfaces.srv import PCSegmentation
from sensor_msgs.msg import PointCloud2, Image
import cv2
from cv_bridge import CvBridge
from pypcd4 import PointCloud

class ServiceTestNode(Node):
   def __init__(self):
       super().__init__('service_test_node')
       self.client = self.create_client(PCSegmentation, 'segment_pointcloud')
       self.bridge = CvBridge()

   def load_and_publish_data(self, image_path, pointcloud_path):
       img = cv2.imread(image_path)
       img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
       
       pc = PointCloud.from_path(pointcloud_path)
       pc_msg = pc.to_msg()

       request = PCSegmentation.Request()
       request.rgb_image = img_msg
       request.pc = pc_msg
       request.labels = "cup: Cup, ball: ball"
       
       try:
           response = self.client.call(request)
           print('Response:', response.segmented_clouds)
           return response
       except Exception as e:
           self.get_logger().error(f'Service call failed: {str(e)}')
           return None

def main():
   rclpy.init()
   node = ServiceTestNode()
   
   response = node.load_and_publish_data(
       '/home/demo/ros2/data_rosbag2_2024_11_23-15_37_20/color_226_1732376243177459245.jpg',
       '/home/demo/ros2/data_rosbag2_2024_11_23-15_37_20/points_224_1732376243176561318.pcd'
   )
   
   node.destroy_node()
   rclpy.shutdown()

if __name__ == '__main__':
   main()