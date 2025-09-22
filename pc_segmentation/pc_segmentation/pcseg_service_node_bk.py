# import os
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2, Image
# from pc_segmentation_interfaces.srv import PCSegmentation
# from pypcd4 import PointCloud
# import message_filters
# from cv_bridge import CvBridge
# import numpy as np
# import cv2
# from pc_segmentation.detect_and_segment import PCSegmentationModel

# class SegmentationServiceNode(Node):
#     def __init__(self, save_dir='tmp/', box_threshold=0.5, to_save=True):
#         super().__init__('pcseg_service_node')

#         # Create a CvBridge to convert ROS Image messages to OpenCV images
#         self.bridge = CvBridge()

#         # Create message filters to synchronize the PointCloud2 and Image topics
#         self.rgb_image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_rect_raw')
#         self.pointcloud_sub = message_filters.Subscriber(self, PointCloud2, '/camera/depth/color/points')
#         # self.rgb_image_sub = self.create_subscription(Image, self.image_path_callback, 10)
#         # self.pointcloud_sub = self.create_subscription(PointCloud2, self.pointcloud_path_callback, 10)

#         # Create a time synchronizer to synchronize the point cloud and RGB image
#         self.ts = message_filters.TimeSynchronizer([self.rgb_image_sub, self.pointcloud_sub], 10)
#         self.ts.registerCallback(self.image_pointcloud_callback)

#         # Service server
#         self.srv = self.create_service(PCSegmentation, 'segment_pointcloud', self.handle_segmentation_request)

#         # Store the latest synchronized image and point cloud
#         self.latest_rgb_image = None
#         self.latest_pointcloud = None

#         self.get_logger().info("Segmentation service node started. Waiting for synchronized data...")

#         # Make a directory to save the intermediates
#         self.save_dir = save_dir
#         if not os.path.exists(self.save_dir):
#             os.mkdir(self.save_dir)
        
#         self.box_threshold = box_threshold
#         self.to_save = to_save
        
#         self.img_dir = save_dir
    
#     def image_path_callback(self, msg):
#         self.latest_image_path = msg.data
#         # Load image if needed
#         if os.path.exists(self.latest_image_path):
#             self.latest_rgb_image = cv2.imread(self.latest_image_path)

#     def pointcloud_path_callback(self, msg):
#         self.latest_pointcloud_path = msg.data
#         # Load pointcloud if needed
#         if os.path.exists(self.latest_pointcloud_path):
#             self.latest_pointcloud = PointCloud.from_path(self.latest_pointcloud_path)

#     def image_pointcloud_callback(self, rgb_image_msg, pointcloud_msg):
#         """
#         Callback to receive and synchronize RGB image and point cloud data.
#         """
#         self.latest_rgb_image = rgb_image_msg
#         self.latest_pointcloud = pointcloud_msg
#         self.get_logger().info("Received synchronized RGB image and point cloud.")

#     def handle_segmentation_request(self, request, response):
#         """
#         Handle segmentation requests by performing segmentation on a synchronized point cloud and RGB image.
#         """
#         self.get_logger().info(f"Received request.labels: {request.labels}")

#         # Get the input text captions of interested object classes
#         captions_label = request.labels  # Input string must be as "caption1: label1, caption2: label2, ..."

#         # Get the latest synchronized point cloud and image
#         pc_msg, rgb_msg = self.get_latest_data()

#         if pc_msg is None or rgb_msg is None:
#             self.get_logger().error("No synchronized data available for segmentation.")
#             return response

#         # Perform segmentation
#         segmented_clouds = self.perform_segmentation(rgb_msg, pc_msg, captions_label)

#         # Populate the response with the segmented clouds
#         response.segmented_clouds = segmented_clouds
        
#         print(response)
        
#         return response

#     def get_latest_data(self):
#         """
#         Get the latest synchronized point cloud and RGB image.
#         """
#         if self.latest_pointcloud is None or self.latest_rgb_image is None:
#             self.get_logger().warning("No synchronized data received yet.")
#         return self.latest_pointcloud, self.latest_rgb_image
    
#     def convert_msg_to_pc(self, pc_msg):
#         # Convert the input PointCloud2 message to a usable pypcd4.PointCloud format
#         pc = PointCloud.from_msg(pc_msg)

#         return pc
    
#     def convert_msg_to_rgb(self, rgb_msg):
#         # Convert the input Image message to a usable OpenCV format (BGR)
#         img_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        
#         # img_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

#         # # Convert from BGR to RGB
#         # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#         cv2.imwrite(os.path.join(self.img_dir, 'image.jpg'))
#         print('saved temp img: ', os.path.join(self.img_dir, 'image.jpg'))

#         return img_rgb

#     def perform_segmentation(self, rgb_msg, pc_msg, captions_label):
#         """
#         Perform segmentation on the given point cloud.

#         Args:
#             rgb_msg (Image): The raw (color) image message.
#             pc_msg (PointCloud2): The raw point cloud message.
#             captions_label (string): The caption-label pairs e.g. 'ball: Ball, cup: Cup, jelly box: Box'

#         Returns:
#             list: A list of segmented PointCloud2 messages.
#         """
#         # Convert the input captions-label string to Caption Ontology for the GroundedSAM model
#         caption_ontology = {caption.strip(): label.strip() for s in captions_label.split(',') for (label, caption) in [s.split(':')]}

#         # Convert the input ROS2 data to usable formats
#         rgb = self.convert_msg_to_rgb(rgb_msg)
#         pc = self.convert_msg_to_pc(pc_msg)

#         # Save the rgb image to feed its path into the GroundedSAM model
#         rgb_path = os.path.join(self.save_dir, 'image.jpg')
#         cv2.write(rgb_path, rgb)

#         # Perform point cloud segmentation using GroundedSAM
#         model = PCSegmentationModel(caption_ontology, box_threshold=self.box_threshold, to_save=self.to_save)
#         res = model.detect(rgb_path).segment(pc)
#         segmented_clouds = [pc.to_msg(pc_msg.header) for pc in res]

#         return segmented_clouds


# def main(args=None):
#     rclpy.init(args=args)
#     node = SegmentationServiceNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
import os
import rclpy
from rclpy.node import Node
from pc_segmentation_interfaces.srv import PCSegmentation
from cv_bridge import CvBridge
import cv2
from pc_segmentation.detect_and_segment import PCSegmentationModel
from pypcd4 import PointCloud

class SegmentationServiceNode(Node):
    def __init__(self, save_dir='tmp/'):
        super().__init__('pcseg_service_node')
        self.bridge = CvBridge()
        self.srv = self.create_service(PCSegmentation, 'segment_pointcloud', self.handle_segmentation_request)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    # def handle_segmentation_request(self, request, response):
    #     self.get_logger().info(f"Received request.labels: {request.labels}")
    #     try:
    #         rgb_msg = request.rgb_image
    #         pc_msg = request.pc
    #         segmented_clouds = self.perform_segmentation(rgb_msg, pc_msg, request.labels)
    #         response.segmented_clouds = segmented_clouds
    #         print("Segmentation completed:", response)
            
    #         for i, pc in enumerate(response.segmented_clouds):
                
    #             pc.save(f'segment_{i}.pcd')
    #     except Exception as e:
    #         self.get_logger().error(f"Segmentation failed: {str(e)}")
    #     return response
    
    def handle_segmentation_request(self, request, response):
        try:
            rgb_msg = request.rgb_image
            pc_msg = request.pc
            segmented_clouds = self.perform_segmentation(rgb_msg, pc_msg, request.labels)
            response.segmented_clouds = segmented_clouds

            # Convert and save each segmented cloud
            for i, pc_msg in enumerate(segmented_clouds):
                pc = PointCloud.from_msg(pc_msg)
                pc.save(f'segment_{i}.pcd')
                
        except Exception as e:
            self.get_logger().error(f"Segmentation failed: {str(e)}")
        return response

    # def perform_segmentation(self, rgb_msg, pc_msg, captions_label):
    #     rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
    #     pc = PointCloud.from_msg(pc_msg)
    #     rgb_path = os.path.join(self.save_dir, 'image.jpg')
    #     cv2.imwrite(rgb_path, rgb)
        
    #     caption_ontology = {caption.strip(): label.strip() 
    #                       for s in captions_label.split(',') 
    #                       for (label, caption) in [s.split(':')]}
        
    #     model = PCSegmentationModel(caption_ontology, box_threshold=0.5)
    #     res = model.detect(rgb_path).segment(pc)
    #     return [pc.to_msg(pc_msg.header) for pc in res]
    
    def perform_segmentation(self, rgb_msg, pc_msg, captions_label):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        pc = PointCloud.from_msg(pc_msg)
        rgb_path = os.path.join(self.save_dir, 'image.jpg')
        cv2.imwrite(rgb_path, rgb)
        
        caption_ontology = {caption.strip(): label.strip() 
                        for s in captions_label.split(',') 
                        for (label, caption) in [s.split(':')]}
        
        print("Caption ontology:", caption_ontology)
        model = PCSegmentationModel(caption_ontology, box_threshold=0.5)
        
        # Add debug prints
        result = model.detect(rgb_path)
        print("Detection result1:", result)
        print("Detection result2:", [mask.shape for mask in result["masks"]])
        print(pc)
        # if result == None:
        #     return []
        
        res = model.segment(pc)
        return [pc.to_msg(pc_msg.header) for pc in res]

def main():
    rclpy.init()
    node = SegmentationServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()