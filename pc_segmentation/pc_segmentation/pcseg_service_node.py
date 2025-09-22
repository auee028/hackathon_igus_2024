import os
import argparse
import rclpy
from rclpy.node import Node
import rclpy.utilities
from sensor_msgs.msg import PointCloud2, Image
from pc_segmentation_interfaces.srv import PCSegmentation
from pypcd4 import PointCloud
import message_filters
from cv_bridge import CvBridge
import numpy as np
import cv2
from pc_segmentation.detect_and_segment import PCSegmentationModel

class SegmentationServiceNode(Node):
    def __init__(self, save_dir='tmp/', box_threshold=0.35, text_threshold=0.25, to_save=True):
        super().__init__('pcseg_service_node')

        # Create a CvBridge to convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Create message filters to synchronize the PointCloud2 and Image topics
        self.rgb_image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_rect_raw')
        self.pointcloud_sub = message_filters.Subscriber(self, PointCloud2, '/camera/depth/color/points')
        # self.rgb_image_sub = self.create_subscription(Image, self.image_path_callback, 10)
        # self.pointcloud_sub = self.create_subscription(PointCloud2, self.pointcloud_path_callback, 10)

        # Create a time synchronizer to synchronize the point cloud and RGB image
        self.ts = message_filters.TimeSynchronizer([self.rgb_image_sub, self.pointcloud_sub], 10)
        self.ts.registerCallback(self.image_pointcloud_callback)

        # Service server
        self.srv = self.create_service(PCSegmentation, 'segment_pointcloud', self.handle_segmentation_request)

        # Store the latest synchronized image and point cloud
        self.latest_rgb_image = None
        self.latest_pointcloud = None

        self.get_logger().info("Segmentation service node started. Waiting for synchronized data...")

        # Make a directory to save the intermediates
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.to_save = to_save
    
    def image_path_callback(self, msg):
        self.latest_image_path = msg.data
        # Load image if needed
        if os.path.exists(self.latest_image_path):
            self.latest_rgb_image = cv2.imread(self.latest_image_path)

    def pointcloud_path_callback(self, msg):
        self.latest_pointcloud_path = msg.data
        # Load pointcloud if needed
        if os.path.exists(self.latest_pointcloud_path):
            self.latest_pointcloud = PointCloud.from_path(self.latest_pointcloud_path)

    def image_pointcloud_callback(self, rgb_image_msg, pointcloud_msg):
        """
        Callback to receive and synchronize RGB image and point cloud data.
        """
        self.latest_rgb_image = rgb_image_msg
        self.latest_pointcloud = pointcloud_msg
        # self.get_logger().info("Received synchronized RGB image and point cloud.")

    def compute_mean_x_y(self, segmented_clouds):
        x_mean = []
        y_mean = []
        for cloud in segmented_clouds:
            cloud_3d = cloud.numpy(("x", "y", "z"))
            x_mean.append(float(np.mean(cloud_3d[:,0])))
            y_mean.append(float(np.mean(cloud_3d[:,1])))
            
        print("computed mean : ", x_mean, y_mean)
        return x_mean, y_mean
            
    def handle_segmentation_request(self, request, response):
        """
        Handle segmentation requests by performing segmentation on a synchronized point cloud and RGB image and a string of caption-label pairs.
        """
        self.get_logger().info(f"Received request.caption_labels: {request.caption_labels}")

        # Get the input text captions of interested object classes
        caption_labels = request.caption_labels  # Input string must be as "caption1: label1, caption2: label2, ..."
        # caption_labels = 'plane white mug: mug, grey piece with a checker board: piece'

        # Get the latest synchronized point cloud and image
        pc_msg, rgb_msg = self.get_latest_data()

        if pc_msg is None or rgb_msg is None:
            self.get_logger().error("No synchronized data available for segmentation.")
            return response

        # Perform segmentation
        print('1  ', caption_labels)
        object_ids, object_labels, segmented_clouds = self.perform_segmentation(rgb_msg, pc_msg, caption_labels)
        x_mean, y_mean = self.compute_mean_x_y(segmented_clouds)
        
        segmented_clouds = [pc.to_msg(pc_msg.header) for pc in segmented_clouds]
        
        # Populate the response with the segmented clouds
        
        response.object_ids = object_ids
        response.object_labels = object_labels
        response.segmented_clouds = segmented_clouds
        response.mean_x = x_mean
        response.mean_y = y_mean
        
        return response

    def get_latest_data(self):
        """
        Get the latest synchronized point cloud and RGB image.
        """
        if self.latest_pointcloud is None or self.latest_rgb_image is None:
            self.get_logger().warning("No synchronized data received yet.")
        return self.latest_pointcloud, self.latest_rgb_image
    
    def convert_msg_to_pc(self, pc_msg):
        # Convert the input PointCloud2 message to a usable pypcd4.PointCloud format
        pc = PointCloud.from_msg(pc_msg)
        # print(pc.numpy().shape)

        return pc
    
    def convert_msg_to_rgb(self, rgb_msg):
        # Convert the input Image message to a usable OpenCV format (BGR)
        img_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        # print(img_rgb.shape)
        
        # img_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        # # Convert from BGR to RGB
        # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(self.save_dir, 'image.jpg'), img_rgb)

        return img_rgb

    def perform_segmentation(self, rgb_msg, pc_msg, caption_labels):
        """
        Perform segmentation on the given point cloud.

        Args:
            rgb_msg (Image): The raw (color) image message.
            pc_msg (PointCloud2): The raw point cloud message.
            caption_labels (string): The caption-label pairs e.g. 'ball: ball, cup: cup, jelly box: jelly_box'

        Returns:
            list: A list of segmented PointCloud2 messages.
        """
        # Convert the input prompt-label string to Caption Ontology for the GroundedSAM model
        
        print('2  ', caption_labels)
        print('3  ', caption_labels.split(','))
        caption_ontology = {caption.strip(): label.strip() for s in caption_labels.split(',') for (caption, label) in [s.split(':')]}

        # Convert the input ROS2 data to usable formats
        rgb = self.convert_msg_to_rgb(rgb_msg)
        pc = self.convert_msg_to_pc(pc_msg)

        # Save the rgb image to feed its path into the GroundedSAM model
        rgb_path = os.path.join(self.save_dir, 'image.jpg')
        cv2.imwrite(rgb_path, rgb)

        # Perform point cloud segmentation using GroundedSAM
        model = PCSegmentationModel(caption_ontology, 
                                    box_threshold=self.box_threshold, 
                                    text_threshold=self.text_threshold, 
                                    to_save=self.to_save,
                                    save_dir=self.save_dir)
        
        detection_result = model.detect(rgb_path)
        # print(detection_result)
        if not detection_result:
            print("*** Nothing detected. ***")
            return []
            
        # res = model.segment(pc)
        # segmented_clouds = [pc.to_msg(pc_msg.header) for pc in res]
        
        object_ids, object_labels, segmented_clouds = model.segment(pc)

        return object_ids, object_labels, segmented_clouds


def main(args=None):
    rclpy.init(args=args)
    
    # node = SegmentationServiceNode(box_threshold=0.7, text_threshold=0.7)
    node = SegmentationServiceNode(box_threshold=0.5, text_threshold=0.4)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
