from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pc_segmentation',
            executable='pcseg_service_node.py',
            name='pcseg_service_node',
            output='screen',
            parameters=[],
            arguments=[]
        )
    ])
