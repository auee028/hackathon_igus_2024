import os
import glob
import numpy as np
import cv2
from pypcd4 import PointCloud
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D  # Required for 'projection=3d'


def remove_outlier_indices(points, n_neighbors=20, radius = 0.05):
    # Use NearestNeighbors to compute distances between points
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Compute the mean distance of each point to its neighbors
    mean_distances = np.mean(distances[:, 1:], axis=1)  # Skip the first distance (distance to itself)

    # Calculate a threshold to classify points as outliers
    threshold = np.percentile(mean_distances, 90)  # For example, the 90th percentile as a threshold

    # Identify the outliers: points with distance greater than threshold
    inlier_indices = mean_distances < threshold
    outlier_indices = ~inlier_indices

    return outlier_indices

def show_pcd_with_open3d(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)

    # o3d.visualization.draw_geometries([downpcd], window_name='Segmented Point Clouds')
    o3d.visualization.draw_geometries([pcd], window_name='Segmented Point Clouds')

def show_point_cloud(pcd_file, verbose=False, to_save=True):
    # Load the PCD file
    pcd = PointCloud.from_path(pcd_file)

    # Extract points
    points = pcd[("x", "y", "z")].numpy()  # Extract 3D points

    # Extract colors
    rgb_data = pcd[("rgb")].numpy()  # RGB data is stored as packed float

    # Convert the packed RGB data (float32) to R, G, B channels
    colors = PointCloud.decode_rgb(rgb_data)

    if verbose:
        print("Points shape:", points.shape)  # Should be (N, 3)
        print("Colors shape:", colors.shape)  # Should be (N, 3)
        print("Sample Colors:", colors[:10])  # Check the first 10 decoded colors

    # Remove outliers
    outlier_indices = remove_outlier_indices(points)
    inlier_indices = ~outlier_indices

    points = points[inlier_indices]
    colors = colors[inlier_indices]

    if verbose:
        print("Points without outliers:", points.shape)  # Should be (N', 3)
        print("Colors without outliers:", colors.shape)  # Should be (N', 3)

    # Set up the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points in 3D, using the color information
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors/255, marker='o', s=1)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud with Original Colors')

    # Display the plot
    if to_save:
        save_file_name = os.path.basename(pcd_file).replace('pcseg', 'vis').replace('.pcd', '.jpg')
        save_path = os.path.join(os.path.dirname(pcd_file), save_file_name)
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Load the PCD file
    pcd_root = '/home/demo/ros2/object_placement/pointcloud_segmentation_ws/tmp/'

    # show_pcd_with_open3d(pcd_path)
    # show_point_cloud(pcd_path)

    for pcd_path in glob.glob(os.path.join(pcd_root, 'pcseg_*.pcd')):
        print(pcd_path)
        # show_point_cloud(pcd_path)
        # show_pcd_with_open3d(pcd_path)
        show_point_cloud(pcd_path, verbose=True, to_save=True)
    