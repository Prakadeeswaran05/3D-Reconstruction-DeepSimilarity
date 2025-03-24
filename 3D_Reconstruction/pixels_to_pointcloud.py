import numpy as np
import cv2
import re
import open3d as o3d
from typing import Dict, Any, Union, Optional

class InverseProjection:    
    def __init__(self, camera_param_file: str, camera_id: str) -> None:
        """Initialize with camera intrinsic parameters.
        
        Args:
            camera_param_file: Path to camera param
        """
        try:
            self.camera_params = self.load_camera_params(camera_param_file, camera_id)
            print("Loaded camera parameters:")
            for key, value in self.camera_params.items():  
                print(f"  {key} = {value}")
        
            self.fx = self.camera_params['fx']
            self.fy = self.camera_params['fy']
            self.cx = self.camera_params['cx']
            self.cy = self.camera_params['cy']
            self.baseline = self.camera_params['Baseline']

        except Exception as e:
            print(f"Failed to load camera parameters: {e}")
    
    def reconstruct_colored_point_cloud(self, disparity_file: str, color_image_file: str, 
                                      disp_thresh: float, 
                                      downsample: bool = False, downsample_size: int = 200) -> o3d.geometry.PointCloud:
        """Reconstruct 3D point cloud with color.
        
        Args:
            disparity_file: Path to NPY disparity map
            color_image_file: Path to color image   
            
        Returns:
            Point cloud as [X, Y, Z, R, G, B] coordinates
        """
        
        disparity_map = np.load(disparity_file)
        height, width = disparity_map.shape
        color_image = cv2.imread(color_image_file)
        if color_image is None:
            raise FileNotFoundError(f"Could not load color image: {color_image_file}")

        filtered_mask = disparity_map > disp_thresh #  to prevent having inf depth since disp closer to zero can produce inf depth
        filtered_points = np.where(filtered_mask)

        depth_filtered = (self.baseline * self.fx) / disparity_map[filtered_mask]  #horizontal stereo setup so fx 

        x_coords_filtered = filtered_points[1]
        y_coords_filtered = filtered_points[0]
        x_cam_filtered = (x_coords_filtered - self.cx) / self.fx
        y_cam_filtered = (y_coords_filtered - self.cy) / self.fy

        num_filtered = len(depth_filtered)
        points = np.zeros((num_filtered, 3))
        colors = np.zeros((num_filtered, 3))

        points[:, 0] = x_cam_filtered * depth_filtered
        points[:, 1] = y_cam_filtered * depth_filtered
        points[:, 2] = depth_filtered

        colors[:, 0] = color_image[y_coords_filtered, x_coords_filtered, 2] / 255.0 #B
        colors[:, 1] = color_image[y_coords_filtered, x_coords_filtered, 1] / 255.0 #G
        colors[:, 2] = color_image[y_coords_filtered, x_coords_filtered, 0] / 255.0 #R

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        
        if downsample:
            print('Downsampling pointcloud')
            voxel_size = downsample_size
            downsampled_pcd = pcd.voxel_down_sample(voxel_size)
            return downsampled_pcd
        else:
            return pcd
        
    def visualize_point_cloud(self, pcd: o3d.geometry.PointCloud) -> None:
        """Visualize point cloud using Open3D.
        
        Args:
            point_cloud: Point cloud array [X, Y, Z, R, G, B]
        """
        o3d.visualization.draw_geometries([pcd],
                                zoom=0.8412,
                                front=[0.5, -0.2, -0.8],
                                lookat=[4.0, 4.0, 1.2],
                                up=[-0.07, -0.95, 0.24])
    
    def load_camera_params(self, file_path: str, camera_id: str = "LEFT_CAM_2K") -> Dict[str, float]:
        """Load camera parameters from a text file.
        
        Args:
            file_path: Path to configuration file
            camera_id: Camera identifier to find
            
        Returns:
            Dictionary of camera parameters
        """
        params = {}
        current_section = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if "[" in line and "]" in line:
                    section_match = re.search(r'\[(.*?)\]', line)
                    if section_match:
                        current_section = section_match.group(1)
                
                if current_section == camera_id or current_section == "STEREO":
                    matches = re.findall(r'(\w+)=(-?\d*\.?\d+)', line)
                    for key, value in matches:
                        params[key] = float(value)
        
        return params


if __name__ == "__main__":
    # Load camera parameters
    inverse_proj = InverseProjection("/home/harish/noteworthy_project/data/stereo_config.txt","LEFT_CAM_FHD")
    
    colored_point_cloud = inverse_proj.reconstruct_colored_point_cloud(
        "/home/harish/noteworthy_project/data/disparity/1694977813_711027992_disparity.npy",
        "/home/harish/noteworthy_project/data/left_stereo/left_1694977813_711027992.jpeg",
        2.0,  # disp_thresh
        False,  # downsample
        200   # downsample_size
    )
    
    inverse_proj.visualize_point_cloud(colored_point_cloud)