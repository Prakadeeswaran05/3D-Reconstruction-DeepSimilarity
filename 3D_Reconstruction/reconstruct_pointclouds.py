import os
import numpy as np
import re
import cv2
import open3d as o3d
import argparse
from pathlib import Path
from pixels_to_pointcloud import InverseProjection

def process_folders(left_image_folder: str,disparity_folder: str,output_folder: str,
    camera_params_file: str,disp_thresh: float = 1.0,
    downsample: bool = False,downsample_size: int = 200,camera_id: str = "LEFT_CAM_2K",viz:bool=False):
    """
    Create point clouds from stereo images and disparity maps.
    
    Args:
        left_image_folder: Left images directory
        disparity_folder: Disparity maps directory
        output_folder: Output directory for point clouds
        camera_params_file: Camera calibration file
        disp_thresh: Min disparity threshold
        depth_thresh: Max depth threshold
        downsample: Enable downsampling
        downsample_size: Voxel size for downsampling
        camera_id: Camera ID in config file
    """
    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    inverse_proj = InverseProjection(camera_params_file,camera_id)
    left_image_files = sorted([f for f in os.listdir(left_image_folder) if f.endswith(('.jpeg', '.jpg', '.png'))])
    disparity_files = sorted([f for f in os.listdir(disparity_folder) if f.endswith('.npy')])
    
    if not left_image_files:
        raise ValueError("No images found in the left_stereo folder.")
        
    
    if not disparity_files:
        raise ValueError(f"No disparity files found in disparity_folder")
        
    
    if len(left_image_files) != len(disparity_files):
        raise ValueError("Mismatch in left image and disparity pairs.")
    
    num_pairs = len(left_image_files)
    print(f"Processing {num_pairs} image-disparity pairs")
    
    for i in range(num_pairs):
        try:
            left_image_file = left_image_files[i]
            disp_file = disparity_files[i]
    
            left_image_base = os.path.splitext(left_image_file)[0]
            left_image_base = left_image_base.replace('left_', '')
            
            print(f"Processing pair {i+1}/{num_pairs}: {left_image_file} + {disp_file}")
            
            # Full paths
            image_path = os.path.join(left_image_folder, left_image_file)
            disparity_path = os.path.join(disparity_folder, disp_file)
            output_path = os.path.join(output_folder, f"{left_image_base}_pointcloud.pcd")

            point_cloud = inverse_proj.reconstruct_colored_point_cloud(
                disparity_path,
                image_path,
                disp_thresh,
                downsample,
                downsample_size
            )
            if i==0 and viz:
                inverse_proj.visualize_point_cloud(point_cloud)
            
            
            o3d.io.write_point_cloud(output_path, point_cloud)
            print(f"Saved point cloud to {output_path}")  
        except Exception as e:
            print(f"Error processing {left_image_file}: {str(e)}")
    
    print("Point cloud generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate point clouds from stereo images')
    # Required arguments
    parser.add_argument('--left', type=str,required=True, help='Left images folder')
    parser.add_argument('--disp', type=str,required=True, help='Disparity maps folder')
    parser.add_argument('--out', type=str,required=True, help='Output folder for point clouds')
    parser.add_argument('--cfg', type=str, required=True, help='Path to camera config file')
    parser.add_argument('--downsample', type=int, required=True,
                    choices=[0, 1], default=0,
                    help='Enable downsampling (1=true, 0=false)')
    parser.add_argument('--viz', type=int, required=True,
                    choices=[0, 1], default=0,
                    help='Visualize pointcloud (1=true, 0=false)')
    parser.add_argument('--voxel_size', type=float, default=200, help='Voxel size (default: 200)')
    # Optional arguments
    parser.add_argument('--disp_thresh', type=float, default=1.0, help='Min disparity (default: 1.0)')
    parser.add_argument('--cam_id', default="LEFT_CAM_FHD", help='Camera ID (default: LEFT_CAM_FHD)')
    
    args = parser.parse_args() 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    print(project_root)

   
    for attr in ['left', 'disp', 'out', 'cfg']:
        setattr(args, attr, os.path.join(project_root, getattr(args, attr)))

    #print(args.left)
    
    process_folders(
        args.left, 
        args.disp, 
        args.out, 
        args.cfg,
        disp_thresh=args.disp_thresh,
        downsample=bool(args.downsample),
        downsample_size=args.voxel_size,
        camera_id=args.cam_id,
        viz=bool(args.viz)
    )