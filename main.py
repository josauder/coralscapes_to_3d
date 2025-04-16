import numpy as np
import os
from pathlib import Path
from colmap_utils import read_cameras_binary, read_images_binary, read_array
from coralscapes_utils import segment, class_to_id, colors
from PIL import Image
from scipy.spatial import cKDTree
import pandas as pd
import open3d as o3d
import argparse

parser = argparse.ArgumentParser(description="Run GLOMAP on a set of images and project Coralscapes segmentations onto the dense point cloud!")
parser.add_argument("--images", type=str, help="Path to image directory")
parser.add_argument("--workspace", type=str, help="Path to workspace directory")


def find_end_files(directory,ending='JPG'):
    end_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ending):
                end_files.append(os.path.join(root, file))
    return end_files




def depth_to_world(depth_map, intrinsics, extrinsics, model):
    """
    Projects a depth map into world coordinates.
    
    :param depth_map: 2D numpy array representing depth values
    :param intrinsics: Intrinsic matrix (3x3)
    :param extrinsics: Extrinsic matrix (4x4)
    :param model: COLMAP camera model type
    :return: Nx3 array of 3D world points
    """
    h, w = depth_map.shape
    
    # Generate pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1).reshape(-1, 3).T
    
    # Convert to normalized camera coordinates
    if model in ["PINHOLE", "SIMPLE_PINHOLE"]:
        inv_intrinsics = np.linalg.inv(intrinsics)
        cam_coords = inv_intrinsics @ pixels * depth_map.flatten()
    elif model in ["RADIAL", "SIMPLE_RADIAL"]:
        f, cx, cy, k1 = intrinsics[:4]  # Extract focal length, principal point, and distortion param
        x_n = (pixels[0] - cx) / f
        y_n = (pixels[1] - cy) / f
        r2 = x_n**2 + y_n**2
        radial_distortion = 1 + k1 * r2
        x_n *= radial_distortion
        y_n *= radial_distortion
        cam_coords = np.vstack((x_n * depth_map.flatten(), y_n * depth_map.flatten(), depth_map.flatten()))
    else:
        raise NotImplementedError(f"Camera model {model} not supported yet.")
    
    # Convert to world coordinates
    cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
    world_coords = (extrinsics @ cam_coords)[:3].T
    
    return world_coords


def backproject(depth, K, R, t):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    pixels_hom = np.stack((i, j, np.ones_like(i)), axis=-1)  # (H, W, 3)
    pixels_cam = np.linalg.inv(K) @ pixels_hom.reshape(-1, 3).T  # (3, H*W)
    points_cam = pixels_cam * depth.reshape(-1)  # scale rays by depth
    points_world = R.T @ (points_cam - t.reshape(3, 1))  # World = R^T*(X - t)
    return points_world.T  # (N, 3)



def main(base_path):
    depth_map_dir = os.path.join(base_path, "stereo/depth_maps/")
    depth_map_paths = sorted(find_end_files(depth_map_dir, ending='geometric.bin'))
    image_dir = os.path.join(base_path,"images")


    # Path to COLMAP output directory
    model_dir = Path(os.path.join(base_path, "sparse"))

    # Read cameras and images
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = read_images_binary(model_dir / "images.bin")


    intrinsic_params = list(cameras.values())[0].params

    intrinsics = np.array([
        [intrinsic_params[0], 0, intrinsic_params[2]],
        [0, intrinsic_params[1], intrinsic_params[3]],
        [0, 0, 1]
    ])

    point_cloud_xyz = []
    point_cloud_rgb = []
    point_cloud_d2cam = []
    point_cloud_seg = []

    from tqdm import tqdm
    for img_id in tqdm(sorted(list(images.keys()))):
        img = images[img_id]

        R = img.qvec2rotmat()
        t = img.tvec
        
        depth_map = read_array(os.path.join(depth_map_dir, img.name+".geometric.bin"))    
        
        pil_image = Image.open(os.path.join(image_dir, img.name))
        seg = segment(pil_image).reshape(-1)

        rgb = np.array(pil_image)[:,:,:3].reshape(-1, 3).astype(np.uint8)
        z = depth_map.reshape(-1)    
        mask = np.logical_and(z>0, ~np.logical_or(np.logical_or(seg == class_to_id['fish'],seg == class_to_id['background']), seg == class_to_id['human']))
        pointmap = backproject(depth_map, intrinsics, R, t)
        

        if img_id == 1:
            point_cloud_xyz = pointmap[mask].astype(np.float32)
            point_cloud_rgb = rgb[mask]
            point_cloud_d2cam = z[mask]
            point_cloud_seg = seg[mask]
        else:
            # Make sure we don't have duplicate points     
            tree = cKDTree(point_cloud_xyz, balanced_tree = False)
            
            # Identify already present points
            distances, indices = tree.query(pointmap[mask].astype(np.float32))
            already_present = (distances<10e-3)
            already_present_indices = indices[already_present]
            
            # For points already present, check if they are closer to camera from the current view
            distances_of_already_present_indices = point_cloud_d2cam[already_present_indices]        
            distances_of_current_indices = z[mask][already_present]
            seg_of_current_indices = seg[mask][already_present]
            rgb_of_current_indices = rgb[mask][already_present]

            # If they are closer, replace their d2cam, rgb, and segmentation
            better = distances_of_already_present_indices>distances_of_current_indices
            point_cloud_d2cam[already_present_indices][better] = distances_of_current_indices[better]
            point_cloud_seg[already_present_indices][better] = seg_of_current_indices[better]
            point_cloud_rgb[already_present_indices][better] = rgb_of_current_indices[better]
            
            # Add all points that are not already present!
            not_present = ~already_present
            point_cloud_xyz = np.concatenate([point_cloud_xyz, pointmap[mask][not_present].astype(np.float32)])
            point_cloud_rgb = np.concatenate([point_cloud_rgb, rgb[mask][not_present]])
            point_cloud_d2cam = np.concatenate([point_cloud_d2cam, z[mask][not_present]])
            point_cloud_seg = np.concatenate([point_cloud_seg, seg[mask][not_present]])
        

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_xyz)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    _, ind2 = pcd.remove_radius_outlier(nb_points=20, radius=0.04)
    ind_combined = np.intersect1d(ind, ind2)
        
    point_cloud_seg_rgb = np.zeros((point_cloud_seg.shape[0], 3), dtype=np.uint8)
    for k,v in colors.items():
        point_cloud_seg_rgb[point_cloud_seg==k] = v

    df_inliers = pd.DataFrame({
        "x":(point_cloud_xyz)[:,0],
        "y":(point_cloud_xyz)[:,1],
        "z":(point_cloud_xyz)[:,2],
        "r":(point_cloud_rgb)[:,0],
        "g":(point_cloud_rgb)[:,1],
        "b":(point_cloud_rgb)[:,2],
        "r_seg":(point_cloud_seg_rgb)[:,0],
        "g_seg":(point_cloud_seg_rgb)[:,1],
        "b_seg":(point_cloud_seg_rgb)[:,2],
        "r_combined":(point_cloud_rgb)[:,0]//2 + (point_cloud_seg_rgb)[:,0]//2,
        "g_combined":(point_cloud_rgb)[:,1]//2 + (point_cloud_seg_rgb)[:,1]//2,
        "b_combined":(point_cloud_rgb)[:,2]//2 + (point_cloud_seg_rgb)[:,2]//2,
    }).iloc[ind_combined].sample(frac=0.1).reset_index(drop=True)

    df_inliers.to_csv(os.path.join(base_path,"point_cloud.csv"), index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    image_path = args.images
    workspace_path = args.workspace
    database_path = os.path.join(workspace_path, "database.db")
    os.makedirs(workspace_path,exist_ok=True)

    os.system("colmap feature_extractor --ImageReader.camera_model RADIAL --image_path "+image_path+" --database_path "+database_path+" --ImageReader.single_camera 1")
    os.system("colmap exhaustive_matcher --database_path "+database_path)
    os.system("glomap mapper --database_path "+database_path+" --image_path "+image_path+" --output_path "+os.path.join(workspace_path, "sparse"))
    os.system("colmap image_undistorter \
        --image_path "+image_path+" \
        --input_path "+os.path.join(workspace_path, "sparse/0")+" \
        --output_path "+os.path.join(workspace_path, "dense")+" \
        --output_type COLMAP \
        --max_image_size 2000")
    os.system("colmap patch_match_stereo \
        --workspace_path "+os.path.join(workspace_path, "dense")+" \
        --workspace_format COLMAP \
        --PatchMatchStereo.cache_size 64 \
        --PatchMatchStereo.filter true \
        --PatchMatchStereo.window_step 2 \
        --PatchMatchStereo.geom_consistency true")
    
    main(os.path.join(workspace_path,"dense"))

