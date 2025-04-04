
import open3d as o3d
import mmcv
from mmdet.datasets import CustomDataset, DATASETS
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
# from ..core import rot_mat_to_yaw
import os
import os.path as osp
# trunc_ignore_thres=0.8,
# min_box_size=4.0,
# min_visibility=1
# CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
#             'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
#             'barrier')
# CAMS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
#         'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
# data = mmcv.load('/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/nuscenes/nuscenes_annotations_val.pkl')  # JSON is loaded as a dictionary

# data_infos = []

# for frame_info in data['infos']:
#             sample_token = frame_info['token']
#             for cam_id, cam in enumerate(CAMS):
#                 cam_info = frame_info['cams'][cam]
        
#                 data_path = cam_info['data_path']
#                 pts_path = cam_info.get('pts_path', None)
#                 cam_intrinsic = cam_info['cam_intrinsic']
#                 # if path_mapping is not None:
#                 #     for old, new in path_mapping.items():
#                 #         data_path = data_path.replace(old, new)
#                 #         pts_path = pts_path.replace(old, new)
#                 gt_bboxes_2d = []
#                 gt_bboxes_ignore = []
#                 gt_labels = []
#                 gt_center_2d = []
#                 gt_bboxes_3d = []

#                 object_ids = []
#                 print(data_path)
#                 image = cv2.imread(data_path)
#                 filename = os.path.basename(data_path)
#                 for object_id, ann_record in enumerate(cam_info['ann_records']):
#                     visibility = int(ann_record['visibility'])
                    
#                     truncation = ann_record['truncation']
#                     print(truncation)
#                     # visibility and class filtering
#                     # if visibility >= min_visibility and ann_record['cat_name'] in CLASSES:
#                     bbox = np.array(ann_record['bbox'], dtype=np.int32)
#                     wh = bbox[2:] - bbox[:2]

#                     gt_bboxes_2d.append(bbox)
#                     gt_labels.append(ann_record['cat_id'])

#                     x1, y1, x2, y2 = bbox
                    
#                     # Draw bounding box
#                     cv2.rectangle(image, (x1, y1), (x2,y2), (0, 255, 0), 2)

#                     # Put visibility number
#                     text = f"trunc: {truncation}"
#                     cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                                 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                    
#                 # output_path = os.path.join(f"{sample_token}_{cam}.jpg")
#                 cv2.imwrite(f'/simplstor/ypatel_dataset/truncation/{filename}', image)
extrinsic = np.eye(4)

rotation = np.array([[0.71998532   ,   -0.0251382,    -0.69353386],
                    [-0.01445758 ,   -0.99967017 ,    0.02122559],
                    [-0.69383868 ,  -0.00525529  , -0.72011129]])
t = np.array([88.4800033569336,4.539999961853027, 30.829999923706056])

extrinsic[:3, :3] =  rotation
extrinsic[:3, 3] = t
c_T_w = np.linalg.inv(extrinsic)
intrinsic = np.array([[2262.56, 0, 494.75], 
                [0, 2262.56, 494.75], 
                [0, 0, 1]])
distCoeffs = np.zeros(5)
def draw_3d_bbox(image, projected_points, color=(0, 255, 0), thickness=2):
    """
    Draws a 3D bounding box on the given image using projected 2D points.

    :param image: Input image (numpy array).
    :param projected_points: 8x2 array of projected 2D points.
    :param color: Color of the bounding box (default is green).
    :param thickness: Line thickness.
    """
    projected_points = projected_points.astype(int)

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges connecting top and bottom
    ]
    for edge in edges:
        pt1 = tuple(projected_points[edge[0]])
        pt2 = tuple(projected_points[edge[1]])
        cv2.line(image, pt1, pt2, color, thickness)
    
    return image
def proj_to_img(pts, proj_mat, z_clip=1e-4):
    pts_2d = pts @ proj_mat.T

    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:].clip(min=z_clip)
    return pts_2d
def yaw_to_rot_mat(yaw):
    """
    Args:
        yaw: (*)

    Returns:
        rot_mats: (*, 3, 3)
    """

    pkg = np
    device_kwarg = dict()
    sin_yaw = pkg.sin(yaw)
    cos_yaw = pkg.cos(yaw)
    # [[ cos_yaw, 0, sin_yaw],
    #  [       0, 1,       0],
    #  [-sin_yaw, 0, cos_yaw]]
    rot_mats = pkg.zeros(yaw.shape + (3, 3), dtype=pkg.float32, **device_kwarg)
    rot_mats[..., 0, 0] = cos_yaw
    rot_mats[..., 2, 2] = cos_yaw
    rot_mats[..., 0, 2] = sin_yaw
    rot_mats[..., 2, 0] = -sin_yaw
    rot_mats[..., 1, 1] = 1
    return rot_mats
def compute_box_3d(bbox_3d):
    """
    Args:
        bbox_3d: (*, 7)

    Returns:
        corners: (*, 8, 3)
        edge_corner_idx: (12, 2)
    """
    bs = bbox_3d.shape[:-1]
    rotation_matrix = yaw_to_rot_mat(bbox_3d[..., 6])  # (*bs, 3, 3)
    edge_corner_idx = np.array([[0, 1],
                         [1, 2],
                         [2, 3],
                         [3, 0],
                         [4, 5],
                         [5, 6],
                         [6, 7],
                         [7, 4],
                         [0, 4],
                         [1, 5],
                         [2, 6],
                         [3, 7]])
    # corners = np.array([[ 0.5,  0.5,  0.5],
    #                     [ 0.5,  0.5, -0.5],
    #                     [-0.5,  0.5, -0.5],
    #                     [-0.5,  0.5,  0.5],
    #                     [ 0.5, -0.5,  0.5],
    #                     [ 0.5, -0.5, -0.5],
    #                     [-0.5, -0.5, -0.5],
    #                     [-0.5, -0.5,  0.5]], dtype=np.float32)
    corners = np.array([[ 0.5,  0.5,  0.5],
                        [ 0.5,  0.5, -0.5],
                        [-0.5,  0.5, -0.5],
                        [-0.5,  0.5,  0.5],
                        [ 0.5, -0.5,  0.5],
                        [ 0.5, -0.5, -0.5],
                        [-0.5, -0.5, -0.5],
                        [-0.5, -0.5,  0.5]], dtype=np.float32)
    corners = corners * bbox_3d[..., None, :3]  # (*bs, 8, 3)
    corners = (rotation_matrix[..., None, :, :] @ corners[..., None]).reshape(*bs, 8, 3) \
              + bbox_3d[..., None, 3:6]
    return corners, edge_corner_idx
def compute_mask_overlap(mask1, mask2):
    # Convert 255 values to 1 for easier calculations
    mask1 = (mask1 == 255).astype(np.uint8)
    mask2 = (mask2 == 255).astype(np.uint8)

    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Compute IoU (Intersection over Union)
    iou = intersection / union if union != 0 else 0
    return iou
def ground_truth(cars,frame_name):
    img_path = f"/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/int3_test/{frame_name}"
    img = cv2.imread(img_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    for j,car in enumerate(cars):
        class_cat = car['car_name']
        extracted_name = frame_name.rsplit(".", 1)[0]
        mask_full = f"/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/int3_test/{extracted_name}_{class_cat}_full.png"
        mask_full = cv2.imread(mask_full)
        mask_occlusion = f"/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/int3_test/{extracted_name}_{class_cat}_occlusion.png"
        mask_occlusion = cv2.imread(mask_occlusion)
        iou = compute_mask_overlap(mask_full,mask_occlusion)
        if iou > 0.4:
            if class_cat[:3] in ('Car', 'Pol', 'Jee', 'Tax'):
                label_id = 0
                # gt_labels.append(0)  
            elif  class_cat[:3] in ('Cit', 'Min', 'Sch'):
                label_id = 3
                # gt_labels.append(3)
            else:
                label_id = 1
                # gt_labels.append(1)

            # 3d bbox center i.e. xyz to camera space -------------------
            bbox_3d_center = np.array([-car['bounding_box_center']['x'], 
                                    car['bounding_box_center']['y'], 
                                    car['bounding_box_center']['z']])
            bbox3d_center_homo = np.append(bbox_3d_center, 1) 
            bbox_3d_center_in_cam_homo = c_T_w @ bbox3d_center_homo
            bbox_3d_center_in_cam = bbox_3d_center_in_cam_homo[:-1].flatten()

            # yaw angle -------------------------------------------------
            bbox_3d_yaw = car["bounding_box_rotation"]
            rot = R.from_quat([bbox_3d_yaw['x'],bbox_3d_yaw['y'],bbox_3d_yaw['z'],bbox_3d_yaw['w']])
            rot_mat= rot.as_matrix()
            rot_mat[:, 0] = -rot_mat[:,0]
            r_c =  np.linalg.inv(rotation) @ rot_mat
            yaw = np.arctan2(r_c[0, 2], r_c[0, 0])
            
            # 8 bbox corners to camera space -----------------------------
            bbox_3d_corners = car["bounding_box"]
            bbox_3d_corners = [[-pt['x'], pt['y'], pt['z']] for pt in bbox_3d_corners]
            bbox_3d_corners = np.array(bbox_3d_corners, dtype=np.float32)

            points = o3d.utility.Vector3dVector(bbox_3d_corners) 
            # Compute oriented bounding box
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(points)
            # Extract length, width, and height from bounding box extents
            length, width, height = obb.extent
            bbox_3d = np.array([[length,
                                            height,
                                            width,
                                            bbox_3d_center_in_cam[0],
                                            bbox_3d_center_in_cam[1],
                                            bbox_3d_center_in_cam[2],
                                            -yaw]])
            corners, edge_idx = compute_box_3d(bbox_3d)

            projected_corners= proj_to_img(corners.reshape(8,3),intrinsic)
            image_vis = draw_3d_bbox(img_rgb, projected_corners)

            # 2d bbox --------------------------------------------
            projected_2d, _ = cv2.projectPoints(
                                                bbox_3d_corners, 
                                                c_T_w[:3,:3],              
                                                c_T_w[:3,3],              
                                                intrinsic,     
                                                distCoeffs         
                                                )
            projected_2d = projected_2d.reshape(-1, 2)
    if iou <0.4:
        return img_rgb
    return image_vis  


data = mmcv.load(os.path.join('/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/int3_test', 'test.json'))  # JSON is loaded as a dictionary
frames = data['frames']

for i,frame in enumerate(frames):
    print(i)
    frame_name = frame['frameName']
    cars = frame['cars']
    img = ground_truth(cars,frame_name)
    cv2.imwrite(f'/simplstor/ypatel_dataset/int3_test_results/gt_synthetic/image_{i}.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
