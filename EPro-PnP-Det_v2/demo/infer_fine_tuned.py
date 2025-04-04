"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import argparse
import numpy as np
import open3d as o3d
import mmcv
from scipy.spatial.transform import Rotation as R
import cv2
from shapely.geometry import Polygon
from shapely.affinity import rotate
from scipy.optimize import linear_sum_assignment
from mmcv.utils import track_iter_progress
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

def parse_args():
    parser = argparse.ArgumentParser(description='Infer from images in a directory')
    # parser.add_argument('image_dir', help='directory of input images')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--intrinsic', help='camera intrinsic matrix in .csv format',
                        default='demo/nus_cam_front.csv')
    parser.add_argument(
        '--show-dir', 
        help='directory where visualizations will be saved (default: $IMAGE_DIR/viz)')
    parser.add_argument('--json_file', help='custom annotation data file path')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '--show-score-thr', type=float, default=0.3, help='bbox score threshold for visialization')
    parser.add_argument(
        '--show-views',
        type=str,
        nargs='+',
        help='views to show, e.g., "--show-views 2d 3d bev mc score pts orient" '
             'to fully visulize EProPnPDet')
    args = parser.parse_args()
    return args

def ground_truth(cars):
    gt_bboxes_3d = []
    gt_bbox = []
    gt_label = []
    for j,car in enumerate(cars):
        class_cat = car['car_name']
        # extracted_name = frame_name.rsplit(".", 1)[0]
        # mask_full = f"/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/int_3/{extracted_name}_{class_cat}_full.png"
        # mask_full = cv2.imread(mask_full)
        # mask_occlusion = f"/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/int_3/{extracted_name}_{class_cat}_occlusion.png"
        # mask_occlusion = cv2.imread(mask_occlusion)
        # iou = compute_mask_overlap(mask_full,mask_occlusion)
        # if iou > 0.4:
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

        # 2d bbox --------------------------------------------
        projected_2d, _ = cv2.projectPoints(
                                            bbox_3d_corners, 
                                            c_T_w[:3,:3],              
                                            c_T_w[:3,3],              
                                            intrinsic,     
                                            distCoeffs         
                                            )
        projected_2d = projected_2d.reshape(-1, 2)
        # x_values = projected_2d[:, 0]
        # y_values = projected_2d[:, 1]
        x_values = projected_2d[:, 0]
        y_values = projected_2d[:, 1]
        bbox_2d_xyxy = [min(x_values), 
                        min(y_values), 
                        max(x_values), 
                        max(y_values)]
        
        gt_bbox.append(bbox_2d_xyxy)
        
        gt_bboxes_3d.append([length,
                                        height,
                                        width,
                                        bbox_3d_center_in_cam[0],
                                        bbox_3d_center_in_cam[1],
                                        bbox_3d_center_in_cam[2],
                                        -yaw])
        gt_label.append(label_id)
    return gt_bboxes_3d, gt_bbox,gt_label

def compute_iou_matrix(gt_boxes, pred_boxes):
    """
    gt_boxes: (N, 4)
    pred_boxes: (M, 4)
    Returns IoU matrix of shape (M, N)
    """
    M, N = len(pred_boxes), len(gt_boxes)
    iou_matrix = np.zeros((M, N))

    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            x1 = max(pb[0], gb[0])
            y1 = max(pb[1], gb[1])
            x2 = min(pb[2], gb[2])
            y2 = min(pb[3], gb[3])

            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter_area = inter_w * inter_h

            pb_area = (pb[2] - pb[0]) * (pb[3] - pb[1])
            gb_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
            union_area = pb_area + gb_area - inter_area
            
            iou_matrix[i, j] = inter_area / union_area if union_area > 0 else 0.0

    return iou_matrix 
def create_o3d_box(pred_3d):
    """
    center: (x, y, z)
    size: (l, w, h)
    yaw: in radians
    """
    center = pred_3d[3:6]
    size = pred_3d[0:3]
    yaw = pred_3d[6]

    box = o3d.geometry.OrientedBoundingBox()
    box.center = center
    box.extent = size
    R = box.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    box.R = R
    return box

def get_bev_corners(x, z, yaw, l, w):
    # Compute 2D rectangle corners (BEV, X-Z plane)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    dx = l / 2
    dz = w / 2
    corners = np.array([
        [ dx,  dz],
        [-dx,  dz],
        [-dx, -dz],
        [ dx, -dz]
    ])
    R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    rotated_corners = corners @ R.T
    rotated_corners += np.array([x, z])
    return rotated_corners

def compute_height_overlap(pred_box, gt_box):
    """
    Y-axis overlap
    """
    pred_ymin = pred_box[4] - pred_box[1] / 2
    pred_ymax = pred_box[4] + pred_box[1] / 2
    gt_ymin = gt_box[4] - gt_box[1] / 2
    gt_ymax = gt_box[4] + gt_box[1] / 2

    y_overlap = max(0, min(pred_ymax, gt_ymax) - max(pred_ymin, gt_ymin))
    total_height = max(pred_ymax, gt_ymax) - min(pred_ymin, gt_ymin)
    return y_overlap / total_height if total_height > 0 else 0.0

def matching(gt_bboxes_3d, gt_bbox, gt_label, pred_bbox_3d, pred_bbox_2d):
    
    # gt_bbox = np.array(gt_bbox)
    # pred_bbox_2d = pred_bbox_2d.reshape(-1,5)
    iou_matrix = np.zeros((len(pred_bbox_3d), len(gt_bboxes_3d)))

    for i,pred_3d in enumerate(pred_bbox_3d):
        pred_bev_corner = get_bev_corners(pred_3d[3],
                                          pred_3d[5],
                                          pred_3d[6],
                                          pred_3d[0],
                                          pred_3d[2])
        for j,gt_3d in enumerate(gt_bboxes_3d):
            gt_bev_corner = get_bev_corners(gt_3d[3],gt_3d[5],gt_3d[6],gt_3d[0],gt_3d[2])
            poly1 = Polygon(pred_bev_corner)
            poly2 = Polygon(gt_bev_corner)
            if not poly1.is_valid or not poly2.is_valid:
                return 0.0
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area

            bev_iou = inter_area / union_area if union_area > 0 else 0.0
            height_overlap = compute_height_overlap(pred_3d,gt_3d)
            iou = bev_iou * height_overlap
            iou_matrix[i, j] = iou
    cost_matrix = 1 - iou_matrix
    pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

    matches = []
    for p, g in zip(pred_idx, gt_idx):
        if iou_matrix[p, g] >= 0.3:
            matches.append((p, g))
    return matches, iou_matrix

def eval(matches, pred_bbox_3d, gt_bbox_3d,gt_label):
    error_data = []
    for m in matches:
        pred_ind = m[0]
        gt_ind = m[1]

        pred_match = pred_bbox_3d[pred_ind]
        pred_x, pred_z, pred_yaw = pred_match[3], pred_match[5], pred_match[6]


        gt_match = gt_bbox_3d[gt_ind]
        gt_x, gt_z, gt_yaw = gt_match[3], gt_match[5], gt_match[6]

        dist = np.sqrt((pred_x - gt_x)**2 + (pred_z - gt_z)**2)

        diff = np.abs(pred_yaw - gt_yaw) % (2 * np.pi)
        aoe = min(diff, 2 * np.pi - diff)
        label_match = gt_label[gt_ind]
        error_data.append({
            'label': label_match,
            'ATE': dist,
            'AOE': aoe,
        })
    return error_data
def final_average(all_frame_metric):
    # label_ate_values = defaultdict(list)

    # Flatten and collect all ATE values
    all_ate = [item['ATE'] for group in all_frame_metric for item in group]

    # Compute overall average ATE
    overall_average_ate = np.mean(all_ate)

    # Print result
    print(f"Overall Average ATE: {overall_average_ate:.4f}")
def main():
    args = parse_args()
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_ids = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    if len(gpu_ids) != 1:
        raise NotImplementedError('multi-gpu inference is not yet supported')

    from mmcv.utils import track_iter_progress
    from mmcv.cnn import fuse_conv_bn
    from epropnp_det.apis import init_detector, inference_detector, show_result

    # image_dir = args.image_dir
    # assert os.path.isdir(image_dir)
    show_dir = args.show_dir
    # if show_dir is None:
    #     show_dir = os.path.join(image_dir, 'viz')
    os.makedirs(show_dir, exist_ok=True)
    cam_mat = np.loadtxt(args.intrinsic, delimiter=',').astype(np.float32)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    model = fuse_conv_bn(model)
    model.test_cfg['debug'] = args.show_views if args.show_views is not None else []

    # img_list = []
    # for filename in os.listdir(image_dir):
    #     if os.path.splitext(filename)[1] in ['.jpg', '.jpeg', '.png']:
    #         img_list.append(filename)
    # img_list.sort()
    kwargs = dict(views=args.show_views) if args.show_views is not None else dict()

    data = mmcv.load(os.path.join(args.json_file, 'test.json'))  # JSON is loaded as a dictionary
    frames = data['frames']
    all_frame_metric = []
    e=0
    mismatched_count=0
    for i,frame in enumerate(frames):
        frame_name = frame['frameName']
        cars = frame['cars']

        result, data = inference_detector(
            model, [os.path.join(args.json_file, frame_name)], cam_mat)
        
        gt_bboxes_3d, gt_bbox, gt_label = ground_truth(cars)
        # ori_img_path = os.path.join('/simplstor/ypatel_dataset/int3_test_results/gt_synthetic',f'image_{i}.png')

        pred_bbox_3d, pred_bbox_2d = show_result(
            model, result, data,
            show=False, out_dir=show_dir, show_score_thr=args.show_score_thr,
            **kwargs)
    #     if len(pred_bbox_3d) == 0:
    #         e=e+1
    #     elif len(pred_bbox_3d) != len(gt_bboxes_3d):
    #         mismatched_count += 1
    # print(e)
    # print(mismatched_count)   

    #     matches, iou_matrix = matching(gt_bboxes_3d, gt_bbox, gt_label,pred_bbox_3d, pred_bbox_2d)
    #     error_data = eval(matches, pred_bbox_3d, gt_bboxes_3d,gt_label)
    #     all_frame_metric.append(error_data)

    # final_average(all_frame_metric)

if __name__ == '__main__':
    main()
