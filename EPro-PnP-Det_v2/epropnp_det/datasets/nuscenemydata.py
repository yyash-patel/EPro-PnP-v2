import open3d as o3d
import mmcv
from mmdet.datasets import CustomDataset, DATASETS
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from ..core import rot_mat_to_yaw

@DATASETS.register_module()
class Nuscenemydata(CustomDataset):
    CLASSES = ('car', 'taxi')  # Include other vehicle types as necessary
    extrinsic = np.eye(4)
    # rotation = np.array([[0.719   ,   - 0.0043,    -0.69],
    #                      [0.0145 ,   -0.9997 ,   0.0212],
    #                      [-0.6945 ,  -0.0253  , -0.7190]])
    # t = np.array([88.48, 4.54, 30.83])#.reshape(3, 1)

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
    width = 990
    height = 990
    crop_box = [0, 318, 990, 990]

    def compute_mask_overlap(self, mask1, mask2):
        # Convert 255 values to 1 for easier calculations
        mask1 = (mask1 == 255).astype(np.uint8)
        mask2 = (mask2 == 255).astype(np.uint8)

        # Compute intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        # Compute IoU (Intersection over Union)
        iou = intersection / union if union != 0 else 0
        return iou

    def load_annotations(self, ann_file):
        """
        Load annotations from the JSON file.

        Args:
            ann_file (str): Path to the JSON annotation file.

        Returns:
            list[dict]: A list of dictionaries, each containing annotation info for a frame.
        """
        # Load JSON file
        data = mmcv.load(ann_file)  # JSON is loaded as a dictionary
        frames = data['frames']

        data_infos = []

        for frame_info in data['infos']:
            sample_token = frame_info['token']
            for cam_id, cam in enumerate(CAMS):
                cam_info = frame_info['cams'][cam]
                data_path = cam_info['data_path']
                pts_path = cam_info.get('pts_path', None)
                cam_intrinsic = cam_info['cam_intrinsic']
                # if path_mapping is not None:
                #     for old, new in path_mapping.items():
                #         data_path = data_path.replace(old, new)
                #         pts_path = pts_path.replace(old, new)
                gt_bboxes_2d = []
                gt_bboxes_ignore = []
                gt_labels = []
                gt_center_2d = []
                gt_bboxes_3d = []

                object_ids = []
                for object_id, ann_record in enumerate(cam_info['ann_records']):
                    visibility = int(ann_record['visibility'])
                    truncation = ann_record['truncation']
                    # visibility and class filtering
                    if visibility >= self.min_visibility and ann_record['cat_name'] in self.CLASSES:
                        bbox = np.array(ann_record['bbox'], dtype=np.float32)
                        wh = bbox[2:] - bbox[:2]
                        # truncation & size filtering
                        if truncation <= self.trunc_ignore_thres and wh.min() >= self.min_box_size:
                            gt_bboxes.append(bbox)
                            gt_labels.append(ann_record['cat_id'])
                            gt_attr.append(ann_record['attr_id'])
                            gt_velo.append(np.array(ann_record['velo'], dtype=np.float32))
                            # gt_truncation.append(truncation)
                            # gt_visibility.append(visibility)
                            object_ids.append(object_id)
                            # convert 3d box into KITTI format
                            bbox3d = ann_record['bbox3d']
                            lhw = bbox3d.wlh[[1, 2, 0]].astype(np.float32)
                            center = bbox3d.center.astype(np.float32)
                            rotation_matrix = bbox3d.rotation_matrix @ self.KITTI2NUS_ROT
                            yaw = rot_mat_to_yaw(rotation_matrix).astype(np.float32)
                            gt_bboxes_3d.append(np.concatenate(
                                [lhw, center, [yaw]]))
                        else:
                            gt_bboxes_ignore.append(bbox)

                if gt_bboxes:
                    gt_bboxes = np.stack(gt_bboxes, axis=0)
                    gt_labels = np.array(gt_labels, dtype=np.int64)
                    gt_attr = np.array(gt_attr, dtype=np.int64)
                    gt_velo = np.stack(gt_velo, axis=0)
                    gt_bboxes_3d = np.stack(gt_bboxes_3d, axis=0)
                    object_ids = np.array(object_ids, dtype=np.int)
                else:
                    gt_bboxes = np.empty((0, 4), dtype=np.float32)
                    gt_labels = np.empty(0, dtype=np.int64)
                    gt_attr = np.empty(0, dtype=np.int64)
                    gt_velo = np.empty((0, 2), dtype=np.float32)
                    gt_bboxes_3d = np.empty((0, 7), dtype=np.float32)
                    object_ids = np.empty(0, dtype=np.int)

                if gt_bboxes_ignore:
                    gt_bboxes_ignore = np.stack(gt_bboxes_ignore, axis=0)
                else:
                    gt_bboxes_ignore = np.empty((0, 4), dtype=np.float32)
                    
                    # gt_bboxes_2d.append([bbox_2d_xyxy])
                    # gt_bboxes_3d.append([length,
                    #                     height,
                    #                     width,
                    #                     bbox_3d_center_in_cam[0],
                    #                     bbox_3d_center_in_cam[1],
                    #                     bbox_3d_center_in_cam[2],
                    #                     -yaw])
                    
                    # gt_labels.append(label_id)
                    # gt_center_2d.append([obj_center[0][0],obj_center[0][1]])
                    gt_x3d.append(points3d_in_cam)
                    gt_x2d.append(projected_x2d)

            # cam_intrinsic = self.intrinsic
            img_transform = np.array(
                                    [[1, 0, -self.crop_box[0]],
                                    [0, 1, -self.crop_box[1]],
                                    [0, 0, 1]], dtype=np.float32) 

            # Add frame information to dataset
            data_infos.append(
                dict(
                    filename=data_path,
                    width=self.width,
                    height=self.height,
                    ann=dict(
                        bboxes=gt_bboxes_3d,
                        bboxes_2d=gt_bboxes_2d,
                        labels=gt_labels,
                        center_2d =gt_center_2d,
                        cam_intrinsic = cam_intrinsic,
                        pts_x3d=gt_x3d,
                        pts_x2d=gt_x2d,
                        img_transform = img_transform
                    )
                )
            )

        return data_infos

    def get_ann_info(self, idx):
        """
        Get annotation info for a specific index.
        Args:
            idx (int): Index of the annotation info.
        Returns:
            dict: Annotation information for the specified index.
        """
        return self.data_infos[idx]['ann']
    # def get_ann_info(self, idx):
    #     return self._parse_ann_info(self.data_infos[idx])
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and len(self.get_ann_info(i)['bboxes_2d']) == 0:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds
    

import mmcv
CAMS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
data = mmcv.load('/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/nuscenes/nuscenes_annotations_val.pkl')
data_infos = []
for frame_info in data['infos']:
    sample_token = frame_info['token']
    for cam_id, cam in enumerate(CAMS):
        cam_info = frame_info['cams'][cam]
        data_path = cam_info['data_path']
        pts_path = cam_info.get('pts_path', None)
        # if path_mapping is not None:
        #     for old, new in path_mapping.items():
        #         data_path = data_path.replace(old, new)
        #         pts_path = pts_path.replace(old, new)
        for object_id, ann_record in enumerate(cam_info['ann_records']):
            print(ann_record.keys())
            dsf