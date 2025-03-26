
import open3d as o3d
import mmcv
from mmdet.datasets import CustomDataset, DATASETS
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
# from ..core import rot_mat_to_yaw
import os
import os.path as osp
trunc_ignore_thres=0.8,
min_box_size=4.0,
min_visibility=1
CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'barrier')
CAMS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
data = mmcv.load('/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/nuscenes/nuscenes_annotations_val.pkl')  # JSON is loaded as a dictionary

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
                print(data_path)
                image = cv2.imread(data_path)
                filename = os.path.basename(data_path)
                for object_id, ann_record in enumerate(cam_info['ann_records']):
                    visibility = int(ann_record['visibility'])
                    
                    truncation = ann_record['truncation']
                    print(truncation)
                    # visibility and class filtering
                    # if visibility >= min_visibility and ann_record['cat_name'] in CLASSES:
                    bbox = np.array(ann_record['bbox'], dtype=np.int32)
                    wh = bbox[2:] - bbox[:2]

                    gt_bboxes_2d.append(bbox)
                    gt_labels.append(ann_record['cat_id'])

                    x1, y1, x2, y2 = bbox
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2,y2), (0, 255, 0), 2)

                    # Put visibility number
                    text = f"trunc: {truncation}"
                    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 255), 2, cv2.LINE_AA)
                    
                # output_path = os.path.join(f"{sample_token}_{cam}.jpg")
                cv2.imwrite(f'/simplstor/ypatel_dataset/truncation/{filename}', image)