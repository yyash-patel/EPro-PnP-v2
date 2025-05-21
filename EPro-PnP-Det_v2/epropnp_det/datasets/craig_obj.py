import open3d as o3d
import mmcv
from mmdet.datasets import CustomDataset, DATASETS
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

@DATASETS.register_module()
class craig_syn(CustomDataset):
    # CLASSES = ('car', 'taxi')  # Include other vehicle types as necessary
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
    extrinsic = np.eye(4)
    c_T_w = np.linalg.inv(extrinsic)
    intrinsic = np.array([[571.76166, 0, 500.57865], 
                    [0,  762.343, 470.111], 
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
        for i, frame in enumerate(frames):
            frame_name = frame['frameName']
            if i == 0 :
                continue
            # print(frame_name)
            # image_ = cv2.imread(f'/simplstor/ypatel/datasets/NUSCENES/rendered_images/{frame_name}')
            # print(image_)
            # height,width,_ = image_.shape

            # if frame_name.startswith('frame_0423'):
            #     continue
            cars = frame['cars']
            gt_bboxes_3d = []
            gt_bboxes_2d = []
            gt_center_2d = []
            gt_labels = []
            gt_x3d = []
            gt_x2d = []
            for j,car in enumerate(cars):
                class_cat = car['car_name']

                # extracted_name = frame_name.rsplit(".", 1)[0]
                # mask_full = f"/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/craig_fifth/{extracted_name}_{class_cat}_full.png"
                # mask_full = cv2.imread(mask_full)
                # mask_occlusion = f"/simplstor/ypatel/workspace/EPro-PnP-v2/EPro-PnP-Det_v2/data/craig_fifth/{extracted_name}_{class_cat}_occlusion.png"
                # mask_occlusion = cv2.imread(mask_occlusion)
                # iou = self.compute_mask_overlap(mask_full,mask_occlusion)
                # if iou > 0.4:
                # if 0.6 <= iou <=0.8:

                if class_cat[:3] in ('Car', 'Pol', 'Jee', 'Tax'):
                    label_id = 0
                    # gt_labels.append(0)  
                elif  class_cat[:3] in ('Cit', 'Min', 'Sch'):
                    label_id = 3
                    # gt_labels.append(3)
                else:
                    label_id = 1
                    # gt_labels.append(1)

                # object 3d points 
                points3d = car["points"]
                points3d = [[pt['x'], pt['y'], pt['z']] for pt in points3d]
                points3d = np.array(points3d, dtype=np.float32)

                points3d_homo = np.hstack((points3d, np.ones((points3d.shape[0], 1), dtype=np.float32)))
                points3d_in_cam_homo = (self.c_T_w @ points3d_homo.T).T
                points3d_in_cam = points3d_in_cam_homo[:, :3]

                # object 2d points
                projected_x2d, _ = cv2.projectPoints(
                                                    points3d, 
                                                    self.c_T_w[:3,:3],              
                                                    self.c_T_w[:3,3],              
                                                    self.intrinsic,     
                                                    self.distCoeffs         
                                                    )
                projected_x2d = projected_x2d.reshape(-1,2)

                # 3d bbox center i.e. xyz to camera space -------------------
                bbox_3d_center = np.array([car['bounding_box_center']['x'], 
                                        car['bounding_box_center']['y'], 
                                        car['bounding_box_center']['z']])
                bbox3d_center_homo = np.append(bbox_3d_center, 1) 
                bbox_3d_center_in_cam_homo = self.c_T_w @ bbox3d_center_homo
                bbox_3d_center_in_cam = bbox_3d_center_in_cam_homo[:-1].flatten()

                # yaw angle -------------------------------------------------
                bbox_3d_yaw = car["bounding_box_rotation"]
                size = car["size"]

                # 8 bbox corners to camera space -----------------------------
                bbox_3d_corners = car["bounding_box"]
                bbox_3d_corners = [[pt['x'], pt['y'], pt['z']] for pt in bbox_3d_corners]
                bbox_3d_corners = np.array(bbox_3d_corners, dtype=np.float32)

                points = o3d.utility.Vector3dVector(bbox_3d_corners) 
                # Compute oriented bounding box
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(points)
                # Extract length, width, and height from bounding box extents
                # length, width, height = obb.extent

                # 2d object centers -----------------------------------------
                projected_2d_center, _ = cv2.projectPoints(
                                                    bbox_3d_center, 
                                                    self.c_T_w[:3,:3],              
                                                    self.c_T_w[:3,3],              
                                                    self.intrinsic,     
                                                    self.distCoeffs         
                                                    )
                obj_center = projected_2d_center.reshape(-1, 2)
                
                # 2d bbox --------------------------------------------
                projected_2d, _ = cv2.projectPoints(
                                                    bbox_3d_corners, 
                                                    self.c_T_w[:3,:3],              
                                                    self.c_T_w[:3,3],              
                                                    self.intrinsic,     
                                                    self.distCoeffs         
                                                    )
                projected_2d = projected_2d.reshape(-1, 2)
                x_values = projected_2d[:, 0]
                y_values = projected_2d[:, 1]
                x1 = min(x_values)
                y1 = min(y_values)-self.crop_box[1]
                x2 = max(x_values)
                y2 = max(y_values)-self.crop_box[1]
                x1_clamped = max(0, int(x1))
                y1_clamped = max(0, int(y1))
                x2_clamped = min(self.width - 1, int(x2))
                y2_clamped = min(self.height - 1, int(y2))
                # bbox_2d_xyxy = [min(x_values), 
                #                 min(y_values)-self.crop_box[1], 
                #                 max(x_values), 
                #                 max(y_values)-self.crop_box[1]]
                bbox_2d_xyxy = [x1_clamped, 
                                y1_clamped, 
                                x2_clamped, 
                                y2_clamped]
                if any(val < 0 for val in bbox_2d_xyxy):
                    continue
                
                gt_bboxes_2d.append([bbox_2d_xyxy])
                gt_bboxes_3d.append([size[0],
                                    size[1],
                                    size[2],
                                    bbox_3d_center_in_cam[0],
                                    bbox_3d_center_in_cam[1],
                                    bbox_3d_center_in_cam[2],
                                    bbox_3d_yaw])
                
                gt_labels.append(label_id)
                gt_center_2d.append([obj_center[0][0],obj_center[0][1]-self.crop_box[1]])
                gt_x3d.append(points3d_in_cam)
                gt_x2d.append(projected_x2d)
  
            cam_intrinsic = self.intrinsic
            img_transform = np.array(
                                    [[1, 0, -self.crop_box[0]],
                                    [0, 1, -self.crop_box[1]],
                                    [0, 0, 1]], dtype=np.float32) 
            # Add frame information to dataset

            data_infos.append(
                dict(
                    filename=f'{i}.png',
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

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and len(self.get_ann_info(i)['bboxes_2d']) == 0:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds