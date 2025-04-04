# Modified from https://github.com/tjiiv-cprg/EPro-PnP

"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import cv2
import numpy as np

from .. import compute_box_3d


def compute_box_bev(label, with_arrow=True):
    ry = label[6]
    bl = label[0]
    bw = label[2]
    t = label[3:6][:, None]
    r_mat = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, +np.cos(ry)]])
    if with_arrow:
        corners = np.array([[bl / 2, bl / 2, -bl / 2, -bl / 2,
                             bl / 2, bl / 2 + bw / 2, bl / 2],
                            [0, 0, 0, 0, 0, 0, 0],
                            [bw / 2, -bw / 2, -bw / 2, bw / 2,
                             bw / 2, 0, -bw / 2]])
    else:
        corners = np.array([[bl / 2, bl / 2, -bl / 2, -bl / 2, bl / 2],
                            [0, 0, 0, 0, 0],
                            [bw / 2, -bw / 2, -bw / 2, bw / 2, bw / 2]])
    corners = r_mat @ corners + t
    return corners


def show_bev(
        img, bbox_results, bbox_3d_results, cali_mat, width=None, height=None,
        scale=10, pose_samples=None, pose_sample_weights=None, orientation=None,
        gt_bboxes_3d=None, gt_orientation=None, score_thr=0.1, thickness=2,
        frustum_color=(220, 220, 220), box_color=(10, 60, 240), gt_box_color=(20, 180, 20),
        sample_color=(0.9, 0.5, 0.1), intensity_scale=40, sample_size=3,
        range_mark=[10, 20, 30, 40, 50, 60],
        num_orient_bin=24, orient_size=0.5, orient_color=(128, 128, 128)):
    """
    Args:
        bbox_results (list[ndarray]): multiclass results,
            in format [x1, y1, x2, y2, score]
        bbox_3d_results (list[ndarray]): multiclass results,
            in format [l, h, w, x, y, z, ry, score, ...]
    """
    if (width is None) or (height is None):
        height, width = img.shape[:2]
    bev_img = np.full((height, width, 3), 255, dtype=np.uint8)
    origin = np.array([int(width / 16), int(height / 2)])
    # preproc
    fx = cali_mat[0, 0]
    cx = cali_mat[0, 2]
    proj_mat = np.array(
        [[0, scale],
         [scale, 0]])
    # draw FOV line
    fov_line_x_extend = (-100, 100)
    end_pt_left = np.array([fov_line_x_extend[0],
                            -(fx * fov_line_x_extend[0] / cx)])
    end_pt_right = np.array([fov_line_x_extend[1],
                             -(fx * fov_line_x_extend[1] / (cx - img.shape[1] + 1))])
    end_pt_left = np.round(proj_mat @ end_pt_left) + origin
    end_pt_right = np.round(proj_mat @ end_pt_right) + origin
    frustum = np.stack((origin,
                        end_pt_right,
                        end_pt_left), axis=0).astype(np.int)
    cv2.fillPoly(bev_img, pts=[frustum], color=frustum_color)
    for range_mark_single in range_mark:
        cv2.circle(bev_img, origin * 8, round(range_mark_single * 8 * scale),
                   (255, 255, 255), thickness=thickness, shift=3)

    bbox_3d_results = np.concatenate(bbox_3d_results, axis=0)
    score = bbox_3d_results[:, 7]
    mask = score >= score_thr
    bbox_3d_results = bbox_3d_results[mask]

    if pose_samples is not None and pose_sample_weights is not None:
        pose_samples = np.concatenate(pose_samples, axis=0)
        pose_sample_weights = np.concatenate(pose_sample_weights, axis=0)
        scores = np.concatenate([bbox_single[:, 4] for bbox_single in bbox_results], axis=0)
        pose_sample_weights_ = pose_sample_weights * (scores[:, None] * intensity_scale)
        # (num_obj, num_sample, 2)
        pose_bev = (np.round(pose_samples[..., [0, 2]] @ proj_mat.T) + origin).astype(np.int64)
        in_bev_mask = (pose_bev >= 0).all(axis=-1) & (pose_bev[..., 0] < width) & (pose_bev[..., 1] < height)
        pose_bev = pose_bev[in_bev_mask]  # (n, 2)
        pose_sample_weights_ = pose_sample_weights_[in_bev_mask]  # (n, )
        sample_inds = pose_bev[:, 1] * width + pose_bev[:, 0]  # (n, )
        density = np.bincount(
            sample_inds, weights=pose_sample_weights_, minlength=height * width
        ).reshape(height, width)
        density = cv2.blur(density, [sample_size, sample_size]) * (sample_size * sample_size)
        colored_density = np.power(sample_color, density[..., None])
        bev_img = (bev_img * colored_density).astype(np.uint8)

        # draw orientation distr
        pose_samples = pose_samples[mask]
        pose_sample_weights = pose_sample_weights[mask]
        bbox_yaw = bbox_3d_results[:, 6]
        bin_half_width = (2 * np.pi) / (num_orient_bin * 2)
        sample_yaw_relative = pose_samples[..., 3] - bbox_yaw[:, None]  # (num_obj, num_sample)
        sample_yaw_relative = (sample_yaw_relative + bin_half_width) % (2 * np.pi) - bin_half_width
        yaw_relative_grid = np.linspace(
            0, 2 * np.pi, num=num_orient_bin, endpoint=False, dtype=np.float32)
        for bbox_single, sample_yaw_relative_single, sample_weights_single in zip(
                bbox_3d_results, sample_yaw_relative, pose_sample_weights):
            bbox_xz = bbox_single[[3, 5]]
            orient_histo = np.histogram(
                sample_yaw_relative_single, bins=num_orient_bin,
                range=(-bin_half_width, 2 * np.pi - bin_half_width), weights=sample_weights_single)[0]
            yaw_grid = yaw_relative_grid + bbox_single[6]
            points_xz = np.stack(
                [np.cos(yaw_grid), -np.sin(yaw_grid)], axis=0
            ) * (np.sqrt(orient_histo * num_orient_bin) * orient_size) + bbox_xz[:, None]
            points_bev = (proj_mat @ points_xz).T + origin
            points_bev = (points_bev * 8).astype(np.int32)
            cv2.polylines(bev_img, points_bev[None, ...], True,
                          orient_color, thickness=thickness, shift=3)
            # cv2.line(bev_img,
            #          points_bev[0],
            #          ((proj_mat @ bbox_xz + origin) * 8).astype(np.int32),
            #          color=box_color, thickness=thickness, shift=3)

    if orientation is not None:
        orientation = np.concatenate(orientation, axis=0)
        orientation = orientation[mask]

    # draw det results:
    for i, bbox_3d_result_single in enumerate(bbox_3d_results):
        # draw boxes
        with_arrow = orientation[i] if (
            orientation is not None and (pose_samples is None or pose_sample_weights is None)
        ) else None
        corners = compute_box_bev(bbox_3d_result_single, with_arrow=with_arrow)
        corners_bev = (proj_mat @ corners[[0, 2], :]).T + origin
        cv2.polylines(bev_img, (corners_bev * 8).astype(np.int32)[None, ...], False,
                      box_color, thickness=thickness, shift=3)

    if gt_bboxes_3d is not None and gt_orientation is not None:
        for gt_bbox_3d, with_arrow in zip(gt_bboxes_3d, gt_orientation):
            corners = compute_box_bev(gt_bbox_3d, with_arrow=with_arrow)
            corners_bev = (proj_mat @ corners[[0, 2], :]).T + origin
            cv2.polylines(bev_img, (corners_bev * 8).astype(np.int32)[None, ...], False,
                          gt_box_color, thickness=thickness, shift=3)

    return bev_img


def draw_box_3d_pred(image, bbox_3d_results, cam_intrinsic, score_thr=0.1, z_clip=0.1,
                     color=(10, 60, 240), thickness=2):
    """
    Args:
        bbox_3d_results (list[ndarray]): multiclass results,
            in format [l, h, w, x, y, z, ry, score, ...]
    """
    bbox_3d_results = np.concatenate(bbox_3d_results, axis=0)
    sort_idx = np.argsort(bbox_3d_results[:, 5])[::-1]
    bbox_3d_results = bbox_3d_results[sort_idx]
    bbox_3d_list = []
    projected_corners_list = []
    for bbox_3d in bbox_3d_results:
        if bbox_3d[7] < score_thr:
            continue
        bbox_3d_list.append(bbox_3d[:8])
        corners, edge_idx = compute_box_3d(bbox_3d)
        corners_in_front = corners[:, 2] >= z_clip
        edges_0_in_front = corners_in_front[edge_idx[:, 0]]
        edges_1_in_front = corners_in_front[edge_idx[:, 1]]
        edges_in_front = edges_0_in_front & edges_1_in_front
        edge_idx_in_front = edge_idx[edges_in_front]
        # project to image
        corners_2d = (proj_to_img(corners, cam_intrinsic, z_clip=z_clip)
                      * 8).astype(np.int)
        if np.any(edges_in_front):
            lines = np.stack([corners_2d[edge_idx_single]
                              for edge_idx_single in edge_idx_in_front],
                             axis=0)  # (n, 2, 2)
            cv2.polylines(image, lines, False, color,
                          thickness=thickness, shift=3)
            projected_corners_list.append(corners_2d)
        # compute intersection
        edges_clipped = edges_0_in_front ^ edges_1_in_front
        if np.any(edges_clipped):
            edge_idx_to_clip = edge_idx[edges_clipped]
            edges_0 = corners[edge_idx_to_clip[:, 0]]
            edges_1 = corners[edge_idx_to_clip[:, 1]]
            z0 = edges_0[:, 2]
            z1 = edges_1[:, 2]
            weight_0 = z1 - z_clip
            weight_1 = z_clip - z0
            intersection = (edges_0 * weight_0[:, None] + edges_1 * weight_1[:, None]
                            ) * (1 / (z1 - z0)).clip(min=-1e6, max=1e6)[:, None]
            keep_idx = np.where(z0 > z_clip,
                                edge_idx_to_clip[:, 0],
                                edge_idx_to_clip[:, 1])
            intersection_2d = (proj_to_img(intersection, cam_intrinsic, z_clip=z_clip)
                               * 8).astype(np.int)  # (n, 2)
            keep_2d = corners_2d[keep_idx]  # (n, 2)
            lines = np.stack([keep_2d, intersection_2d], axis=1)  # (n, 2, 2)
            cv2.polylines(image, lines, False, color,
                          thickness=thickness, shift=3)
            projected_corners_list.append(corners_2d)
    return bbox_3d_list, projected_corners_list


def proj_to_img(pts, proj_mat, z_clip=1e-4):
    pts_2d = pts @ proj_mat.T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:].clip(min=z_clip)
    return pts_2d
