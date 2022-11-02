# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Bbox utils"""

import math
import itertools as it
import numpy as np
from src.model_utils.config import config

# class GeneratDefaultGridCells:
#     def __init__(self):
#         # feature_size = [[40,40],[20,20], [10,10]]
#         # steps = [8, 16, 32]
#         fk = config.img_shape[0] / np.array(config.strides)
#         feature_size = np.array(config.feature_size)
#         scales = np.array(config.scales)
#         strides = np.array(config.strides)
#         anchor_size = np.array(config.anchor_size)
#         self.default_multi_level_grid_cells = []
#         # config.feature_size = [40, 20, 10]
#         for idex, feature_size in enumerate(config.feature_size):
#             base_size = anchor_size[idex] / config.img_shape[0]
#             size = base_size * scales[idex]
#             all_size = []
#             for aspect_ratio in config.aspect_ratios:
#                 w, h = size * math.sqrt(aspect_ratio), size / math.sqrt(aspect_ratio)
#                 all_size.append((h, w))
#
#             stride = strides[idex]
#             h, w = feature_size, feature_size
#             x_range = (np.arange(w)+0.5) * stride
#             y_range = (np.arange(h)+0.5) * stride
#             y_feat, x_feat = np.meshgrid(x_range, y_range)
#             y_feat, x_feat = y_feat.flatten(), x_feat.flatten()
#             grid_cells = np.stack(
#                 [
#                     x_feat - 0.5 * stride,
#                     y_feat - 0.5 * stride,
#                     x_feat + 0.5 * stride,
#                     y_feat + 0.5 * stride
#                 ],
#                 axis=-1
#             )
#
#             self.default_multi_level_grid_cells.append(grid_cells)

class GeneratDefaultGridCellsII:
    def __init__(self):
        fk = config.img_shape[0] / np.array(config.strides)
        scales = np.array(config.scales)
        anchor_size = np.array(config.anchor_size)
        self.default_multi_level_grid_cells = []
        # config.feature_size = [40, 20, 10]
        for idex, feature_size in enumerate(config.feature_size):
            base_size = anchor_size[idex] / config.img_shape[0]
            size = base_size * scales[idex]
            all_sizes = []
            w, h = size * math.sqrt(config.aspect_ratio), size / math.sqrt(config.aspect_ratio)
            all_sizes.append((h, w))
            for i, j in it.product(range(feature_size), repeat=2):
                for h, w in all_sizes:
                    cx, cy = (j + 0.5) / fk[idex], (i + 0.5) / fk[idex]
                    self.default_multi_level_grid_cells.append([cx,cy,h,w])
        def to_ltrb(cy, cx, h, w):
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

        self.default_multi_level_grid_cells_ltrb = np.array(tuple(to_ltrb(*i) for i in self.default_multi_level_grid_cells), dtype='float32')
        self.default_multi_level_grid_cells = np.array(self.default_multi_level_grid_cells, dtype='float32')


default_multi_level_grid_cells_ltrb = GeneratDefaultGridCellsII().default_multi_level_grid_cells_ltrb
default_multi_level_grid_cells = GeneratDefaultGridCellsII().default_multi_level_grid_cells
num_level_cells_list = [1600, 400, 100]
mlvl_grid_cells = default_multi_level_grid_cells_ltrb
y1, x1, y2, x2 = np.split(default_multi_level_grid_cells_ltrb[:, :4], 4, axis=-1)
# The area of Anchor
vol_anchors = (x2 - x1) * (y2 - y1)

# nanodet list
# default_multi_level_grid_cells = GeneratDefaultGridCells().default_multi_level_grid_cells
# num_level_cells_list = [grid_cells.shape[0] for grid_cells in default_multi_level_grid_cells]
# mlvl_grid_cells = np.concatenate(default_multi_level_grid_cells, axis=0, dtype=np.float32)
# y1, x1, y2, x2 = np.split(mlvl_grid_cells[:, :4], 4, axis=-1)
# vol_anchors = (x2 - x1) * (y2 - y1)


def nanodet_bboxes_encode(boxes):
    def bbox_overlaps(bbox):
        ymin = np.maximum(y1, bbox[0])
        xmin = np.maximum(x1, bbox[1])
        ymax = np.minimum(y2, bbox[2])
        xmax = np.minimum(x2, bbox[3])
        # 并行化运算
        w = np.maximum(xmax - xmin, 0.)
        h = np.maximum(ymax - ymin, 0.)

        inter_vol = h * w
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        iou = inter_vol / union_vol
        return np.squeeze(iou)

    def atssAssign(gt_bboxes, gt_labels, overlaps):
        INF = 100000
        num_gt = gt_bboxes.shape[0]
        num_grid_cells = mlvl_grid_cells.shape[0]
        assigned_gt_inds = np.full(shape=(num_grid_cells,), fill_value=0, dtype=np.int32)
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_point = np.stack((gt_cx, gt_cy), axis=1)

        grid_cells_cx = (mlvl_grid_cells[:, 0] + mlvl_grid_cells[:, 2]) / 2.0
        grid_cells_cy = (mlvl_grid_cells[:, 1] + mlvl_grid_cells[:, 3]) / 2.0
        grid_cells_points = np.stack((grid_cells_cx, grid_cells_cy), axis=1)

        distances = grid_cells_points[:, None, :] - gt_point[None, :, :]
        distances = np.power(distances, 2)
        distances = np.sum(distances, axis=-1)
        distances = np.sqrt(distances)

        candidate_idxs = []
        start_idx = 0
        topk = 9
        for level, cells_per_level in enumerate(num_level_cells_list):
            end_idx = start_idx + cells_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(config.topk, cells_per_level)
            topk_idxs_per_level = np.argsort(distances_per_level, axis=0)
            topk_idxs_per_level = topk_idxs_per_level[:selectable_k]
            candidate_idxs.append(topk_idxs_per_level+start_idx)
            start_idx = end_idx
        candidate_idxs = np.concatenate(candidate_idxs, axis=0)
        candidate_overlaps = overlaps[candidate_idxs, np.arange(num_gt)]
        overlaps_mean_per_gt = np.mean(candidate_overlaps, axis=0)
        overlaps_std_per_gt = np.std(candidate_overlaps, axis=0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_grid_cells
        ep_bboxes_cx = np.reshape(grid_cells_cx, (1, -1)).repeat(num_gt, axis=0).reshape(-1)
        ep_bboxes_cy = np.reshape(grid_cells_cy, (1, -1)).repeat(num_gt, axis=0).reshape(-1)
        candidate_idxs = candidate_idxs.reshape(-1)
        l_ = ep_bboxes_cx[candidate_idxs].reshape(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].reshape(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].reshape(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].reshape(-1, num_gt)
        # is_in_gts = np.stack([l_, t_, r_, b_], axis=1).min(axis=1) > 0.01
        # is_pos = is_pos & is_in_gts
        overlaps_inf = np.full_like(overlaps, -INF).T.reshape(-1)
        index = candidate_idxs.reshape(-1)[is_pos.reshape(-1)]
        overlaps_inf[index] = overlaps.T.reshape(-1)[index]
        overlaps_inf = overlaps_inf.reshape(num_gt, -1).T

        max_overlaps = np.max(overlaps_inf, axis=1)
        argmax_overlaps = np.argmax(overlaps_inf, axis=1)

        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = np.full_like(assigned_gt_inds, -1, dtype=np.int32)
            pos_inds = np.nonzero(assigned_gt_inds > 0)[0].squeeze()
            if pos_inds.size > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return num_gt, assigned_gt_inds, max_overlaps, assigned_labels

    def sample(assigned_gt_inds, gt_bboxes):
        pos_inds = np.nonzero(assigned_gt_inds > 0)[0].squeeze()
        neg_inds = np.nonzero(assigned_gt_inds == 0)[0].squeeze()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if len(gt_bboxes.shape) < 2:
            gt_bboxes = gt_bboxes.reshape(-1, 4)
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def target_assign_single_img(pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds, gt_labels):
        num_grid_cells = mlvl_grid_cells.shape[0]

        bbox_targets = np.zeros_like(mlvl_grid_cells, dtype=np.float32)
        bbox_weights = np.zeros_like(mlvl_grid_cells, dtype=np.float32)
        assign_labels = np.full((num_grid_cells,), 80, dtype=np.int32)
        assign_labels_weights = np.zeros((num_grid_cells,), dtype=np.float32)

        if pos_inds.size > 0:
            pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                assign_labels[pos_inds] = 0
            else:
                assign_labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            assign_labels_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            assign_labels_weights[neg_inds] = 1.0

        return assign_labels, assign_labels_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

    def bbox2distance(points, bbox, max_dis=None, eps=0.1):
        left = points[:, 0] - bbox[:, 0]
        top = points[:, 1] - bbox[:, 1]
        right = bbox[:, 2] - points[:, 0]
        bottom = bbox[:, 3] - points[:, 1]
        if max_dis is not None:
            left = np.clip(left, 0, max_dis - eps)
            top = np.clip(top, 0, max_dis - eps)
            right = np.clip(right, 0, max_dis - eps)
            bottom = np.clip(bottom, 0, max_dis - eps)
        return np.stack([left, top, right, bottom], -1)

    def grid_cells_to_center(grid_cells):
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
        return np.stack([cells_cx, cells_cy], axis=-1)

    gt_labels = []
    gt_bboxes = []
    overlaps = []
    for bbox in boxes:
        gt_labels.append(bbox[4]-1)
        label = int(bbox[4])
        gt_bboxes.append(bbox[:4])
        overlap = bbox_overlaps(bbox)
        overlaps.append(overlap)
    overlaps = np.stack(overlaps, axis=-1)
    gt_labels = np.array(gt_labels, dtype=np.int32)
    gt_bboxes = np.stack(gt_bboxes, 0)

    num_gt, assigned_gt_inds, max_overlaps, assigned_labels = atssAssign(gt_bboxes, gt_labels, overlaps)
    pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = sample(assigned_gt_inds, gt_bboxes)
    assign_labels, assign_labels_weights, bbox_targets, bbox_weights, pos_inds, neg_inds = \
        target_assign_single_img(pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds, gt_labels)

    grid_cells = mlvl_grid_cells.reshape(-1, 4)
    bbox_targets = bbox_targets.reshape(-1, 4)
    assign_labels = assign_labels.reshape(-1)
    assign_labels_weights = assign_labels_weights.reshape(-1)
    # bg_class_ind = config.num_classes
    bg_class_ind = 80

    pos_inds = np.nonzero((assign_labels >= 0) & (assign_labels < bg_class_ind))[0]
    score = np.zeros(assign_labels.shape)

    pos_bbox_targets = bbox_targets[pos_inds]
    pos_grid_cells = grid_cells[pos_inds]

    pos_grid_cell_centers = grid_cells_to_center(pos_grid_cells)
    target_corners = bbox2distance(pos_grid_cell_centers, pos_bbox_targets, 7).reshape(-1)

    return pos_inds, pos_grid_cell_centers, pos_bbox_targets, target_corners, assign_labels


def intersect(box_a, box_b):
    """Compute the intersect of two sets of boxes."""
    max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union

