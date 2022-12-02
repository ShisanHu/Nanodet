import math
import itertools as it
import numpy as np
from src.model_utils.config import config


class GeneratDefaultGridCells:
    def __init__(self):
        self.center_priors = []
        anchor_size = np.array(config.anchor_size)
        for i, feature_size in enumerate(config.feature_size):
            w, h = anchor_size[i], anchor_size[i]
            for i, j in it.product(range(feature_size), repeat=2):
                cx, cy = (j + 0.5) * w, (i + 0.5) * w
                self.center_priors.append([cy, cx, h, w])

        def to_ltrb(cy, cx, h, w):
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

        self.center_priors = np.array(self.center_priors, dtype='float32')
        self.center_priors_ltrb = np.array(tuple(to_ltrb(*i) for i in self.center_priors), dtype='float32')

center_priors = GeneratDefaultGridCells().center_priors
center_priors_ltrb = GeneratDefaultGridCells().center_priors_ltrb
y1, x1, y2, x2 = np.split(center_priors_ltrb[:, :4], 4, axis=-1)
num_level_cells_list = [1600, 400, 100]
# The area of Anchor
vol_anchors = (x2 - x1) * (y2 - y1)


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

    def atssAssign(bbox):
        INF = 100000
        num_grid_priors = center_priors_ltrb.shape[0]
        grid_priors = center_priors_ltrb
        overlaps = bbox_overlaps(bbox)

        gt_cy = (bbox[0] + bbox[2]) / 2.0
        gt_cx = (bbox[1] + bbox[3]) / 2.0
        gt_points = np.stack((gt_cy, gt_cx), axis=0)

        grid_priors_cy = (grid_priors[:, 0] + grid_priors[:, 2]) / 2.0
        grid_priors_cx = (grid_priors[:, 1] + grid_priors[:, 3]) / 2.0
        grid_cells_points = np.stack((grid_priors_cy, grid_priors_cx), axis=1)

        distances = grid_cells_points - gt_points
        distances = np.power(distances, 2)
        distances = np.sum(distances, axis=-1)
        distances = np.sqrt(distances)

        candidate_idxs = []
        start_idx = 0
        topk = 9
        for level, cells_per_level in enumerate(num_level_cells_list):
            end_idx = start_idx + cells_per_level
            distances_per_level = distances[start_idx:end_idx]
            selectable_k = min(config.topk, cells_per_level)
            topk_idxs_per_level = np.argsort(distances_per_level, axis=0)
            topk_idxs_per_level = topk_idxs_per_level[:selectable_k]
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = np.concatenate(candidate_idxs, axis=0)
        candidate_overlaps = overlaps[candidate_idxs]
        overlaps_mean = candidate_overlaps.mean(0)
        overlaps_std = candidate_overlaps.std(0)
        overlaps_thr = overlaps_mean + overlaps_std
        is_pos = candidate_overlaps >= overlaps_thr
        # print(is_pos)
        cx = grid_priors_cx[candidate_idxs]
        cy = grid_priors_cy[candidate_idxs]

        l_ = grid_priors_cx[candidate_idxs] - bbox[1]
        t_ = grid_priors_cy[candidate_idxs] - bbox[0]
        r_ = bbox[3] - grid_priors_cx[candidate_idxs]
        b_ = bbox[2] - grid_priors_cy[candidate_idxs]
        is_in_gts = np.min(np.stack([l_, t_, r_, b_], axis=0), axis=0) > 0.01
        # print(bbox)
        # print(is_in_gts)
        is_pos = is_pos & is_in_gts
        overlaps_inf = np.full_like(overlaps, -INF)
        index = candidate_idxs[is_pos]
        overlaps_inf[index] = overlaps[index]
        return overlaps_inf

    def bbox2distance(points, bbox, max_dis=None, eps=0.1):
        left = points[:, 1] - bbox[:, 1]
        top = points[:, 0] - bbox[:, 0]
        right = bbox[:, 3] - points[:, 1]
        bottom = bbox[:, 2] - points[:, 0]
        if max_dis is not None:
            left = np.clip(left, 0, max_dis - eps)
            top = np.clip(top, 0, max_dis - eps)
            right = np.clip(right, 0, max_dis - eps)
            bottom = np.clip(bottom, 0, max_dis - eps)
        return np.stack([left, top, right, bottom], -1)

    #     boxes = np.array([[50, 50, 60, 60, 2], [100, 100, 110, 115, 7], [30, 30, 80, 80, 1], [110, 80, 200, 200, 10],
    #                       [60, 70, 150, 150, 4]], dtype=np.float32)

    labels = []
    overlaps_inf_list = []
    INF = 100000

    res_boxes = np.zeros((config.num_nanodet_boxes, 4), dtype=np.float32)
    res_corners = np.zeros((config.num_nanodet_boxes, 4), dtype=np.float32)
    res_labels = np.full((config.num_nanodet_boxes), -1, dtype=np.int64)
    res_center_priors = np.zeros((config.num_nanodet_boxes, 4), dtype=np.float32)

    for bbox in boxes:
        label = int(bbox[4])
        labels.append(label)
        overlaps_inf = atssAssign(bbox)
        overlaps_inf_list.append(overlaps_inf)

    gt_labels = np.array(labels, dtype=np.int32)
    assigned_labels = np.full((config.num_nanodet_boxes,), -1, dtype=np.int32)
    overlaps_inf = np.stack(overlaps_inf_list, axis=0).T
    max_overlaps = np.max(overlaps_inf, axis=1)
    argmax_overlaps = np.argmax(overlaps_inf, axis=1)
    assigned_labels[max_overlaps != -INF] = gt_labels[argmax_overlaps[max_overlaps != -INF]]
    temp = gt_labels[argmax_overlaps[max_overlaps != -INF]]
    pos_inds = np.nonzero(assigned_labels != -1)[0]
    neg_inds = np.nonzero(assigned_labels == -1)[0]
    # print(max_overlaps[pos_inds])
    if len(boxes.shape) < 2:
        boxes = boxes.reshape(-1, 5)

    res_boxes[pos_inds] = boxes[argmax_overlaps[max_overlaps != -INF], :4]
    pos_bbox = res_boxes[pos_inds]
    pos_grid_priors_center = center_priors[pos_inds][..., :2]

    div = center_priors[pos_inds, None, 2]
    target_corners = bbox2distance(pos_grid_priors_center,
                                   pos_bbox) / center_priors[pos_inds, None, 2]
    target_corners = target_corners.clip(min=0, max=7 - 0.1)
    # print(target_corners)
    res_labels[pos_inds] = assigned_labels[pos_inds]
    res_corners[pos_inds] = target_corners
    if len(pos_inds) == 0:
        print(pos_inds)
        print("有0存在！！！")
    num_match = np.array([len(pos_inds)], dtype=np.int32)
    res_center_priors = center_priors
    return res_boxes, res_labels, res_center_priors, num_match

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


# if __name__ == "__main__":
#     def nanodet_bboxes_encode(boxes):
#         def bbox_overlaps(bbox):
#             ymin = np.maximum(y1, bbox[0])
#             xmin = np.maximum(x1, bbox[1])
#             ymax = np.minimum(y2, bbox[2])
#             xmax = np.minimum(x2, bbox[3])
#             # 并行化运算
#             w = np.maximum(xmax - xmin, 0.)
#             h = np.maximum(ymax - ymin, 0.)
#
#             inter_vol = h * w
#             union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
#             iou = inter_vol / union_vol
#             return np.squeeze(iou)
#
#
#         def atssAssign(bbox):
#             INF = 100000
#             num_grid_priors = center_priors_ltrb.shape[0]
#             grid_priors = center_priors_ltrb
#             overlaps = bbox_overlaps(bbox)
#
#             gt_cy = (bbox[0] + bbox[2]) / 2.0
#             gt_cx = (bbox[1] + bbox[3]) / 2.0
#             gt_points = np.stack((gt_cy, gt_cx), axis=0)
#
#             grid_priors_cy = (grid_priors[:, 0] + grid_priors[:, 2]) / 2.0
#             grid_priors_cx = (grid_priors[:, 1] + grid_priors[:, 3]) / 2.0
#             grid_cells_points = np.stack((grid_priors_cy, grid_priors_cx), axis=1)
#
#             distances = grid_cells_points - gt_points
#             distances = np.power(distances, 2)
#             distances = np.sum(distances, axis=-1)
#             distances = np.sqrt(distances)
#
#             candidate_idxs = []
#             start_idx = 0
#             topk = 9
#             for level, cells_per_level in enumerate(num_level_cells_list):
#                 end_idx = start_idx + cells_per_level
#                 distances_per_level = distances[start_idx:end_idx]
#                 selectable_k = min(config.topk, cells_per_level)
#                 topk_idxs_per_level = np.argsort(distances_per_level, axis=0)
#                 topk_idxs_per_level = topk_idxs_per_level[:selectable_k]
#                 candidate_idxs.append(topk_idxs_per_level + start_idx)
#                 start_idx = end_idx
#             candidate_idxs = np.concatenate(candidate_idxs, axis=0)
#             candidate_overlaps = overlaps[candidate_idxs]
#             overlaps_mean = candidate_overlaps.mean(0)
#             overlaps_std = candidate_overlaps.std(0)
#             overlaps_thr = overlaps_mean + overlaps_std
#             is_pos = candidate_overlaps >= overlaps_thr
#             cx = grid_priors_cx[candidate_idxs]
#             cy = grid_priors_cy[candidate_idxs]
#
#
#             l_ = grid_priors_cx[candidate_idxs] - bbox[1]
#             t_ = grid_priors_cy[candidate_idxs] - bbox[0]
#             r_ = bbox[3] - grid_priors_cx[candidate_idxs]
#             b_ = bbox[2] - grid_priors_cy[candidate_idxs]
#             is_in_gts = np.min(np.stack([l_, t_, r_, b_], axis=0), axis=0) > 0.01
#             is_pos = is_pos & is_in_gts
#
#             overlaps_inf = np.full_like(overlaps, -INF)
#             index = candidate_idxs[is_pos]
#             overlaps_inf[index] = overlaps[index]
#
#             return overlaps_inf
#
#         def bbox2distance(points, bbox, max_dis=None, eps=0.1):
#             left = points[:, 1] - bbox[:, 1]
#             top = points[:, 0] - bbox[:, 0]
#             right = bbox[:, 3] - points[:, 1]
#             bottom = bbox[:, 2] - points[:, 0]
#             if max_dis is not None:
#                 left = np.clip(left, 0, max_dis - eps)
#                 top = np.clip(top, 0, max_dis - eps)
#                 right = np.clip(right, 0, max_dis - eps)
#                 bottom = np.clip(bottom, 0, max_dis - eps)
#             return np.stack([left, top, right, bottom], -1)
#
#         boxes = np.array([[50,50,60,60,2], [100,100,110,115,7], [30,30,80,80,1],[110,80,200,200,10],[60,70,150,150,4]], dtype=np.float32)
#         labels = []
#         overlaps_inf_list = []
#         INF = 100000
#         res_boxes = np.zeros((config.num_nanodet_boxes, 4), dtype=np.float32)
#         res_corners = np.zeros((config.num_nanodet_boxes, 4), dtype=np.float32)
#         res_labels = np.full((config.num_nanodet_boxes), -1, dtype=np.int64)
#         res_center_priors = np.zeros((config.num_nanodet_boxes, 4), dtype=np.float32)
#
#         for bbox in boxes:
#             label = int(bbox[4])
#             labels.append(label)
#             overlaps_inf = atssAssign(bbox)
#             overlaps_inf_list.append(overlaps_inf)
#
#         gt_labels = np.array(labels, dtype=np.int32)
#         assigned_labels = np.full((config.num_nanodet_boxes,), -1, dtype=np.int32)
#         overlaps_inf = np.stack(overlaps_inf_list, axis=0).T
#         max_overlaps = np.max(overlaps_inf, axis=1)
#         argmax_overlaps = np.argmax(overlaps_inf, axis=1)
#         assigned_labels[max_overlaps != -INF] = gt_labels[argmax_overlaps[max_overlaps != -INF]]
#         temp = gt_labels[argmax_overlaps[max_overlaps != -INF]]
#         pos_inds = np.nonzero(assigned_labels != -1)[0]
#         neg_inds = np.nonzero(assigned_labels == -1)[0]
#
#         if len(boxes.shape) < 2:
#             boxes = boxes.reshape(-1, 5)
#
#         res_boxes[pos_inds] = boxes[argmax_overlaps[max_overlaps != -INF], :4]
#         pos_bbox = res_boxes[pos_inds]
#         pos_grid_priors_center = center_priors[pos_inds][..., :2]
#
#         div = center_priors[pos_inds, None, 2]
#         target_corners = bbox2distance(pos_grid_priors_center,
#                                        pos_bbox) / center_priors[pos_inds, None, 2]
#         target_corners = target_corners.clip(min=0, max=7 - 0.1)
#         res_labels[pos_inds] = assigned_labels[pos_inds]
#         res_corners[pos_inds] = target_corners
#         num_match = np.array([len(pos_inds)], dtype=np.int32)
#         res_center_priors = center_priors
#         return res_boxes, res_labels, res_center_priors, num_match
#
# # nanodet_bboxes_encode(None)
#     print("1")
