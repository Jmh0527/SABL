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
from .model_utils.config import config
import mindspore.nn as nn
from mindspore import ops
import mindspore as ms

from mindspore import Tensor
from mindspore.ops import operations as P

class GeneratDefaultBoxes():
    """
    Generate Default boxes for retinanet, follows the order of (W, H, archor_sizes).
    `self.default_boxes` has a shape of [archor_sizes, H, W, 4], the last dimension is [y_c, x_c, h, w].
    `self.default_boxes_ltrb` has a shape as `self.default_boxes`, the last dimension is [y1, x1, y2, x2].
    """
# yolo: [xmid, ymid, w, h]，归一化到0-1 voc: [x1, y1, x2, y2] coco: [xmin, ymin, w, h]
    def __init__(self):
        fk = config.img_shape[0] / np.array(config.steps) # 1056 / [8, 16, 32, 64, 128] = [75, 37.5, 18.75, 9.375, 4.6875]
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        anchor_size = np.array(config.anchor_size) # [32, 64, 128, 256, 512] 对滴 也要step * 4
        self.default_boxes = []
        for idex, feature_size in enumerate(config.feature_size):# 按特征图操作的而不是fk [132, 66, 33, 17, 9] 
            base_size = anchor_size[idex] / config.img_shape[0] # 归一化 隐性要求是img_shape[0] == img_shape[1]
            size1 = base_size * scales[0] # 三个面积尺度
            size2 = base_size * scales[1]
            size3 = base_size * scales[2]
            all_sizes = []
            for aspect_ratio in config.aspect_ratios[idex]: # 每个面积尺度有三个ratios:[0.5, 1.0, 2.0]
                w1, h1 = size1 * math.sqrt(aspect_ratio), size1 / math.sqrt(aspect_ratio)
                all_sizes.append((h1, w1))
                w2, h2 = size2 * math.sqrt(aspect_ratio), size2 / math.sqrt(aspect_ratio)
                all_sizes.append((h2, w2))
                w3, h3 = size3 * math.sqrt(aspect_ratio), size3 / math.sqrt(aspect_ratio)
                all_sizes.append((h3, w3))

            for i, j in it.product(range(feature_size), repeat=2): # range(5),repeat=2 :(0,0)(0,1)...(4,4)
                for h, w in all_sizes:
                    cx, cy = j / fk[idex], i / fk[idex]
                    self.default_boxes.append([cy, cx, h, w]) # 迭代五次 最后是75*75+37*37+18*18+9*9+4*4

        def to_ltrb(cy, cx, h, w):
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

        # For IoU calculation
        self.default_boxes_ltrb = np.array(tuple(to_ltrb(*i) for i in self.default_boxes), dtype='float32') # y_min x_min y_max x_max 为了计算IoU
        self.default_boxes = np.array(self.default_boxes, dtype='float32')


class GeneratSquareBoxes():
    """生成方框"""
    def __init__(self):
        fk = config.img_shape[0] / np.array(config.steps) # 1056 / [8, 16, 32, 64, 128] = [132, 66, 33, 17, 9]
        anchor_size = np.array(4 * np.array(config.steps)) 
        self.default_square_boxes = []
        for idex, feature_size in enumerate(config.feature_size):# 按特征图操作的 
            base_size = anchor_size[idex] / config.img_shape[0] # 归一化 隐性要求是img_shape[0] == img_shape[1]
            all_sizes = []
            all_sizes.append((base_size, base_size))

            for i, j in it.product(range(feature_size), repeat=2): # range(5),repeat=2 :(0,0)(0,1)...(4,4)
                for h, w in all_sizes:
                    cx, cy = j / fk[idex], i / fk[idex]
                    self.default_square_boxes.append([cy, cx, h, w]) # len(self.default_square_boxes)=7555 迭代五次 最后是75*75+38*38+19*19+10*10+5*5=7555
    
        def to_ltrb(cy, cx, h, w):
                return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2
        
        # For IoU calculation
        self.default_square_boxes_ltrb = np.array(tuple(to_ltrb(*i) for i in self.default_square_boxes), dtype='float32') # y_min x_min y_max x_max 为了计算IoU
        self.default_square_boxes = np.array(self.default_square_boxes, dtype='float32') # (y_c, x_c, h, w)

default_square_boxes_ltrb = GeneratSquareBoxes().default_square_boxes_ltrb # default_square_boxes_ltrb.shape:(7555, 4)
default_square_boxes = GeneratSquareBoxes().default_square_boxes

default_boxes_ltrb = GeneratDefaultBoxes().default_boxes_ltrb # y_min x_min y_max x_max的所有框框
default_boxes = GeneratDefaultBoxes().default_boxes # cy, cx, h, w
y1, x1, y2, x2 = np.split(default_boxes_ltrb[:, :4], 4, axis=-1) # 这几行代码是全局变量
vol_anchors = (x2 - x1) * (y2 - y1) # 预设框的面积
matching_threshold = config.match_thershold # match_thershold: 0.5

def generate_buckets(proposals, num_buckets, scale_factor=1.0):
    """Generate buckets w.r.t bucket number and scale factor of proposals.

    Args:
        proposals (Tensor): Shape (n, 4)
        num_buckets (int): Number of buckets.
        scale_factor (float): Scale factor to rescale proposals.

    Returns:
        tuple[Tensor]: (bucket_w, bucket_h, l_buckets, r_buckets,
         t_buckets, d_buckets)

            - bucket_w: Width of buckets on x-axis. Shape (n, ).
            - bucket_h: Height of buckets on y-axis. Shape (n, ).
            - l_buckets: Left buckets. Shape (n, ceil(side_num/2)).
            - r_buckets: Right buckets. Shape (n, ceil(side_num/2)).
            - t_buckets: Top buckets. Shape (n, ceil(side_num/2)).
            - d_buckets: Down buckets. Shape (n, ceil(side_num/2)).
    """
    proposals = bbox_rescale(proposals, scale_factor) # scale_factor:3 把proosals放大了3倍 先变为xc yc w h 放大三倍后再放回来x1 y1 x2 y2
    # import pdb
    # pdb.set_trace()
    # number of buckets in each side
    side_num = int(np.ceil(num_buckets / 2.0)) # 7
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    px1 = proposals[..., 0]
    py1 = proposals[..., 1]
    px2 = proposals[..., 2]
    py2 = proposals[..., 3]

    bucket_w = pw / num_buckets
    bucket_h = ph / num_buckets

    # left buckets
    l_buckets = px1[:, None] + (0.5 + np.arange(
        0, side_num))[None, :] * bucket_w[:, None]
    # right buckets
    r_buckets = px2[:, None] - (0.5 + np.arange(
        0, side_num))[None, :] * bucket_w[:, None]
    # top buckets
    t_buckets = py1[:, None] + (0.5 + np.arange(
        0, side_num))[None, :] * bucket_h[:, None]
    # down buckets
    d_buckets = py2[:, None] - (0.5 + np.arange(
        0, side_num))[None, :] * bucket_h[:, None]
    return bucket_w, bucket_h, l_buckets, r_buckets, t_buckets, d_buckets

def bbox2bucket(proposals,
                gt,
                num_buckets,
                scale_factor,
                offset_topk=2,
                offset_upperbound=1.0,
                cls_ignore_neighbor=True):
    """Generate buckets estimation and fine regression targets.

    Args:
        proposals (Tensor): Shape (n, 4)
        gt (Tensor): Shape (n, 4)
        num_buckets (int): Number of buckets.
        scale_factor (float): Scale factor to rescale proposals. 3.0
        offset_topk (int): Topk buckets are used to generate
             bucket fine regression targets. Defaults to 2.
        offset_upperbound (float): Offset allowance to generate
             bucket fine regression targets.
             To avoid too large offset displacements. Defaults to 1.0.
        cls_ignore_neighbor (bool): Ignore second nearest bucket or Not.
             Defaults to True.

    Returns:
        tuple[Tensor]: (offsets, offsets_weights, bucket_labels, cls_weights).

            - offsets: Fine regression targets. \
                Shape (n, num_buckets*2).
            - offsets_weights: Fine regression weights. \
                Shape (n, num_buckets*2).
            - bucket_labels: Bucketing estimation labels. \
                Shape (n, num_buckets*2).
            - cls_weights: Bucketing estimation weights. \
                Shape (n, num_buckets*2).
    """
    # import pdb
    # pdb.set_trace()
    # generate buckets
    # proposals[0]: tensor([536., 296., 568., 328.])
    # bbox_rescale(proposals, 3)[0]: tensor([504., 264., 600., 360.])
    # bucket_w[0]: tensor(6.8571) bucket_h[0]: tensor(6.8571)
    # l_buckets[0]: tensor([507.4286, 514.2857, 521.1429, 528.0000, 534.8571, 541.7143, 548.5714]
    # r_buckets[0]: tensor([596.5714, 589.7143, 582.8571, 576.0000, 569.1429, 562.2857, 555.4286]
    (bucket_w, bucket_h, l_buckets, r_buckets, t_buckets,
     d_buckets) = generate_buckets(proposals, num_buckets, scale_factor) # 生成了bucket的框

    gx1 = gt[..., 0]
    gy1 = gt[..., 1]
    gx2 = gt[..., 2]
    gy2 = gt[..., 3]

    # generate offset targets and weights
    # offsets from buckets to gts
    l_offsets = (l_buckets - gx1[:, None]) / bucket_w[:, None]
    r_offsets = (r_buckets - gx2[:, None]) / bucket_w[:, None]
    t_offsets = (t_buckets - gy1[:, None]) / bucket_h[:, None]
    d_offsets = (d_buckets - gy2[:, None]) / bucket_h[:, None]

    # select top-k nearest buckets
    l_topk, l_label = topk_(abs(l_offsets), 2, 1)
    r_topk, r_label = topk_(abs(r_offsets), 2, 1)
    t_topk, t_label = topk_(abs(t_offsets), 2, 1)
    d_topk, d_label = topk_(abs(d_offsets), 2, 1)

    offset_l_weights = np.zeros_like(l_offsets)
    offset_r_weights = np.zeros_like(r_offsets)
    offset_t_weights = np.zeros_like(t_offsets)
    offset_d_weights = np.zeros_like(d_offsets)
    inds = np.arange(0, proposals.shape[0])

    # generate offset weights of top-k nearest buckets 每个边7个bucket有2个1
    for k in range(offset_topk):
        if k >= 1:
            offset_l_weights[inds, l_label[:,
                                           k]] = (l_topk[:, k] <
                                                  offset_upperbound).astype(np.float32)
            offset_r_weights[inds, r_label[:,
                                           k]] = (r_topk[:, k] <
                                                  offset_upperbound).astype(np.float32)
            offset_t_weights[inds, t_label[:,
                                           k]] = (t_topk[:, k] <
                                                  offset_upperbound).astype(np.float32)
            offset_d_weights[inds, d_label[:,
                                           k]] = (d_topk[:, k] <
                                                  offset_upperbound).astype(np.float32)
        else:
            offset_l_weights[inds, l_label[:, k]] = 1.0
            offset_r_weights[inds, r_label[:, k]] = 1.0
            offset_t_weights[inds, t_label[:, k]] = 1.0
            offset_d_weights[inds, d_label[:, k]] = 1.0

    offsets = np.concatenate([l_offsets, r_offsets, t_offsets, d_offsets], -1)
    offsets_weights = np.concatenate([offset_l_weights, offset_r_weights, offset_t_weights, offset_d_weights],-1)

    # generate bucket labels and weight
    side_num = int(np.ceil(num_buckets / 2.0))
    labels = np.concatenate(
        [l_label[:, 0][:,None], r_label[:, 0][:,None], t_label[:, 0][:,None], d_label[:, 0][:,None]], -1)

    batch_size = labels.shape[0]
    bucket_labels = np.eye(side_num)[labels.reshape(-1)].reshape(batch_size,-1).astype(np.float32)
    bucket_cls_l_weights = (abs(l_offsets) < 1).astype(np.float32)
    bucket_cls_r_weights = (abs(r_offsets) < 1).astype(np.float32)
    bucket_cls_t_weights = (abs(t_offsets) < 1).astype(np.float32)
    bucket_cls_d_weights = (abs(d_offsets) < 1).astype(np.float32)
    bucket_cls_weights =  np.concatenate([
        bucket_cls_l_weights, bucket_cls_r_weights, bucket_cls_t_weights,
        bucket_cls_d_weights
    ], -1)
    # ignore second nearest buckets for cls if necessary
    if cls_ignore_neighbor:
        bucket_cls_weights = (~((bucket_cls_weights == 1) &
                                (bucket_labels == 0))).astype(np.float32)
    else:
        bucket_cls_weights[:] = 1.0
    return offsets, offsets_weights, bucket_labels, bucket_cls_weights

def retinanet_bboxes_encode(boxes):
    """
    Labels anchors with ground truth inputs.

    Args:
        boxes: ground truth with shape [N, 5], for each row, it stores [ymin, xmin, ymax, xmax, cls].

    Returns:
        gt_loc: location ground truth with shape [num_anchors, 4].
        gt_label: class ground truth with shape [num_anchors, 1].
        num_matched_boxes: number of positives in an image.
    """

    def jaccard_with_anchors(bbox):
        """Compute jaccard score a box and the anchors."""
        # Intersection bbox and volume.
        ymin = np.maximum(y1, bbox[0])
        xmin = np.maximum(x1, bbox[1])
        ymax = np.minimum(y2, bbox[2])
        xmax = np.minimum(x2, bbox[3])
        w = np.maximum(xmax - xmin, 0.)
        h = np.maximum(ymax - ymin, 0.)

        # Volumes.
        inter_vol = h * w                                                               # 相交面积
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol # 相并面积
        jaccard = inter_vol / union_vol
        return np.squeeze(jaccard)
    # import pdb 
    # pdb.set_trace()
    pre_scores = np.zeros((config.num_retinanet_boxes*9), dtype=np.float32) # num_retinanet_boxes*9: 67995
    t_boxes = np.zeros((config.num_retinanet_boxes*9, 4), dtype=np.float32)
    t_label = np.zeros((config.num_retinanet_boxes*9), dtype=np.int64)
    
    for bbox in boxes: # (y1,x1,y2,x2) boxes.shape (8,5) np.float32  bbox.shape (5,) np.float32 
        label = int(bbox[4]) # label:40
        scores = jaccard_with_anchors(bbox)
        idx = np.argmax(scores) # 和bboxIoU最大的必选，即使不大于0.5
        scores[idx] = 2.0
        mask = (scores > matching_threshold) # matching_threshold:0.5 mask.shape (67995,) numpy bool
        mask = mask & (scores > pre_scores) 
        pre_scores = np.maximum(pre_scores, scores * mask) # IoU>0.5的score是IoU，否则都是0
        t_label = mask * label + (1 - mask) * t_label # True的地方是label，False的地方是t_label
        for i in range(4):
            t_boxes[:, i] = mask * bbox[i] + (1 - mask) * t_boxes[:, i] 
            # True的地方是boxes坐标，False的地方是其他bbox对应出来的boxes坐标 剩下的都是0
            # t_boxes:(ymin,xmin,ymax,xmax)
    
    """判断方框所分配的GT"""
    pre_scores_for_bucket = pre_scores.reshape(-1, 9) # (7555, 9) 每9个框找到pre_scores最大的索引 再和index取交集 就是我们要找的方框索引
    index_for_bucket = pre_scores_for_bucket.argmax(-1) # index_for_bucket.shape:(7555,)
    for i, v in enumerate(index_for_bucket):
        index_for_bucket[i] = index_for_bucket[i] + 9 * i

    index = np.nonzero(t_label) # 找到有目标的样本 index是一个tuple  index[0].shape (66,)
    index_for_bucket = np.intersect1d(index_for_bucket, index) # index_for_bucket.shape:(21,)

    t_boxes_for_bucket = np.zeros((config.num_retinanet_boxes, 4), dtype=np.float32)
    t_boxes_for_bucket[index_for_bucket//9] = t_boxes[index_for_bucket] # 找到了方框正样本对应的GT坐标 (y1,x1,y2,x2)

    proposals = default_square_boxes_ltrb[index_for_bucket//9]
    gt = t_boxes_for_bucket[index_for_bucket//9]
    # import pdb
    # pdb.set_trace()
    offsets, offsets_weights, bucket_labels, bucket_cls_weights = bbox2bucket(proposals,gt,14,3)

    bbox = np.zeros((config.num_retinanet_boxes, 28))
    bbox_weights = np.zeros((config.num_retinanet_boxes, 28))
    buk_label = np.zeros((config.num_retinanet_boxes, 28))
    buk_label_weights = np.zeros((config.num_retinanet_boxes, 28))

    bbox[index_for_bucket//9] = offsets
    bbox_weights[index_for_bucket//9] = offsets_weights
    buk_label[index_for_bucket//9] = bucket_labels
    buk_label_weights[index_for_bucket//9] = bucket_cls_weights

    temp = np.zeros(config.num_retinanet_boxes)
    temp[index_for_bucket//9] = t_label[index_for_bucket]
    t_label = temp
    for i in range(pre_scores_for_bucket.shape[0]):
        if pre_scores_for_bucket[i].max() >= 0.4 and pre_scores_for_bucket[i].max() < 0.5: # 忽略iou在0.4-0.5之间的样本
            t_label[i//9] = -1
    t_label = t_label.astype(np.int32)
    num_match = np.array([len(np.nonzero(t_label)[0])], dtype=np.int32)
    # bboxes, labels, scores, confids = retinanetInferWithDecoder2(Tensor(default_square_boxes_ltrb).astype(ms.float32),Tensor(bbox).astype(ms.float32),Tensor(t_label),Tensor(buk_label)).retinanet_decode()
    # bboxes, labels, scores, confids = retinanetInferWithDecoder(Tensor(default_square_boxes_ltrb).astype(ms.float32),Tensor(bbox).astype(ms.float32),Tensor(t_label),Tensor(buk_label)).retina_decode()
    return bbox, bbox_weights, t_label, buk_label, buk_label_weights, num_match 

    """
    # bboxes (67995,4) 返回的是正样本的需要预测的回归参数 不是正样本的都是0
    # t_label.astype(np.int32) (67995,) int64 -> int32
    # num_match  array([66], dtype=int32)
    """
    
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

def bbox_rescale(bboxes, scale_factor=1.0):
    """Rescale bounding box w.r.t. scale_factor.

    Args:
        bboxes (Tensor): Shape (n, 4) for bboxes or (n, 5) for rois
        scale_factor (float): rescale factor

    Returns:
        Tensor: Rescaled bboxes.
    """

    bboxes_ = bboxes
    cx = (bboxes_[:, 0] + bboxes_[:, 2]) * 0.5
    cy = (bboxes_[:, 1] + bboxes_[:, 3]) * 0.5
    w = bboxes_[:, 2] - bboxes_[:, 0]
    h = bboxes_[:, 3] - bboxes_[:, 1]
    w = w * scale_factor
    h = h * scale_factor
    x1 = cx - 0.5 * w
    x2 = cx + 0.5 * w
    y1 = cy - 0.5 * h
    y2 = cy + 0.5 * h
    rescaled_bboxes = np.concatenate((x1[:,None], y1[:,None], x2[:,None], y2[:,None]), -1)
    return rescaled_bboxes

def topk_(matrix, K, axis=1):
    column_index = np.arange(matrix.shape[1 - axis])[:, None]
    topk_index = np.argpartition(matrix, K, axis=axis)[:, 0:K]
    topk_data = matrix[column_index, topk_index]
    topk_index_sort = np.argsort(topk_data, axis=axis)
    topk_data_sort = topk_data[column_index, topk_index_sort]
    topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort


class retinanetInferWithDecoder():
    """
    retinanet Infer wrapper to decode the bbox locations.

    Args:
        network (Cell): the origin retinanet infer network without bbox decoder.
        default_boxes (Tensor): the default_boxes from anchor generator
        config (dict): retinanet config
    Returns:
        Tensor, the locations for bbox after decoder representing (y0,x0,y1,x1)
        Tensor, the prediction labels.

    """
    def __init__(self, default_boxes, bbox, t_label, buk_label):
        super(retinanetInferWithDecoder, self).__init__()
        self.default_boxes = default_boxes # 这里传入的default_boxes是加上bucket的方框(-1,32)
        self.bbox_reg_pred, self.scores, self.bbox_cls_pred = bbox, t_label, buk_label
        from mindspore.ops import operations as P
        import mindspore.common.dtype as mstype
        self.scores = P.OneHot()(self.scores, 81, Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32))
        self.scores = self.scores[:,1:]

    
    def _bbox_post_process(self,
                        mlvl_scores,
                        mlvl_labels,
                        mlvl_bboxes,
                        scale_factor,
                        cfg,
                        rescale=False,
                        with_nms=True,
                        mlvl_score_factors=None,
                        **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None: # mlvl_score_factors[0].shape [1000] 是个列表一共有5个元素 这里主要是把图片和bbox缩放回原来的尺寸
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels
    
    def filter_scores_and_topk(self, scores, score_thr=0.05, topk=1000, results=None):
        """Filter results using score threshold and topk candidates.

        Args:
            scores (Tensor): The scores, shape (num_bboxes, K).
            score_thr (float): The score filter threshold.
            topk (int): The number of topk candidates.
            results (dict or list or Tensor, Optional): The results to
            which the filtering rule is to be applied. The shape
            of each item is (num_bboxes, N).

        Returns:
            tuple: Filtered results

                - scores (Tensor): The scores after being filtered, \
                    shape (num_bboxes_filtered, ).
                - labels (Tensor): The class labels, shape \
                    (num_bboxes_filtered, ).
                - anchor_idxs (Tensor): The anchor indexes, shape \
                    (num_bboxes_filtered, ).
                - filtered_results (dict or list or Tensor, Optional): \
                    The filtered results. The shape of each item is \
                    (num_bboxes_filtered, N).
        """
        # scores torch.Size([15200, 80])
        # score_thr 0.05
        # topk 1000 这里取了前1000个框剩下的都不会进行decode
        # import pdb 
        # pdb.set_trace()
        valid_mask = scores > 0.5 # score_thr=0.05 # valid_mask.shape [15200,80]
        scores = Tensor(scores.asnumpy()[valid_mask.asnumpy()])
        # scores = scores[valid_mask] # 把大于0.05的scores选出来 [9582]
        temp = valid_mask.asnumpy().nonzero()
        valid_idx1 = Tensor(temp[0])[:,None]
        valid_idx2 = Tensor(temp[1])[:,None]
        valid_idxs = ops.Concat(axis=1)((valid_idx1,valid_idx2))
        # valid_idxs = torch.nonzero(valid_mask) # [9582,2] 15200里面的第几个框和该框是第几类

        num_topk = min(topk, valid_idxs.shape[0]) # 1000
        # torch.sort is actually faster than .topk (at least on GPUs)
        # scores, idxs = scores.sort(descending=True) # 把9582个分数排序 idxs是对应分数所在的位置 是9582中的第几个
        scores, idxs = ops.Sort(descending=True)(scores)
        scores = scores[:num_topk] # 选出来前1000个得分最高的分数
        topk_idxs = valid_idxs[idxs[:num_topk]] # 索引出来是第几类最大 0-79
        keep_idxs, labels = ops.Unstack(axis=1)(topk_idxs)
        # keep_idxs, labels = topk_idxs.unbind(dim=1) # labels 是0-79 keep_idxs是保留的哪些位置的scores

        filtered_results = None
        if results is not None:
            if isinstance(results, dict):
                filtered_results = {k: v[keep_idxs] for k, v in results.items()}
            elif isinstance(results, list):
                filtered_results = [result[keep_idxs] for result in results]
            elif isinstance(results, torch.Tensor):
                filtered_results = results[keep_idxs]
            else:
                raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                        f'but get {type(results)}.')
        # socres [1000] labels [1000] 最后返回的scores是前1000个框的scores labels不是独热编码 是搞好的label: 0-79
        return scores, labels, keep_idxs, filtered_results

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        """Apply transformation `pred_bboxes` to `boxes`.
        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Predictions for bucketing estimation
                and fine regression
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        # import pdb 
        # pdb.set_trace()
        assert len(pred_bboxes) == 2
        cls_preds, offset_preds = pred_bboxes
        assert cls_preds.shape[0] == bboxes.shape[0] and offset_preds.shape[0] == bboxes.shape[0]
        decoded_bboxes = self.bucket2bbox(bboxes, cls_preds, offset_preds,
                                     14, 3.0,
                                     max_shape, True)

        return decoded_bboxes
    
    def bbox_rescale(self, bboxes, scale_factor=1.0):
        """Rescale bounding box w.r.t. scale_factor.

        Args:
            bboxes (Tensor): Shape (n, 4) for bboxes or (n, 5) for rois
            scale_factor (float): rescale factor

        Returns:
            Tensor: Rescaled bboxes.
        """
        bboxes_ = bboxes
        cx = (bboxes_[:, 0] + bboxes_[:, 2]) * 0.5
        cy = (bboxes_[:, 1] + bboxes_[:, 3]) * 0.5
        w = bboxes_[:, 2] - bboxes_[:, 0]
        h = bboxes_[:, 3] - bboxes_[:, 1]
        w = w * scale_factor
        h = h * scale_factor
        x1 = cx - 0.5 * w
        x2 = cx + 0.5 * w
        y1 = cy - 0.5 * h
        y2 = cy + 0.5 * h
        rescaled_bboxes = ops.Stack(axis=-1)((x1, y1, x2, y2))
        return rescaled_bboxes

    def bucket2bbox(self,
                    proposals,
                    cls_preds,
                    offset_preds,
                    num_buckets,
                    scale_factor=1.0,
                    max_shape=None,
                    clip_border=True):
        """Apply bucketing estimation (cls preds) and fine regression (offset
        preds) to generate det bboxes.

        Args:
            proposals (Tensor): Boxes to be transformed. Shape (n, 4)
            cls_preds (Tensor): bucketing estimation. Shape (n, num_buckets*2).
            offset_preds (Tensor): fine regression. Shape (n, num_buckets*2).
            num_buckets (int): Number of buckets.
            scale_factor (float): Scale factor to rescale proposals.
            max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
            clip_border (bool, optional): Whether clip the objects outside the
                border of the image. Defaults to True.

        Returns:
            tuple[Tensor]: (bboxes, loc_confidence).

                - bboxes: predicted bboxes. Shape (n, 4)
                - loc_confidence: localization confidence of predicted bboxes.
                    Shape (n,).
        """
        # import pdb
        # pdb.set_trace()
        side_num = int(np.ceil(num_buckets / 2.0)) # 7
        cls_preds = cls_preds.view(-1, side_num).astype(ms.float32) # [4000,7] 4个边 每个边1000个 要转为float32
        offset_preds = offset_preds.view(-1, side_num).astype(ms.float32) # [4000,7]

        # scores = F.softmax(cls_preds, dim=1) # 做了softmax
        scores = ops.Softmax(-1)(cls_preds)
        # score_topk, score_label = scores.topk(2, dim=1, largest=True, sorted=True) # 维度都是[4000, 2]
        score_topk, score_label = ops.TopK(sorted=True)(scores, 2) # 沿最后一维度排序

        rescaled_proposals = self.bbox_rescale(proposals, scale_factor)

        pw = rescaled_proposals[..., 2] - rescaled_proposals[..., 0]
        ph = rescaled_proposals[..., 3] - rescaled_proposals[..., 1]
        px1 = rescaled_proposals[..., 0]
        py1 = rescaled_proposals[..., 1]
        px2 = rescaled_proposals[..., 2]
        py2 = rescaled_proposals[..., 3]

        bucket_w = pw / num_buckets
        bucket_h = ph / num_buckets

        score_inds_l = score_label[0::4, 0] # [0,4,8...]
        score_inds_r = score_label[1::4, 0] # [1,5,9...]
        score_inds_t = score_label[2::4, 0] # [2,6,10...]
        score_inds_d = score_label[3::4, 0] # [3,7,11...]
        l_buckets = px1 + (0.5 + score_inds_l.astype(ms.float32)) * bucket_w
        r_buckets = px2 - (0.5 + score_inds_r.astype(ms.float32)) * bucket_w
        t_buckets = py1 + (0.5 + score_inds_t.astype(ms.float32)) * bucket_h
        d_buckets = py2 - (0.5 + score_inds_d.astype(ms.float32)) * bucket_h

        offsets = offset_preds.view(-1, 4, side_num)
        inds = ms.numpy.arange(proposals.shape[0])
        # inds = torch.arange(proposals.size(0)).to(proposals).long()
        l_offsets = offsets[:, 0, :][inds, score_inds_l]
        r_offsets = offsets[:, 1, :][inds, score_inds_r]
        t_offsets = offsets[:, 2, :][inds, score_inds_t]
        d_offsets = offsets[:, 3, :][inds, score_inds_d]

        x1 = l_buckets - l_offsets * bucket_w
        x2 = r_buckets - r_offsets * bucket_w
        y1 = t_buckets - t_offsets * bucket_h
        y2 = d_buckets - d_offsets * bucket_h

        if clip_border and max_shape is not None:
            x1 = ops.clip_by_value(x1,clip_value_min=0, clip_value_max=max_shape[1] - 1)
            y1 = ops.clip_by_value(y1,clip_value_min=0, clip_value_max=max_shape[0] - 1)
            x2 = ops.clip_by_value(x2,clip_value_min=0, clip_value_max=max_shape[1] - 1)
            y2 = ops.clip_by_value(y2,clip_value_min=0, clip_value_max=max_shape[0] - 1)
            # x1 = x1.clamp(min=0, max=max_shape[1] - 1)
            # y1 = y1.clamp(min=0, max=max_shape[0] - 1)
            # x2 = x2.clamp(min=0, max=max_shape[1] - 1)
            # y2 = y2.clamp(min=0, max=max_shape[0] - 1)
        bboxes = ops.Concat(-1)([x1[:, None], y1[:, None], x2[:, None], y2[:, None]])
        # bboxes = torch.cat([x1[:, None], y1[:, None], x2[:, None], y2[:, None]],
        #                 dim=-1)
        # bboxes [1000,4]
        # tensor([[290.9958, 315.8408, 341.7504, 344.8170],
        #         [290.8769, 315.9640, 341.9824, 345.2071],
        #         [290.2606, 316.0638, 341.9099, 344.8912],
        #         ...,
        #         [615.1074, 216.7199, 675.5566, 282.3963],
        #         [651.0364, 158.5985, 692.8834, 200.2746],
        #         [ 55.1382, 672.6702, 182.0291, 696.9214]], device='cuda:2')

        # bucketing guided rescoring
        loc_confidence = score_topk[:, 0]
        top2_neighbor_inds = (score_label[:, 0] - score_label[:, 1]).abs() == 1
        loc_confidence += score_topk[:, 1] * top2_neighbor_inds.astype(ms.float32)
        loc_confidence = loc_confidence.view(-1, 4).mean(axis=1)

        return bboxes, loc_confidence
        # loc_confidence.max() tensor(0.9979, device='cuda:2')
        # loc_confidence.min() tensor(0.4733, device='cuda:2')

    def retina_decode(self):
        """construct"""
        # bbox_reg_pred, scores, bbox_cls_pred = self.network(x)
        # import pdb 
        # pdb.set_trace()
        bbox_reg_pred, scores, bbox_cls_pred = self.bbox_reg_pred, self.scores, self.bbox_cls_pred
        scores = ops.Sigmoid()(scores)
        results = self.filter_scores_and_topk(
                scores,
                results = 
                    dict(
                    anchors=self.default_boxes, 
                    bbox_cls_pred=bbox_cls_pred,
                    bbox_reg_pred=bbox_reg_pred)) 
        scores, labels, _, filtered_results = results # 这里的labels是直接数值 不是独热编码
        anchors = filtered_results['anchors']
        bbox_cls_pred = filtered_results['bbox_cls_pred']
        bbox_reg_pred = filtered_results['bbox_reg_pred']
        bbox_preds = [
                bbox_cls_pred,
                bbox_reg_pred
            ]
        bboxes, confids = self.decode(
                anchors, bbox_preds, max_shape=(2,2,3)) # 因为后面max_shape要减去1 作为clip的最大值 而我们这里预测的是归一化的 
        return bboxes, labels, scores, confids






class retinanetInferWithDecoder2():
    """
    retinanet Infer wrapper to decode the bbox locations.

    Args:
        network (Cell): the origin retinanet infer network without bbox decoder.
        default_boxes (Tensor): the default_boxes from anchor generator
        config (dict): retinanet config
    Returns:
        Tensor, the locations for bbox after decoder representing (y0,x0,y1,x1)
        Tensor, the prediction labels.

    """
    def __init__(self, default_boxes, bbox, t_label, buk_label):
        super(retinanetInferWithDecoder2, self).__init__()
        self.default_boxes = default_boxes # 这里传入的default_boxes是加上bucket的方框(-1,32)
        self.bbox_reg_pred, self.scores, self.bbox_cls_pred = bbox, t_label, buk_label
        from mindspore.ops import operations as P
        import mindspore.common.dtype as mstype
        self.scores = P.OneHot()(self.scores, 81, Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32))
        self.scores = self.scores[:,1:]
    
    def filter_scores_and_topk(self, scores, score_thr=0.05, topk=1000, results=None):
        # 注意他是5层特征图每层最多1000个，不是5层一共最多1000个
        """Filter results using score threshold and topk candidates.

        Args:
            scores (Tensor): The scores, shape (num_bboxes, K).
            score_thr (float): The score filter threshold.
            topk (int): The number of topk candidates.
            results (dict or list or Tensor, Optional): The results to
            which the filtering rule is to be applied. The shape
            of each item is (num_bboxes, N).

        Returns:
            tuple: Filtered results

                - scores (Tensor): The scores after being filtered, \
                    shape (num_bboxes_filtered, ).
                - labels (Tensor): The class labels, shape \
                    (num_bboxes_filtered, ).
                - anchor_idxs (Tensor): The anchor indexes, shape \
                    (num_bboxes_filtered, ).
                - filtered_results (dict or list or Tensor, Optional): \
                    The filtered results. The shape of each item is \
                    (num_bboxes_filtered, N).
        """
        # scores torch.Size([15200, 80])
        # score_thr 0.05
        # topk 1000 这里取了前1000个框剩下的都不会进行decode 
        # import pdb 
        # pdb.set_trace()
        valid_mask = scores > score_thr # score_thr=0.05 # valid_mask.shape [15200,80]
        if not valid_mask.any():
            return 0
        scores = Tensor(scores.asnumpy()[valid_mask.asnumpy()]) # 把大于0.05的scores选出来 [9582]
        # valid_idxs = torch.nonzero(valid_mask) # [9582,2] 15200里面的第几个框和该框是第几类
        temp = valid_mask.asnumpy().nonzero()
        valid_idx1 = Tensor(temp[0])[:,None]
        valid_idx2 = Tensor(temp[1])[:,None]
        valid_idxs = ops.concat((valid_idx1,valid_idx2), 1)

        num_topk = min(topk, valid_idxs.shape[0]) # 1000
        # torch.sort is actually faster than .topk (at least on GPUs)
        scores, idxs = ops.Sort(descending=True)(scores) # 把9582个分数排序 idxs是对应分数所在的位置 是9582中的第几个
        scores = scores[:num_topk] # 选出来前1000个得分最高的分数
        topk_idxs = valid_idxs[idxs[:num_topk]] # 索引出来是第几类最大 0-79
        keep_idxs, labels = ops.Unstack(axis=1)(topk_idxs) # labels 是0-79 keep_idxs是保留的哪些位置的scores

        filtered_results = None
        if results is not None:
            if isinstance(results, dict):
                filtered_results = {k: v[keep_idxs] for k, v in results.items()}
            elif isinstance(results, list):
                filtered_results = [result[keep_idxs] for result in results]
            elif isinstance(results, torch.Tensor):
                filtered_results = results[keep_idxs]
            else:
                raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                        f'but get {type(results)}.')
        # socres [1000] labels [1000] 最后返回的scores是前1000个框的scores labels不是独热编码 是搞好的label: 0-79
        return scores, labels, keep_idxs, filtered_results

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        """Apply transformation `pred_bboxes` to `boxes`.
        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_bboxes (torch.Tensor): Predictions for bucketing estimation
                and fine regression
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        # import pdb 
        # pdb.set_trace()
        assert len(pred_bboxes) == 2
        cls_preds, offset_preds = pred_bboxes
        assert cls_preds.shape[0] == bboxes.shape[0] and offset_preds.shape[0] == bboxes.shape[0]
        # import pdb 
        # pdb.set_trace()
        decoded_bboxes = self.bucket2bbox(bboxes, cls_preds, offset_preds,
                                     14, 3.0,
                                     max_shape, True)

        return decoded_bboxes
    
    def bbox_rescale(self, bboxes, scale_factor=1.0):
        """Rescale bounding box w.r.t. scale_factor.

        Args:
            bboxes (Tensor): Shape (n, 4) for bboxes or (n, 5) for rois
            scale_factor (float): rescale factor

        Returns:
            Tensor: Rescaled bboxes.
        """
        # import pdb 
        # pdb.set_trace()
        # bboxes_ = bboxes.asnumpy()
        bboxes_ = bboxes
        cx = (bboxes_[:, 0] + bboxes_[:, 2]) * 0.5
        cy = (bboxes_[:, 1] + bboxes_[:, 3]) * 0.5
        w = bboxes_[:, 2] - bboxes_[:, 0]
        h = bboxes_[:, 3] - bboxes_[:, 1]
        w = w * scale_factor
        h = h * scale_factor
        x1 = cx - 0.5 * w
        x2 = cx + 0.5 * w
        y1 = cy - 0.5 * h
        y2 = cy + 0.5 * h
        # rescaled_bboxes = np.concatenate((x1[:,None], y1[:,None], x2[:,None], y2[:,None]), -1)
        rescaled_bboxes = ops.Stack(axis=-1)((x1, y1, x2, y2))
        return rescaled_bboxes

    def bucket2bbox(self, 
                    proposals,
                    cls_preds,
                    offset_preds,
                    num_buckets,
                    scale_factor=1.0,
                    max_shape=None,
                    clip_border=True):
        """Apply bucketing estimation (cls preds) and fine regression (offset
        preds) to generate det bboxes.

        Args:
            proposals (Tensor): Boxes to be transformed. Shape (n, 4)
            cls_preds (Tensor): bucketing estimation. Shape (n, num_buckets*2).
            offset_preds (Tensor): fine regression. Shape (n, num_buckets*2).
            num_buckets (int): Number of buckets.
            scale_factor (float): Scale factor to rescale proposals.
            max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
            clip_border (bool, optional): Whether clip the objects outside the
                border of the image. Defaults to True.

        Returns:
            tuple[Tensor]: (bboxes, loc_confidence).

                - bboxes: predicted bboxes. Shape (n, 4)
                - loc_confidence: localization confidence of predicted bboxes.
                    Shape (n,).
        """
        # import pdb
        # pdb.set_trace()
        side_num = int(np.ceil(num_buckets / 2.0)) # 7
        cls_preds = cls_preds.view(-1, side_num).astype(ms.float32) # [4000,7] 4个边 每个边1000个
        offset_preds = offset_preds.view(-1, side_num).astype(ms.float32) # [4000,7]

        # scores = F.softmax(cls_preds, dim=1) # 做了softmax
        scores = ops.Softmax(-1)(cls_preds)
        # score_topk, score_label = scores.topk(2, dim=1, largest=True, sorted=True) # 维度都是[4000, 2]
        score_topk, score_label = ops.TopK(sorted=True)(scores, 2) # 沿最后一维度排序
        # import pdb 
        # pdb.set_trace()

        rescaled_proposals = self.bbox_rescale(proposals, scale_factor)

        pw = rescaled_proposals[..., 2] - rescaled_proposals[..., 0]
        ph = rescaled_proposals[..., 3] - rescaled_proposals[..., 1]
        px1 = rescaled_proposals[..., 0]
        py1 = rescaled_proposals[..., 1]
        px2 = rescaled_proposals[..., 2]
        py2 = rescaled_proposals[..., 3]

        bucket_w = pw / num_buckets
        bucket_h = ph / num_buckets

        score_inds_l = score_label[0::4, 0] # [0,4,8...]
        score_inds_r = score_label[1::4, 0] # [1,5,9...]
        score_inds_t = score_label[2::4, 0] # [2,6,10...]
        score_inds_d = score_label[3::4, 0] # [3,7,11...]
        l_buckets = px1 + (0.5 + score_inds_l.astype(ms.float32)) * bucket_w
        r_buckets = px2 - (0.5 + score_inds_r.astype(ms.float32)) * bucket_w
        t_buckets = py1 + (0.5 + score_inds_t.astype(ms.float32)) * bucket_h
        d_buckets = py2 - (0.5 + score_inds_d.astype(ms.float32)) * bucket_h

        offsets = offset_preds.view(-1, 4, side_num)
        # inds = torch.arange(proposals.size(0)).to(proposals).long()
        inds = ms.numpy.arange(proposals.shape[0])
        l_offsets = offsets[:, 0, :][inds, score_inds_l]
        r_offsets = offsets[:, 1, :][inds, score_inds_r]
        t_offsets = offsets[:, 2, :][inds, score_inds_t]
        d_offsets = offsets[:, 3, :][inds, score_inds_d]

        x1 = l_buckets - l_offsets * bucket_w
        x2 = r_buckets - r_offsets * bucket_w
        y1 = t_buckets - t_offsets * bucket_h
        y2 = d_buckets - d_offsets * bucket_h

        if clip_border and max_shape is not None:
            x1 = ops.clip_by_value(x1,clip_value_min=0, clip_value_max=max_shape[1] - 1)
            y1 = ops.clip_by_value(y1,clip_value_min=0, clip_value_max=max_shape[0] - 1)
            x2 = ops.clip_by_value(x2,clip_value_min=0, clip_value_max=max_shape[1] - 1)
            y2 = ops.clip_by_value(y2,clip_value_min=0, clip_value_max=max_shape[0] - 1)
            # x1 = x1.clamp(min=0, max=max_shape[1] - 1)
            # y1 = y1.clamp(min=0, max=max_shape[0] - 1)
            # x2 = x2.clamp(min=0, max=max_shape[1] - 1)
            # y2 = y2.clamp(min=0, max=max_shape[0] - 1)
        # bboxes = torch.cat([x1[:, None], y1[:, None], x2[:, None], y2[:, None]],
        #                 dim=-1)
        bboxes = ops.concat([x1[:, None], y1[:, None], x2[:, None], y2[:, None]], -1)

        # bboxes [1000,4]
        # tensor([[290.9958, 315.8408, 341.7504, 344.8170],
        #         [290.8769, 315.9640, 341.9824, 345.2071],
        #         [290.2606, 316.0638, 341.9099, 344.8912],
        #         ...,
        #         [615.1074, 216.7199, 675.5566, 282.3963],
        #         [651.0364, 158.5985, 692.8834, 200.2746],
        #         [ 55.1382, 672.6702, 182.0291, 696.9214]], device='cuda:2')

        # bucketing guided rescoring
        loc_confidence = score_topk[:, 0]
        top2_neighbor_inds = (score_label[:, 0] - score_label[:, 1]).abs() == 1
        loc_confidence += score_topk[:, 1] * top2_neighbor_inds.astype(ms.float32)
        loc_confidence = loc_confidence.view(-1, 4).mean(axis=1)

        return bboxes, loc_confidence
        # loc_confidence.max() tensor(0.9979, device='cuda:2')
        # loc_confidence.min() tensor(0.4733, device='cuda:2')

    def retinanet_decode(self):
        """construct"""
        bbox_reg_pred, scores, bbox_cls_pred = self.bbox_reg_pred, self.scores, self.bbox_cls_pred
        scores = ops.Sigmoid()(scores)

        idx = [0,17424,21780,22869,23158,23239]
        cat = ops.Concat(0)
        final_scores = Tensor([0.0])
        final_labels = Tensor([0]).astype(ms.int64)
        final_anchors = Tensor([[0.0]*4])
        final_bbox_cls_pred = Tensor([[0.0]*28])
        final_bbox_reg_pred = Tensor([[0.0]*28])
        for i in range(5): # 五张特征图 每个特征图要提取前最多1000个
            results = self.filter_scores_and_topk(
                    scores[idx[i]:idx[i+1],:],
                    results = 
                        dict(
                        anchors=self.default_boxes[idx[i]:idx[i+1],:], 
                        bbox_cls_pred=bbox_cls_pred[idx[i]:idx[i+1],:],
                        bbox_reg_pred=bbox_reg_pred[idx[i]:idx[i+1],:]))
            # import pdb 
            # pdb.set_trace()
            if results == 0:
                continue
            scores_level, labels_level, _, filtered_results_level = results # 这里的labels是直接数值 不是独热编码
            final_scores = cat((final_scores,scores_level))
            final_labels = cat((final_labels,labels_level))
            final_anchors = cat((final_anchors,filtered_results_level['anchors']))
            final_bbox_cls_pred = cat((final_bbox_cls_pred,filtered_results_level['bbox_cls_pred'].astype(ms.float32)))
            final_bbox_reg_pred = cat((final_bbox_reg_pred,filtered_results_level['bbox_reg_pred']))
        final_scores = final_scores[1:]
        final_labels = final_labels[1:]
        final_anchors = final_anchors[1:,:]
        final_bbox_cls_pred = final_bbox_cls_pred[1:,:]
        final_bbox_reg_pred = final_bbox_reg_pred[1:,:]
        bbox_preds = [
                final_bbox_cls_pred,
                final_bbox_reg_pred
            ]
        bboxes, confids = self.decode(
                final_anchors, bbox_preds, max_shape=(2,2,3)) # 因为后面max_shape要减去1 作为clip的最大值 max_shape (800, 1199, 3)
        return bboxes, final_labels, final_scores, confids


