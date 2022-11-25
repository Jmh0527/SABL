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
"""Coco metrics utils"""

import os
import json
import numpy as np
from .model_utils.config import config

def apply_softnms(dets, scores, sigma=0.5, method=2, thresh=0.001, Nt=0.1):
    '''
    the soft nms implement using python
    :param dets: the pred_bboxes
    :param method: the policy of decay pred_bbox score in soft nms
    :param thresh: the threshold
    :param Nt: Nt
    :return: the index of pred_bbox after soft nms
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1.) * (x2 - x1 + 1.)
    orders = scores.argsort()[::-1]
    keep = []

    while orders.size > 0:

        i = orders[0]
        keep.append(i)

        for j in orders[1:]:

            xx1 = np.maximum(x1[i], x1[j])
            yy1 = np.maximum(y1[i], y1[j])
            xx2 = np.minimum(x2[i], x2[j])
            yy2 = np.minimum(y2[i], y2[j])
            w = np.maximum(xx2 - xx1 + 1., 0.)
            h = np.maximum(yy2 - yy1 + 1., 0.)

            inter = w * h
            overlap = inter / (areas[i] + areas[j] - inter)

            if method == 1:  # linear

                if overlap > Nt:

                    weight = 1 - overlap

                else:

                    weight = 1

            elif method == 2:  # gaussian

                weight = np.exp(-(overlap * overlap) / sigma)

            else:  # original NMS

                if overlap > Nt:

                    weight = 0

                else:

                    weight = 1

            scores[j] = weight * scores[j]

            if scores[j] < thresh:
                orders = np.delete(orders, np.where(orders == j))

        orders = np.delete(orders, 0)

    return keep


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep


def metrics(pred_data):
    """Calculate mAP of predicted bboxes."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    num_classes = config.num_classes # 80

    coco_root = config.coco_root # '/home/d1/jmh22/coco_dataset'
    data_type = config.val_data_type # 'val2017'

    # Classes need to train or test.
    val_cls = config.coco_classes
    val_cls_dict = {}
    for i, cls in enumerate(val_cls):
        val_cls_dict[i] = cls # 序号和类别对应 如{0: 'background', 1: 'person', 2: 'bicycle',...}

    anno_json = os.path.join(coco_root, config.instances_set.format(data_type)) # '/home/d1/jmh22/coco_dataset/annotations/instances_val2017.json'
    coco_gt = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco_gt.loadCats(coco_gt.getCatIds()) # id和类别名的对应
    for cat in cat_ids:
        classs_dict[cat["name"]] = cat["id"]

    predictions = []
    img_ids = []
    id = 0
    import pdb 
    pdb.set_trace()
    for sample in pred_data: # sample.keys():dict_keys(['boxes', 'box_scores', 'labels', 'img_id', 'image_shape'])
        print(f'{id}/{len(pred_data)}')
        pred_boxes = sample['boxes'] # pred_boxes.shape:(1000, 4)
        box_scores = sample['box_scores'] # box_scores.shape:(1000, )
        labels = sample['labels'] # (1000,)
        img_id = sample['img_id']
        h = config.feature_size[0] * 8
        w = config.feature_size[0] * 8

        final_boxes = []
        final_label = []
        final_score = []
        img_ids.append(img_id)

        temp = np.zeros((box_scores.shape[0], 80))
        for i, v in enumerate(labels):
            temp[i][v] = box_scores[i]  # 把scores改掉 其他位置得分都是0 就label预测到的位置有得分 v是<class 'numpy.float16'> 我在输入的地方改成了int16
        box_scores = temp.copy()
        for c in range(0, num_classes):
            class_box_scores = box_scores[:, c]
            score_mask = class_box_scores > 0 # config.min_score:0.05 这一步已经在解码的时候完成了 只取了>0.05的前1000个框 但是又进行了rescoring 所以可能小于0.05 这里就不做筛选了
            class_box_scores = class_box_scores[score_mask] # score_mask.shape (7555, )
            class_boxes = pred_boxes[score_mask] * [h, w, h, w]

            if score_mask.any():
                nms_index = apply_nms(class_boxes, class_box_scores, config.nms_thershold, config.max_boxes) # nms_thershold:0.5 max_boxes:100
                # apply_softnms( dets, scores,method=2, thresh=0.001, Nt=0.1, sigma=0.5 )
                # nms_index = apply_softnms(class_boxes, class_box_scores, config.softnms_sigma)
                class_boxes = class_boxes[nms_index]
                class_box_scores = class_box_scores[nms_index]

                final_boxes += class_boxes.tolist()
                final_score += class_box_scores.tolist()
                final_label += [classs_dict[val_cls_dict[c]]] * len(class_box_scores)
        # 在最后只保留100个框 不过感觉得分有点低 而且label全部都是1 可能训练不充分？
        max_boxes_filter = np.array(final_score).argsort()[::-1][:config.max_boxes]
        final_boxes = np.array(final_boxes)[max_boxes_filter].tolist()
        final_score = np.array(final_score)[max_boxes_filter].tolist()
        final_label = np.array(final_label)[max_boxes_filter].tolist()
        for loc, label, score in zip(final_boxes, final_label, final_score):
            res = {}
            res['image_id'] = img_id
            res['bbox'] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]] # 预测的bbox应该是(yc xc h w)
            res['score'] = score
            res['category_id'] = label
            predictions.append(res)
        id += 1
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes('predictions.json')
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats[0]
