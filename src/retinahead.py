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

"""retinanet based resnet."""

import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from .bottleneck import FPN

import numpy as np


class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.

    Args:
        config (dict): The default config of retinanet.

    Returns:
        Tensor, flatten predictions.
    """

    def __init__(self, config):
        super(FlattenConcat, self).__init__()
        self.num_retinanet_boxes = config.num_retinanet_boxes
        self.concat = P.Concat(axis=1)
        self.transpose = P.Transpose()

    def construct(self, inputs):
        output = ()
        batch_size = F.shape(inputs[0])[0]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (F.reshape(x, (batch_size, -1)),)
        res = self.concat(output)
        return F.reshape(res, (batch_size, self.num_retinanet_boxes, -1))


def ClassificationModel(in_channel, num_anchors, kernel_size=3, stride=1, pad_mod='same', num_classes=80,
                        feature_size=256):
    conv1 = nn.Conv2d(in_channel, feature_size, kernel_size=3, pad_mode='same')
    conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv5 = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, pad_mode='same')
    return nn.SequentialCell([conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU(), conv4, nn.ReLU(), conv5])


def RegressionModel(in_channel, num_anchors, kernel_size=3, stride=1, pad_mod='same', feature_size=256):
    conv1 = nn.Conv2d(in_channel, feature_size, kernel_size=3, pad_mode='same')
    conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv5 = nn.Conv2d(feature_size, num_anchors * 28, kernel_size=3, pad_mode='same')
    return nn.SequentialCell([conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU(), conv4, nn.ReLU(), conv5])


def BucketModel(in_channel, num_anchors, kernel_size=3, stride=1, pad_mod='same', feature_size=256):
    conv1 = nn.Conv2d(in_channel, feature_size, kernel_size=3, pad_mode='same')
    conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv5 = nn.Conv2d(feature_size, num_anchors * 28, kernel_size=3, pad_mode='same')
    return nn.SequentialCell([conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU(), conv4, nn.ReLU(), conv5])


class MultiBox(nn.Cell):
    """
    Multibox conv layers. Each multibox layer contains class conf scores and localization predictions.

    Args:
        config (dict): The default config of retinanet.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, config):
        super(MultiBox, self).__init__()

        out_channels = config.extras_out_channels
        num_default = config.num_default
        loc_layers = []
        cls_layers = []
        buk_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [RegressionModel(in_channel=out_channel, num_anchors=num_default[k])]
            cls_layers += [ClassificationModel(in_channel=out_channel, num_anchors=num_default[k])]
            buk_layers += [BucketModel(in_channel=out_channel, num_anchors=num_default[k])]

        self.multi_loc_layers = nn.layer.CellList(loc_layers)
        self.multi_cls_layers = nn.layer.CellList(cls_layers)
        self.multi_buk_layers = nn.layer.CellList(buk_layers)
        self.flatten_concat = FlattenConcat(config)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        buk_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)
            buk_outputs += (self.multi_buk_layers[i](inputs[i]),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs), self.flatten_concat(buk_outputs)


class SigmoidFocalClassificationLoss(nn.Cell):
    def __init__(self, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = ms.Tensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss(reduction="none")
        self.is_weight = (weight is not None)
        self.onehot = P.OneHot()

    def reduce_loss(self, loss):
        """Reduce loss as specified.
        Args:
            loss (Tensor): Elementwise loss tensor.
        Return:
            Tensor: Reduced loss tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def weight_reduce_loss(self, loss):
        # if avg_factor is not specified, just reduce the loss
        if self.avg_factor is None:
            loss = self.reduce_loss(loss)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if self.reduction == 'mean':
                loss = loss.sum() / self.avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif self.reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss

    def construct(self, pred, target):
        pred_sigmoid = self.sigmoid(pred)
        target = self.onehot(target, pred.shape[2] + 1, Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32)) # 独热编码还是81维 但是只取后面80维度拟合pred
        target = target[...,1:]
        target = ops.cast(target, pred.dtype)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * ops.pow(pt, self.gamma)
        loss = self.binary_cross_entropy_with_logits(pred, target) * focal_weight
        if self.is_weight:
            weight = self.weight
            if self.weight.shape != loss.shape:
                if self.weight.shape[0] == loss.shape[0]:
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    weight = self.weight.view(-1, 1)
                elif self.weight.size == loss.size:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    weight = self.weight.view(loss.shape[0], -1)
                elif self.weight.ndim != loss.ndim:
                    raise ValueError(f"weight shape {self.weight.shape} is not match to loss shape {loss.shape}")
            loss = loss * weight
        loss = self.weight_reduce_loss(loss)
        return loss


class retinahead(nn.Cell):
    """retinahead"""
    def __init__(self, backbone, config, is_training=True):
        super(retinahead, self).__init__()

        self.fpn = FPN(backbone=backbone, config=config)
        self.multi_box = MultiBox(config)
        self.is_training = is_training
        if not is_training:
            self.activation = P.Sigmoid()

    def construct(self, inputs):
        features = self.fpn(inputs)
        pred_loc, pred_label, pred_buk = self.multi_box(features)
        return pred_loc, pred_label, pred_buk


class retinanetWithLossCell(nn.Cell):
    """"
    Provide retinanet training loss through network.

    Args:
        network (Cell): The training network.
        config (dict): retinanet config.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, network, config):
        super(retinanetWithLossCell, self).__init__()
        self.network = network
        self.less = P.Less()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.expand_dims = P.ExpandDims()
        self.loc_loss = nn.SmoothL1Loss(beta=1/9,reduction='none')
    
    def buk_loss_sigmoid(self, pred_buk, gt_buk, mask):

        batch_size = pred_buk.shape[0]
        pred_buk1 = nn.LogSigmoid()(pred_buk)
        pred_buk2 = nn.LogSigmoid()(1 - pred_buk)
        loss_entropy = self.reduce_sum((gt_buk * pred_buk1 + (1 - gt_buk) * pred_buk2) * mask, -1) / 28
        buk_loss = P.Neg()(self.reduce_sum(loss_entropy, -1))

        return buk_loss

    def construct(self, x, gt_loc, gt_loc_weights, gt_label, gt_buk, gt_buk_weights, num_matched_boxes):
        """construct"""
        """
        gt_loc:  Tensor(float32) shape:((n, 7555, 4)) n为batch_size  7555 = 75*75 + 38*38 + 19*19 + 10*10 +5*5
        gt_label:Tensor(int32)   shape:(n, 7555)
        gt_buk:  Tensor(int32)   shape:(n, 7555, 32)
        num_matched_boxes:Tensor(shape=[], dtype=Float32, value= 27)
        """
        # import pdb
        # pdb.set_trace()
        pred_loc, pred_label, pred_buk = self.network(x)
        num_matched_boxes = self.reduce_sum(F.cast(num_matched_boxes, mstype.float32)) # 27.0 mask的求和

        # import pdb
        # pdb.set_trace()
        # Localization Loss
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * gt_loc_weights # smooth_l1.shape:(n, 7555, 28)
        loss_loc = self.reduce_sum(smooth_l1) / 8.0 / num_matched_boxes # loss_loc.shape:(8,) self.reduce_mean(smooth_l1, -1).shape:(8,7555)
        #  改为了预测(x,y,w,h) 改了之后Tensor(shape=[8], dtype=Float32, value= [ 7.14661865e+01,  9.64509964e+01,  2.26468185e+02,  1.10275787e+02,  2.05055878e+02,  3.03910370e+01,  3.47374908e+02,  6.25472145e+01])

        # Classification Loss
        # import pdb
        # pdb.set_trace()
        mask_cls = F.cast(self.less(-1, gt_label), mstype.float32)
        mask_cls = self.tile(self.expand_dims(mask_cls, -1), (1, 1, 80))
        focal_loss = SigmoidFocalClassificationLoss(weight=mask_cls, avg_factor=num_matched_boxes)
        loss_cls = focal_loss(pred_label, gt_label)
        # loss_cls = self.reduce_sum(loss_cls, (1, 2)) # loss_cls.shape:(n,) (1, 2)确定的是相加消去的维度

        # import pdb
        # pdb.set_trace()
        # Bucket Loss
        bce_loss = nn.BCEWithLogitsLoss(reduction='sum',weight=gt_buk_weights)
        loss_buk = bce_loss(pred_buk, gt_buk)
        # loss_buk = self.buk_loss_sigmoid(pred_buk, gt_buk, gt_buk_weights)
        loss_buk = loss_buk / num_matched_boxes / 4 / 7
        # loss_buk = self.reduce_sum(loss_buk)
        # Tensor(shape=[], dtype=Float32, value= 45.7573)

        print(f'loss_loc:{loss_loc} loss_cls:{loss_cls} loss_buk:{loss_buk}')
        return (loss_cls + 1.5 * loss_loc + 1.5 * loss_buk) # focalloss计算损失用正负样本 但是平均的时候只除以正样本数

class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of retinanet network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss

class retinanetInferWithDecoder(nn.Cell):
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
    def __init__(self, network, default_boxes, config):
        super(retinanetInferWithDecoder, self).__init__()
        self.network = network
        self.default_boxes = default_boxes # 这里传入的default_boxes是加上bucket的方框(-1,32)
    
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

    def construct(self, x):
        """construct"""
        bbox_reg_pred, scores, bbox_cls_pred = self.network(x)
        bbox_reg_pred, scores, bbox_cls_pred = bbox_reg_pred.squeeze(), scores.squeeze(), bbox_cls_pred.squeeze() # eval的时候batch_size是1，所以直接squeeze
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
            final_bbox_cls_pred = cat((final_bbox_cls_pred,filtered_results_level['bbox_cls_pred']))
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
        # bboxes [1000,4] labels [1000] scores [1000] confids [1000] 但是mmdet有的不是1000 5张图可能前两张是1000 后几张就没有1000了
                    # results['anchors'] [15200,4]这里的self.default_boxes必须是没画bucket的方框 mmdet里面用的pad后的图片尺寸 这里还是归一化过的
                    #     tensor([[ -16.,  -16.,   16.,   16.],
                    #             [  -8.,  -16.,   24.,   16.],
                    #             [   0.,  -16.,   32.,   16.],
                    #             ...,
                    #             [1176.,  776., 1208.,  808.],
                    #             [1184.,  776., 1216.,  808.],
                    #             [1192.,  776., 1224.,  808.]], device='cuda:2')

                    # results['bbox_cls_pred']  [15200,28]
                    #     tensor([[-0.8787, -1.3163, -1.2070,  ..., -0.2010, -0.1811, -1.3332],
                    #             [-1.2095, -2.3916, -2.3461,  ..., -0.0480,  0.0962, -2.1478],
                    #             [-1.7109, -2.9341, -2.8480,  ..., -0.1196,  0.3706, -2.2332],
                    #             ...,
                    #             [-1.8371, -1.4826, -0.8867,  ..., -2.4646,  1.4940,  0.4312],
                    #             [-1.0151, -1.1005, -1.2790,  ..., -1.9562,  0.9361,  0.1095],
                    #             [-0.6445, -1.1045, -1.7056,  ..., -1.3750,  0.5043, -0.0891]],device='cuda:2')

                    # results['bbox_reg_pred']  [15200,28]
                    # tensor([[ 0.0547, -0.1717, -0.1116,  ...,  0.0068, -0.1942, -0.3744],
                    #         [ 0.6253, -0.0112, -0.1217,  ...,  0.0706, -0.2690, -0.4680],
                    #         [ 1.0880, -0.1151, -0.2484,  ...,  0.1618, -0.2729, -0.4884],
                    #         ...,
                    #         [ 0.4036, -0.2733, -0.1329,  ...,  0.6721,  0.2346, -0.7443],
                    #         [ 0.3862, -0.0532,  0.0210,  ...,  0.5237,  0.1527, -0.7044],
                    #         [ 0.5673,  0.0306, -0.0683,  ...,  0.3520,  0.0710, -0.4915]],
                    #     device='cuda:2')
