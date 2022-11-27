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

"""Evaluation for retinanet"""

import os
import time
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.retinahead import retinahead, retinanetInferWithDecoder
from src.backbone import resnet101
from src.dataset import create_retinanet_dataset, data_to_mindrecord_byte_image, voc_data_to_mindrecord
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id
from src.coco_eval import metrics
from src.box_utils import default_boxes,default_square_boxes_ltrb

def retinanet_eval(dataset_path, ckpt_path):
    """retinanet evaluation."""
    softmax = nn.Softmax(axis=-1)
    sigmoid = nn.Sigmoid()
    batch_size = 1
    ds = create_retinanet_dataset(dataset_path, batch_size=batch_size, repeat_num=1, is_training=False)
    backbone = resnet101(config.num_classes)
    net = retinahead(backbone, config)
    # net = retinanetInferWithDecoder(net, Tensor(default_boxes), config)
    net = retinanetInferWithDecoder(net, Tensor(default_square_boxes_ltrb), config)
    print("Load Checkpoint!")
    print("ckpt_path: ", ckpt_path)
    param_dict = load_checkpoint(ckpt_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)

    net.set_train(False)
    i = batch_size
    total = ds.get_dataset_size() * batch_size
    start = time.time()
    pred_data = []
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_id = data['img_id']
        img_np = data['image']
        image_shape = data['image_shape']
        scale = data['scale']

        # import pdb
        # pdb.set_trace()
        output = net(Tensor(img_np)) # bboxes, labels, scores, confids   labels是0-79 不是独热编码
        for batch_idx in range(img_np.shape[0]):
            bboxes, labels, scores, confids = output[0].asnumpy(), output[1].asnumpy(), output[2].asnumpy(), output[3].asnumpy() # 做了squeezee而且val的时候batch_size都是1 所以不用[batch_idx]了
            bboxes = bboxes / scale
            # ************************************************************ #
            bboxes[:,[0,2]] = np.clip(bboxes, 0, image_shape[0]) # 有待核验
            bboxes[:,[1,3]] = np.clip(bboxes, 0, image_shape[1])
            # ************************************************************ #
            pred_data.append({"boxes": bboxes.astype(np.float16), # 转为float16之后，显存保持3000不变，否则会一直涨，最后能涨到30000
                              "box_scores": (scores * confids).astype(np.float16), # 在retinahead的decoder中转为了(ymin xmin ymax xmax)
                              "labels":labels.astype(np.int16),
                              "img_id": int(np.squeeze(img_id[batch_idx]))})
        percent = round(i / total * 100., 2)
        print("    {}% [{}/{}]".format(str(percent), i, total))
        i += batch_size
    cost_time = int((time.time() - start) * 1000)
    print("    100% [{arg1}/{arg1}] cost {arg2} ms".format(arg1=total, arg2=cost_time))
    mAP = metrics(pred_data) # pred_data为长度4541的列表，列表元素是字典
    # pred_data[0]['boxes'].shape (7555, 4)
    # pred_data[0]['box_scores'].shape (7555, 81)
    print("\n========================================\n")
    print("mAP: {}".format(mAP))


def modelarts_process():
    if config.need_modelarts_dataset_unzip:
        config.coco_root = os.path.join(config.coco_root, config.modelarts_dataset_unzip_name)
        print(os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))


@moxing_wrapper(pre_process=modelarts_process)
def eval_retinanet_resnet101():
    """ eval_retinanet_resnet101 """
    # context.set_context(mode=context.GRAPH_MODE, device_target=config.run_platform, device_id=get_device_id())
    if config.train_mode == 'Graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=config.run_platform, device_id=get_device_id()) #静态图
    else :
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.run_platform, device_id=get_device_id()) #动态图

    prefix = "retinanet_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if config.dataset == "voc":
        config.coco_root = config.voc_root
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", False, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        elif config.dataset == "voc":
            if os.path.isdir(config.voc_dir) and os.path.isdir(config.voc_root):
                print("Create Mindrecord.")
                voc_data_to_mindrecord(mindrecord_dir, False, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("voc_root or voc_dir not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", False, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.")
    else:
        print("Mindrecord file exists.")

    print("Start Eval!")
    retinanet_eval(mindrecord_file, config.checkpoint_path)


if __name__ == '__main__':
    eval_retinanet_resnet101()
