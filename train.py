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

"""Train retinanet and get checkpoint files."""

import os
import mindspore
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor, Callback
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.retinahead import retinanetWithLossCell, TrainingWrapper, retinahead
from src.backbone import resnet101
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_num, get_device_id
from src.dataset import create_retinanet_dataset, create_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter


set_seed(1)

def resnet_backbone_ckpt():
    ckpt_retinanet = '/home/data/jmh/retinanet_res101_checkpoint/retinanetresnet101_ascend_v190_coco2017_research_cv_mAP36.72.ckpt'
    param_retinanet = mindspore.load_checkpoint(ckpt_retinanet)
    ckpt_resnet = '/home/data/jmh/retinanet_res101_checkpoint/resnet101_ascend_v190_imagenet2012_official_cv_top1acc78.55_top5acc94.34.ckpt'
    param_resnet = mindspore.load_checkpoint(ckpt_resnet)
    # 1 - 312 0-311
    # retinanet : 579 - 892 resnet  524 - 837
    # retinanet : 371 - 578 resnet  316 - 523
    # 370 - 891  315 - 836
    param_res_for_retina = {}
    keys_for_retina = []
    values_for_retina = []
    keys_for_res = []
    values_for_res = []

    keys_for_res_for_retina = []
    values_for_res_for_retina = []
    for k, v in param_retinanet.items():
        keys_for_retina.append(k)
        values_for_retina.append(v)
    for k, v in param_resnet.items():
        keys_for_res.append(k)
        values_for_res.append(v)
    for i in range(312):
        param_res_for_retina[keys_for_retina[i]] = values_for_res[i]
    for i in range(522):
        param_res_for_retina[keys_for_retina[i+370]] = values_for_res[i+315]

    for k, v in param_res_for_retina.items():
        keys_for_res_for_retina.append(k)
        values_for_res_for_retina.append(v)

    final_param = {}
    for index, i in enumerate(values_for_res_for_retina):
        i.name = keys_for_res_for_retina[index]
        final_param[keys_for_res_for_retina[index]] = i
    return final_param


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("lr:[{:8.6f}]".format(self.lr_init[cb_params.cur_step_num-1]), flush=True)


def modelarts_process():
    config.save_checkpoint_path = os.path.join(config.output_path, str(get_device_id()), config.save_checkpoint_path)
    if config.need_modelarts_dataset_unzip:
        config.coco_root = os.path.join(config.coco_root, config.modelarts_dataset_unzip_name)
        print(os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))


@moxing_wrapper(pre_process=modelarts_process)
def train_retinanet_resnet101():
    """ train_retinanet_resnet101 """

    if config.train_mode == 'Graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=config.run_platform) #静态图
    else :
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.run_platform) #动态图

    if config.run_platform == "Ascend":
        if config.distribute:
            if os.getenv("DEVICE_ID", "not_set").isdigit():
                context.set_context(device_id=int(os.getenv("DEVICE_ID")))
            init()
            device_num = get_device_num()
            rank = get_rank()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
        else:
            rank = 0
            device_num = 1
            context.set_context(device_id=get_device_id())

    elif config.run_platform == "GPU":
        rank = config.device_id
        device_num = config.device_num
        if config.distribute:
            init()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)

    else:
        raise ValueError("Unsupported platform, GPU or Ascend is supported only.")

    mindrecord_file = create_mindrecord(config.dataset, "retina2.mindrecord", True)

    if not config.only_create_dataset:
        loss_scale = float(config.loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as retinanet.mindrecord0.
        dataset = create_retinanet_dataset(mindrecord_file, repeat_num=1,
                                           batch_size=config.batch_size, device_num=device_num, rank=rank)

        dataset_size = dataset.get_dataset_size()
        print("Create dataset done!")
        backbone = resnet101(config.num_classes)
        param_res_for_retina = resnet_backbone_ckpt()
        retinanet = retinahead(backbone, config)
        net = retinanetWithLossCell(retinanet, config)
        if config.run_platform == "Ascend":
            net.to_float(mindspore.float16)
        else:
            net.to_float(mindspore.float32)
        import pdb 
        pdb.set_trace()
        init_net_param(net)
        load_param_into_net(net, param_res_for_retina)
        lr = Tensor(get_lr(global_step=config.global_step,
                           lr_init=config.lr_init, lr_end=config.lr_end_rate * config.lr, lr_max=config.lr,
                           warmup_epochs1=config.warmup_epochs1, warmup_epochs2=config.warmup_epochs2,
                           warmup_epochs3=config.warmup_epochs3, warmup_epochs4=config.warmup_epochs4,
                           warmup_epochs5=config.warmup_epochs5, total_epochs=config.epoch_size,
                           steps_per_epoch=dataset_size))
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)
        # model = Model(net, optimizer=opt, amp_level="O2")
        model = Model(net)
        if config.train_mode == "Graph":
            print("Start train retinanet, the first epoch will be slower because of the graph compilation.")
        cb = [TimeMonitor(), LossMonitor()]
        cb += [Monitor(lr_init=lr.asnumpy())]
        config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size * config.save_checkpoint_epochs,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="retinanet", directory=config.save_checkpoint_path, config=config_ck)
        if config.distribute:
            if rank == 0:
                cb += [ckpt_cb]
            model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=False)
        else:
            cb += [ckpt_cb]
            model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=False)


if __name__ == '__main__':
    train_retinanet_resnet101()
