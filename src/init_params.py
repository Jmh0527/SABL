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
"""Parameters utils"""

from mindspore.common.initializer import initializer, TruncatedNormal


def init_net_param(network, initialize_mode='xavier_uniform'):
    """Init the parameters in net."""
    params = network.trainable_params()
    num = 0
    print('Start initialize net parameters!')
    for p in params:
        print(f'{num}/{len(params)}')
        if 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            if 'multi_box' in p.name:
                p.set_data(initializer('normal', p.data.shape, p.data.dtype))
            elif 'fpn.P' in p.name:
                p.set_data(initializer('xavier_uniform', p.data.shape, p.data.dtype))
        num += 1
    print('Initialize net parameters done!')


def filter_checkpoint_parameter(param_dict):
    """remove useless parameters"""
    for key in list(param_dict.keys()):
        if 'multi_loc_layers' in key or 'multi_cls_layers' in key:
            del param_dict[key]
