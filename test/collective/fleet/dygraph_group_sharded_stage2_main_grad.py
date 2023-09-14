# -*- coding: UTF-8 -*-

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

import paddle
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import (
    GroupShardedStage2,
)
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.nn import Linear

seed = 2022
epoch = 2
linear_size = 1000

np.random.seed(seed)
paddle.seed(seed)


class MLP(paddle.nn.Layer):
    def __init__(self, linear_size=1000):
        super().__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples=200, linear_size=1000):
        self.num_samples = num_samples
        self.linear_size = linear_size

    def __getitem__(self, idx):
        img = np.random.rand(self.linear_size).astype('float32')
        label = np.ones(1).astype('int64')
        return img, label

    def __len__(self):
        return self.num_samples


def optimizer_setting(model, main_grad=False):
    if main_grad:
        mix_precision_utils.MixPrecisionLayer(model, dtype='bfloat16')
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=0.001,
        weight_decay=0.00001,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        multi_precision=main_grad,
    )
    if main_grad:
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

    return optimizer


def train_mlp(
    model,
    main_grad=False,
    batch_size=100,
    accumulate_grad=False,
):
    group = paddle.distributed.new_group([0, 1], backend="nccl")
    optimizer = optimizer_setting(model=model, main_grad=main_grad)

    optimizer = GroupShardedOptimizerStage2(
        params=optimizer._parameter_list, optim=optimizer, group=group
    )

    model = GroupShardedStage2(
        model, optimizer, group=group, buffer_max_size=2**21
    )

    paddle.seed(2023)
    np.random.seed(2023)
    train_loader = paddle.io.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    if main_grad:
        custom_white_list = None
        level = "O2"
    else:
        custom_white_list = ["matmul", "add", "cross_entropy_with_softmax"]
        level = "O1"

    model.to(device="gpu")

    losses = []
    for eop in range(epoch):
        model.train()

        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True

            with paddle.amp.auto_cast(
                enable=True, level=level, custom_white_list=custom_white_list
            ):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(
                    input=out, label=label
                )
                loss = paddle.mean(x=loss)
                losses.append(loss)
            loss.backward()

            if not accumulate_grad:
                optimizer.step()
                optimizer.clear_grad()

        if accumulate_grad:
            optimizer.step()
            optimizer.clear_grad()

    return losses


def get_model(state_dict, main_grad=False):
    mlp = MLP()
    mlp.set_state_dict(state_dict)
    if main_grad:
        mlp = paddle.amp.decorate(models=mlp, level='O2', dtype='bfloat16')
    return mlp


def test_sharding_stage2():
    paddle.distributed.init_parallel_env()
    mlp = MLP()
    state_dict = mlp.state_dict()
    mlp1 = get_model(state_dict, main_grad=False)
    mlp2 = get_model(state_dict, main_grad=True)
    mlp3 = get_model(state_dict, main_grad=False)
    mlp4 = get_model(state_dict, main_grad=True)

    # stage2
    o1_losses = train_mlp(mlp1, main_grad=False)
    o2_losses = train_mlp(mlp2, main_grad=True)

    # stage 2 grad accumulation
    o1_losses = train_mlp(mlp3, main_grad=False, accumulate_grad=True)
    o2_losses = train_mlp(mlp4, main_grad=True, accumulate_grad=True)

    return


if __name__ == '__main__':
    test_sharding_stage2()
