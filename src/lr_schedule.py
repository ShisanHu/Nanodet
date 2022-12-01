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

"""Learning rate schedule"""

import math
import numpy as np
from collections import Counter


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate."""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr

def warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    """Warmup step learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr * gamma**milestones_steps_counter[i]
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def multi_step_lr(lr, milestones, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    return warmup_step_lr(lr, milestones, steps_per_epoch, warmup_epochs, max_epoch, gamma=gamma)



def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs1, warmup_epochs2,
           warmup_epochs3, warmup_epochs4, warmup_epochs5, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(float): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps1 = steps_per_epoch * warmup_epochs1
    warmup_steps2 = warmup_steps1 + steps_per_epoch * warmup_epochs2
    warmup_steps3 = warmup_steps2 + steps_per_epoch * warmup_epochs3
    warmup_steps4 = warmup_steps3 + steps_per_epoch * warmup_epochs4
    warmup_steps5 = warmup_steps4 + steps_per_epoch * warmup_epochs5

    for i in range(total_steps):
        if i < warmup_steps1:
            lr = lr_init*(warmup_steps1-i) / (warmup_steps1) + \
            (lr_max*1e-4) * i / (warmup_steps1*3)
        elif warmup_steps1 <= i < warmup_steps2:
            lr = 1e-5*(warmup_steps2-i) / (warmup_steps2 - warmup_steps1) + \
            (lr_max*1e-3) * (i-warmup_steps1) / (warmup_steps2 - warmup_steps1)
        elif warmup_steps2 <= i < warmup_steps3:
            lr = 1e-4*(warmup_steps3-i) / (warmup_steps3 - warmup_steps2) + \
            (lr_max*1e-2) * (i-warmup_steps2) / (warmup_steps3 - warmup_steps2)
        elif warmup_steps3 <= i < warmup_steps4:
            lr = 1e-3*(warmup_steps4-i) / (warmup_steps4 - warmup_steps3) + \
            (lr_max*1e-1) * (i-warmup_steps3) / (warmup_steps4 - warmup_steps3)
        elif warmup_steps4 <= i < warmup_steps5:
            lr = 1e-2*(warmup_steps5-i) / (warmup_steps5 - warmup_steps4) + \
            lr_max  * (i-warmup_steps4) / (warmup_steps5 - warmup_steps4)
        else:
            lr = lr_end + \
                 (lr_max - lr_end) * \
                 (1. + math.cos(math.pi * (i-warmup_steps5) / (total_steps - warmup_steps5))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def get_MultiStepLR(lr, milestones, steps_per_epoch, warmup_epochs, max_epoch, gamma):
    return multi_step_lr(lr, milestones, steps_per_epoch, warmup_epochs, max_epoch, gamma)

if __name__ == "__main__":
    lr = get_MultiStepLR(0.14, [240, 260, 275], 100, 3, 280, 0.1)
    print(lr.size)
    for i in lr:
        print(i)