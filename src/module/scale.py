# check passed
import mindspore as ms
import mindspore.nn as nn

class Scale(nn.Cell):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = ms.Parameter(ms.Tensor(scale, dtype=ms.float32))

    def construct(self, x):
        return x * self.scale

