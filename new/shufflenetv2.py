import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
import mindspore.common.dtype as mstype
import numpy as np

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    # reshape
    x = P.Reshape()(x,(batchsize, groups, channels_per_group, height, width))
    x = P.Transpose()(x, (0, 2, 1, 3, 4))
    x = P.Reshape()(x, (batchsize, -1, height, width))
    return x

class ShuffleV2Block(nn.Cell):
    def __init__(self, inp, oup, stride, activation="ReLU"):
        super(ShuffleV2Block, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride
        branch_features = oup // 2


        if self.stride > 1:
            self.branch1 = nn.SequentialCell([
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp,branch_features,kernel_size=1,stride=1,padding=0,has_bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(),
            ])
        else:
            self.branch1 = nn.SequentialCell()

        self.branch2 = nn.SequentialCell([
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                has_bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                has_bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(),
        ])

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size ,stride, "pad", padding,group=i, has_bias=bias)

    def construct(self, x):
        if self.stride == 1:
            x1, x2 = P.Split(axis=1,output_num=2)(x)
            out = P.Concat(axis=1)((x1, self.branch2(x2)))
        else:
            out = P.Concat(axis=1)((self.branch1(x), self.branch2(x)))
        out = channel_shuffle(out, 2)
        return out

class shuffleNetV2(nn.Cell):
    def __init__(self, model_size="1.0x", out_stages=(2, 3, 4),  kernal_size=3, with_last_conv=False, activation="ReLU"):
        super(shuffleNetV2, self).__init__()
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        self.activation = activation
        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=False),
            nn.BatchNorm2d(output_channels,),
            nn.ReLU(),
        ])
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, self.stage_repeats, self._stage_out_channels[1:]
        ):
            seq = [
                ShuffleV2Block(
                    input_channels, output_channels, 2, activation=activation
                )
            ]
            for i in range(repeats - 1):
                seq.append(
                    ShuffleV2Block(
                        output_channels, output_channels, 1, activation=activation
                    )
                )
            setattr(self, name, nn.SequentialCell(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            conv5 = nn.SequentialCell(
                nn.Conv2d(input_channels, output_channels, kernel_size=1,stride=1,padding=0,has_bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU,
            )
            self.stage4.add_module("conv5", conv5)
            # self._initialize_weights()

    def construct(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self):
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    m.weight.set_data(Tensor(np.random.normal(0, 0.01,
                                                              m.weight.data.shape).astype("float32")))
                else:
                    m.weight.set_data(Tensor(np.random.normal(0, 1.0/m.weight.data.shape[1],
                                                              m.weight.data.shape).astype("float32")))

            if isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))


if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    np.random.seed(1)
    x = Tensor(np.random.randint(0, 255, (1, 3, 320, 320)), mstype.float32)
    backbone = shuffleNetV2()
    for item in backbone.parameters_and_names():
        print(item[0])
    out = backbone(x)
    print("!")