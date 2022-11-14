import mindspore.nn as nn
import mindspore.common.dtype as mstype
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.ops.operations as P
from mindspore.common.initializer import initializer, XavierUniform
from shufflenetv2 import shuffleNetV2

class FPN(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        conv_cfg=None,
        norm_cfg=None,
        activation=None,
    ):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level

        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.CellList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=1,has_bias=True)
            self.lateral_convs.append(l_conv)
        self.init_weights()

    def init_weights(self):
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(initializer(XavierUniform(),m.weight.shape, mstype.float32))

    def construct(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += nn.ResizeBilinear()(laterals[i], scale_factor=2)

        # build outputs
        outs = [
            # self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            laterals[i]
            for i in range(used_backbone_levels)
        ]
        return tuple(outs)

class PAN(FPN):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_outs,
            start_level=0,
            end_level=-1,
            conv_cfg=None,
            norm_cfg=None,
            activation=None,
    ):
        super(PAN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            conv_cfg,
            norm_cfg,
            activation,)
        self.init_weights()
        self.feature_size = [40,20,10]

    def construct(self, inputs):
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += nn.ResizeBilinear()(laterals[i], scale_factor=2)

        inter_outs = [laterals[i] for i in range(used_backbone_levels)]

        for i in range(0, used_backbone_levels - 1):
            laterals[i + 1] += nn.ResizeBilinear()(laterals[i], size=(self.feature_size[i+1], self.feature_size[i+1]) )

        outs = []
        outs.append(inter_outs[0])
        outs.extend([inter_outs[i] for i in range(1, used_backbone_levels)])
        return tuple(outs)

if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    np.random.seed(1)
    x = Tensor(np.random.randint(0, 255, (1, 3, 320, 320)), mstype.float32)
    backbone = shuffleNetV2()
    out = backbone(x)
    pan = PAN(in_channels=[116,232,464],out_channels=96,num_outs=3)
    outs = pan(out)
    for item in pan.parameters_and_names():
        print(item[0])