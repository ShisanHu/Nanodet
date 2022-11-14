import mindspore as ms
import mindspore.nn as nn


class GFLHead(nn.Cell):
    def __init__(
            self,
            num_classes,
            loss,
            input_channel,
            feat_channels=256,
            stacked_convs=4,
            octave_base_scale=4,
            strides=None,
            conv_cfg=None,
            norm_cfg=None,
            reg_max=16,
            use_sigmoid=True,
    ):
        super(GFLHead, self).__init__()
        if strides is None:
            strides = [8, 16, 32]

        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.grid_cell_scale = octave_base_scale
        self.strides = strides
        self.reg_max = reg_max

        self.loss_cfg = loss
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()
        self.init_weights()

    def __init_layers(self):
        self.relu = nn.ReLU()
        self.cls_convs = nn.CellList()
        self.reg_convs = nn.CellList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.SequentialCell([
                    nn.Conv2d(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.feat_channels)
                ])
            )
            self.reg_convs.append(
                nn.SequentialCell([
                    nn.Conv2d(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.feat_channels)
                ])
            )
            self.gfl_cls = nn.Conv2d(
                self.feat_channels, self.cls_out_channels, 3, padding=1
            )
            self.gfl_reg = nn.Conv2d(
                self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1
            )




