import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from typing import Tuple
import numpy as np
from src.model_utils.config import config
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
import time


def ClassificationModel(in_channel, kernel_size=3,
                        stride=1, pad_mod='same', num_classes=80, feature_size=256):
    conv1 = nn.Conv2d(in_channel, feature_size, kernel_size=3, pad_mode='same')
    conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv3 = nn.Conv2d(feature_size, num_classes, kernel_size=3, pad_mode='same')
    return nn.SequentialCell([conv1, nn.ReLU(), conv2, nn.ReLU(), conv3])


def RegressionModel(in_channel, reg_max, kernel_size=3, stride=1, pad_mod='same', feature_size=256):
    conv1 = nn.Conv2d(in_channel, feature_size, kernel_size=3, pad_mode='same')
    conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv3 = nn.Conv2d(feature_size, (reg_max + 1) * 4, kernel_size=3, pad_mode='same')
    return nn.SequentialCell([conv1, nn.ReLU(), conv2, nn.ReLU(), conv3])


def ShuffleBlockMainBranch(in_channels, mid_channels, out_channels, k_size, stride):
    conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, pad_mode='same')
    norm1 = nn.BatchNorm2d(num_features=mid_channels, momentum=0.9)
    relu1 = nn.ReLU()
    conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=k_size, stride=stride,
                      pad_mode='same', group=mid_channels)
    norm2 = nn.BatchNorm2d(num_features=mid_channels, momentum=0.9)
    conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, pad_mode='same')
    norm3 = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
    relu2 = nn.ReLU()
    return nn.SequentialCell([conv1, norm1, relu1, conv2, norm2, conv3, norm3, relu2])


def ShuffleBlockSubBranch(in_channels, k_size, stride):
    conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=k_size, stride=stride,
                      pad_mode='same', group=in_channels)
    norm1 = nn.BatchNorm2d(num_features=in_channels, momentum=0.9)
    conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, pad_mode='same')
    norm2 = nn.BatchNorm2d(num_features=in_channels, momentum=0.9)
    relu = nn.ReLU()
    return nn.SequentialCell([conv1, norm1, conv2, norm2, relu])


class ChannelShuffle(nn.Cell):
    def __init__(self):
        super(ChannelShuffle, self).__init__()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.shape = P.Shape()

    def construct(self, x):
        batchsize, num_channels, height, width = self.shape(x)
        x = self.reshape(x, (batchsize * num_channels // 2, 2, height * width,))
        x = self.transpose(x, (1, 0, 2,))
        x = self.reshape(x, (2, -1, num_channels // 2, height, width,))
        return x[0:1, :, :, :, :], x[-1:, :, :, :, :]


class MultiConcat(nn.Cell):
    def __init__(self):
        super(MultiConcat, self).__init__()
        self.concat = P.Concat(axis=1)
        self.transpose = P.Transpose()

    def construct(self, inputs):
        output = ()
        batch_size = F.shape(inputs[0])[0]
        channel_size = F.shape(inputs[0])[1]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (F.reshape(x, (batch_size, -1, channel_size)),)
        ans = self.concat(output)
        return ans


class Integral(nn.Cell):
    def __init__(self, config):
        super(Integral, self).__init__()
        self.reg_max = config.reg_max
        self.softmax = P.Softmax(axis=-1)
        self.project = ms.numpy.linspace(0, self.reg_max, self.reg_max + 1).expand_dims(0)
        self.project_weight = ms.Parameter(self.project, "project", requires_grad=False)
        self.dense = nn.Dense(self.reg_max + 1, 1, weight_init=self.project_weight)

    def construct(self, x):
        shape = x.shape
        x = self.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1))
        x = self.dense(x).reshape(*shape[:-1], 4)
        return x


class IntegralII(nn.Cell):
    def __init__(self, config):
        super(IntegralII, self).__init__()
        self.reg_max = config.reg_max
        self.softmax = P.Softmax(axis=-1)

        self.start = Tensor(0, mstype.float32)
        self.stop = Tensor(config.reg_max)
        self.linspace = Tensor([[0, 1, 2, 3, 4, 5, 6, 7]], mstype.float32)
        self.dense = nn.Dense(self.reg_max + 1, 1, weight_init=self.linspace)
        self.reshape = P.Reshape()

    def construct(self, x):
        shape = x.shape
        x = self.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1))
        x = self.dense(x).reshape(*shape[:-1], 4)
        return x

class Distance2bbox(nn.Cell):
    def __init__(self, max_shape=None):
        super(Distance2bbox, self).__init__()
        self.max_shape = max_shape
        self.stack = P.Stack(-1)

    def construct(self, points, distance):
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        return self.stack([x1, y1, x2, y2])


class ShuffleV2Block(nn.Cell):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        outputs = oup - inp
        branch_main = [
            # pw
            nn.Conv2d(in_channels=inp, out_channels=mid_channels, kernel_size=1, stride=1,
                      pad_mode='same', padding=0, has_bias=False),
            nn.BatchNorm2d(num_features=mid_channels, momentum=0.9),
            nn.ReLU(),
            # dw
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=ksize, stride=stride,
                      pad_mode='same', padding=0, group=mid_channels, has_bias=False),

            nn.BatchNorm2d(num_features=mid_channels, momentum=0.9),
            # pw-linear
            nn.Conv2d(in_channels=mid_channels, out_channels=outputs, kernel_size=1, stride=1,
                      pad_mode='same', padding=0, has_bias=False),
            nn.BatchNorm2d(num_features=outputs, momentum=0.9),
            nn.ReLU(),
        ]
        self.branch_main = nn.SequentialCell(branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=ksize, stride=stride,
                          pad_mode='same', padding=0, group=inp, has_bias=False),
                nn.BatchNorm2d(num_features=inp, momentum=0.9),
                # pw-linear
                nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=1, stride=1,
                          pad_mode='same', padding=0, has_bias=False),
                nn.BatchNorm2d(num_features=inp, momentum=0.9),
                nn.ReLU(),
            ]
            self.branch_proj = nn.SequentialCell(branch_proj)
        else:
            self.branch_proj = None
        self.squeeze = P.Squeeze(axis=0)

    def construct(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            x_proj = self.squeeze(x_proj)
            x = self.squeeze(x)
            return P.Concat(1)((x_proj, self.branch_main(x)))
        if self.stride == 2:
            x_proj = old_x
            x = old_x
            return P.Concat(1)((self.branch_proj(x_proj), self.branch_main(x)))
        return None

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = P.Shape()(x)
        x = P.Reshape()(x, (batchsize * num_channels // 2, 2, height * width,))
        x = P.Transpose()(x, (1, 0, 2,))
        x = P.Reshape()(x, (2, -1, num_channels // 2, height, width,))
        return x[0:1, :, :, :, :], x[-1:, :, :, :, :]


class ShuffleV2BlockII(nn.Cell):
    def __init__(self, in_channels, mid_channels, out_channels, k_size, stride):
        super(ShuffleV2BlockII, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.k_size = k_size
        self.channel_shuffle = ChannelShuffle()
        self.stride = stride
        self.squeeze = P.Squeeze(axis=0)
        self.concat = P.Concat(axis=1)
        outputs = out_channels - in_channels
        self.main_branch = ShuffleBlockMainBranch(in_channels, mid_channels, outputs, k_size, stride)
        if stride == 2:
            self.sub_branch = ShuffleBlockSubBranch(in_channels, k_size, stride)

    def construct(self, x):
        if self.stride == 1:
            x_sub, x_main = self.channel_shuffle(x)
            x_sub = self.squeeze(x_sub)
            x_main = self.squeeze(x_main)
            return self.concat((x_sub, self.main_branch(x_main)))
        if self.stride == 2:
            x_sub = x
            x_main = x
            return self.concat((self.sub_branch(x_sub), self.main_branch(x_main)))
        return None

class MultiPred(nn.Cell):
    def __init__(self, config):
        super(MultiPred, self).__init__()
        out_channels = config.extras_out_channels
        self.reg_max = config.reg_max
        # out_channels = [96,96,96]
        cls_layers = []
        reg_layers = []
        for i, out_channel in enumerate(out_channels):
            cls_layers += [ClassificationModel(in_channel=out_channel)]
            reg_layers += [RegressionModel(in_channel=out_channel, reg_max=self.reg_max)]

        self.multi_cls_layers = nn.CellList(cls_layers)
        self.multi_reg_layers = nn.CellList(reg_layers)
        self.multi_concat = MultiConcat()

    def construct(self, inputs):
        cls_outputs = ()
        reg_outputs = ()
        # for i in range(len(self.multi_cls_layers)):
        for idx, x in enumerate(self.multi_cls_layers):
            cls_outputs += (self.multi_cls_layers[idx](inputs[idx]),)
            reg_outputs += (self.multi_reg_layers[idx](inputs[idx]),)
        return self.multi_concat(cls_outputs), self.multi_concat(reg_outputs)


class ShuffleNetV2II(nn.Cell):
    def __init__(self, model_size='1.0x'):
        super(ShuffleNetV2II, self).__init__()
        print('model size is ', model_size)
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]

        self.first_conv = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=input_channel, kernel_size=3, stride=2,
                      pad_mode='same', has_bias=False),
            nn.BatchNorm2d(num_features=input_channel, momentum=0.9),
            nn.ReLU(),
        ])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.features = []
        self.cellList = nn.CellList()

        # -> 0 1 2
        for idxstage in range(len(self.stage_repeats)):
            # -> 4 8 4
            numrepeat = self.stage_repeats[idxstage]
            # -> 0 1 2  -> 116 232 464
            output_channel = self.stage_out_channels[idxstage + 2]
            # -> 4 8 4
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel

            self.sequentialCell = nn.SequentialCell([*self.features])
            self.features = []
            self.cellList.append(self.sequentialCell)

    def construct(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)

        outputs = []
        for conv in self.cellList:
            x = conv(x)
            outputs.append(x)

        return outputs

class ShuffleNetV2III(nn.Cell):
    def __init__(self, block, model_size='1.0x'):
        super(ShuffleNetV2III, self).__init__()
        print('model size is ', model_size)
        stage_repeats = [4, 8, 4]
        if model_size == '0.5x':
            stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        input_channel = stage_out_channels[1]
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=input_channel, kernel_size=3, stride=2, pad_mode='same'),
            nn.BatchNorm2d(num_features=input_channel, momentum=0.9),
            nn.ReLU(),
        ])
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.stage2 = self._make_layer(block,
                                       stage_repeats[0],
                                       in_channel=input_channel,
                                       out_channel=stage_out_channels[2],
                                       stride=2)
        self.stage3 = self._make_layer(block,
                                       stage_repeats[1],
                                       in_channel=stage_out_channels[2],
                                       out_channel=stage_out_channels[3],
                                       stride=2)
        self.stage4 = self._make_layer(block,
                                       stage_repeats[2],
                                       in_channel=stage_out_channels[3],
                                       out_channel=stage_out_channels[4],
                                       stride=2)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        layers = []
        nanodet_block = ShuffleV2BlockII(in_channel, mid_channels=out_channel // 2, out_channels=out_channel, k_size=3, stride=stride)
        layers.append(nanodet_block)
        for _ in range(1, layer_num):
            nanodet_block = ShuffleV2BlockII(out_channel // 2, mid_channels=out_channel // 2, out_channels=out_channel, k_size=3, stride=1)
            layers.append(nanodet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        C1 = self.pool1(x)

        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        return C2, C3, C4

def shuffleNet(model_size='1.0x'):
    return ShuffleNetV2III(ShuffleV2BlockII, model_size=model_size)

class NanoDetII(nn.Cell):
    def __init__(self, backbone, config, is_training=True):
        super(NanoDetII, self).__init__()
        self.backbone = backbone
        feature_size = config.feature_size
        self.P4_1 = nn.Conv2d(464, 96, kernel_size=1, stride=1, pad_mode='same')
        self.P_upSample1 = P.ResizeNearestNeighbor((feature_size[1], feature_size[1]))
        self.P4_2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, pad_mode='same')

        self.P3_1 = nn.Conv2d(232, 96, kernel_size=1, stride=1, pad_mode='same')
        self.P_upSample2 = P.ResizeNearestNeighbor((feature_size[0], feature_size[0]))
        self.P_downSample1 = nn.Conv2d(96, 96, kernel_size=3, stride=2, pad_mode='same')
        self.P3_2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, pad_mode='same')

        self.P2_1 = nn.Conv2d(116, 96, kernel_size=1, stride=1, pad_mode='same')
        self.P_downSample2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, pad_mode='same')
        self.P2_2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, pad_mode='same')

        self.multiPred = MultiPred(config)
        self.is_training = is_training

    def construct(self, x):
        C2, C3, C4 = self.backbone(x)

        P4 = self.P4_1(C4)
        P4_upSampled = self.P_upSample1(P4)

        P3 = self.P3_1(C3)
        P3 = P3 + P4_upSampled
        P3_upSampled = self.P_upSample2(P3)

        P2 = self.P2_1(C2)
        P2 = P2 + P3_upSampled

        P3 = P3 + self.P_downSample2(P2)
        P4 = P4 + self.P_downSample1(P3)
        multi_feature = (P2, P3, P4)
        pred_cls, pred_reg = self.multiPred(multi_feature)
        return pred_cls, pred_reg


class QualityFocalLossII(nn.Cell):
    def __init__(self, beta=2.0, loss_weight=1.0):
        super(QualityFocalLossII, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight
        self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = P.Sigmoid()
        self.pow = P.Pow()
        self.zeros = P.Zeros()
        self.logicalAnd = P.LogicalAnd()

    # pres = cls_score分类分支的输出[W*H, num_cls]
    # (labels, score) 中
    # labels是每个点对应的类别, 80代表负样本
    # score代表
    # 原始的score是用labels.shape构造的全是0的一维向量
    # 此时要对是正样本的地方进行score的修改!
    # 进行修改标注的内容竟然是iou！！！
    # 是预测框和gt_bbox之间的iou！！！！
    def construct(self, preds, targets: Tuple[Tensor], pos):
        label, score = targets
        pred_sigmoid = self.sigmoid(preds)
        scale_factor = pred_sigmoid
        zerolabel = self.zeros(preds.shape, ms.float32)
        loss = self.binary_cross_entropy_with_logits(preds, zerolabel) * self.pow(scale_factor, self.beta)
        bg_class_ind = preds.shape[1]
        # pos = np.nonzero(self.logicalAnd(label >= 0, label < bg_class_ind).asnumpy())
        # pos = np.nonzero((label >= 0) & (label < bg_class_ind))[0].squeeze()
        # pos = Tensor(pos)
        # pos_logic = self.logicalAnd(label >= 0, label < bg_class_ind)
        pos_label = label[pos]
        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
        loss[pos, pos_label] = self.binary_cross_entropy_with_logits(
            preds[pos, pos_label], score[pos]
        ) * self.pow(scale_factor.abs(), self.beta)
        loss = loss.sum(axis=1, keepdims=False)
        loss = self.loss_weight * loss
        return loss

class QualityFocalLossIII(nn.Cell):
    def __init__(self, beta=2.0, loss_weight=1.0):
        super(QualityFocalLossIII, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight
        self.sigmoid = P.Sigmoid()
        self.pow = P.Pow()
        self.zeros = P.Zeros()
        self.logicalAnd = P.LogicalAnd()
        self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.nonzero = P.MaskedSelect()
        self.reshape = P.Reshape()
        self.reduce_sum = P.ReduceSum()

    def construct(self, pred:ms.Tensor, label, score, pos):
        # label, score = target
        pred_sigmoid = self.sigmoid(pred)
        scale_factor = pred_sigmoid
        zerolabel = self.zeros(pred.shape, ms.float32)
        loss = self.binary_cross_entropy_with_logits(pred, zerolabel) * self.pow(scale_factor, self.beta)
        pos_label = label[pos]
        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
        loss[pos, pos_label] = self.binary_cross_entropy_with_logits(pred[pos, pos_label], score[pos]) * self.pow(scale_factor.abs(), self.beta)
        loss = self.reduce_sum(loss, axis=1)
        loss = loss * self.loss_weight
        loss = loss / len(pos)
        return loss


class DistributionFocalLossII(nn.Cell):
    def __init__(self, loss_weight=0.25):
        super(DistributionFocalLossII, self).__init__()
        self.loss_weight = loss_weight
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()

    # pred = pred_corners是正样本点上对应的pred_reg [正样本数量*4, self.reg_max + 1]
    # target = target_corners是正样本cells中心点和gt_bbox的中心点距离[l,t,r,b] [正样本数量, 4]
    def construct(self, pred, label, pos):
        dis_left = self.cast(label, ms.int32)
        dis_right = dis_left + 1
        weight_left = self.cast(dis_right, ms.float32) - label
        weight_right = label - self.cast(dis_left, ms.float32)
        dfl_loss = (
                self.cross_entropy(pred, dis_left) * weight_left
                + self.cross_entropy(pred, dis_right) * weight_right)
        dfl_loss = dfl_loss * self.loss_weight
        dfl_loss = dfl_loss / len(pos)
        return dfl_loss

# class DistributionFocalLossIII(nn.Cell):
#     def __init__(self, loss_weight=0.25):
#         super(DistributionFocalLossIII, self).__init__()
#         self.cross_entropy = nn.

class GIouLossII(nn.Cell):
    def __init__(self, eps=1e-6, reduction='mena', loss_weight=2.0):
        super(GIouLossII, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.maximum = P.Maximum()
        self.minimum = P.Minimum()
        self.eps = Tensor(eps, ms.float32)
        self.value_zero = Tensor(0, ms.float32)

    # boxes1 = pos_decode_bbox_pred
    # 用正样本cells的中心点, 还原预测的bbox的坐标,从[l,t,r,b]->[x1,y1,x2,y2]
    # 返回的东西是[正样本数量, 4] 其中4表示 [x1,y1,x2,y2] 表示通过[l,t,r,b]->[x1,y1,x2,y2]
    # 预测框的左上角与右下角坐标
    # boxes2 = pos_decode_bbox_targets为正样本gt_bbox [x1,y1,x2,y2]
    def construct(self, boxes1, boxes2):
        boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        left_up = self.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = self.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = self.maximum(right_down - left_up, self.value_zero)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1Area + boxes2Area - inter_area
        ious = self.maximum(1.0 * inter_area / union_area, self.eps)
        enclose_left_up = self.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = self.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = self.maximum(enclose_right_down - enclose_left_up, self.value_zero)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        gious_loss = ious - 1.0 * (enclose_area - union_area) / enclose_area
        gious_loss = 1 - gious_loss
        gious_loss = self.loss_weight * gious_loss
        return gious_loss


class Overlaps(nn.Cell):
    def __init__(self, eps=1e-6):
        super(Overlaps, self).__init__()
        self.eps = Tensor(eps, mstype.float32)
        self.value_zero = Tensor(0, mstype.float32)
        self.maximum = P.Maximum()
        self.minimum = P.Minimum()

    def construct(self, boxes1, boxes2):
        boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        left_up = self.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = self.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = self.maximum(right_down - left_up, self.value_zero)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1Area + boxes2Area - inter_area
        ious = self.maximum(1.0 * inter_area / union_area, self.eps)
        return ious

class NanoDetWithLossCell(nn.Cell):
    def __init__(self, network, config):
        super(NanoDetWithLossCell, self).__init__()
        self.network = network
        self.strides = config.strides
        self.reg_max = config.reg_max
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.less = P.Less()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.giou_loss = GIouLossII()
        self.qfl_loss = QualityFocalLossIII()
        self.dfs_loss = DistributionFocalLossII()
        self.integral = IntegralII(config)
        # self.zerosLike = P.ZerosLike()
        self.zeros = P.Zeros()
        self.distance2bbox = Distance2bbox()
        self.reshape = P.Reshape()
        self.bbox_overlaps = Overlaps()

    # def construct(self, x, gt_meta):
    def construct(self, x, pos_inds: Tensor, pos_grid_cell_center: Tensor, pos_decode_bbox_targets: Tensor,
                  target_corners: Tensor, assign_labels: Tensor):
        cls_scores, bbox_preds = self.network(x)
        cls_scores = cls_scores.reshape(-1, config.num_classes)
        bbox_preds = bbox_preds.reshape(-1, 4 * (config.reg_max + 1))

        pos_size = pos_inds.size
        pos_inds = pos_inds.squeeze()
        pos_grid_cell_center = pos_grid_cell_center.squeeze()
        pos_decode_bbox_targets = pos_decode_bbox_targets.squeeze()
        target_corners = target_corners.squeeze()
        assign_labels = assign_labels.squeeze()

        pos_bbox_pred = bbox_preds[pos_inds]
        pos_bbox_pred_corners = self.integral(pos_bbox_pred)
        pos_decode_bbox_pred = self.distance2bbox(pos_grid_cell_center, pos_bbox_pred_corners)
        pred_corners = self.reshape(pos_bbox_pred, (-1, config.reg_max + 1))
        # pred_corners = pos_bbox_pred.reshape(-1)
        # int64这里出现
        score = self.zeros(assign_labels.shape, ms.float32)
        temp = self.bbox_overlaps(pos_decode_bbox_pred, pos_decode_bbox_targets)
        score[pos_inds] = self.bbox_overlaps(pos_decode_bbox_pred, pos_decode_bbox_targets)
        # score[None][:, pos_inds] = self.bbox_overlaps(pos_decode_bbox_pred, pos_decode_bbox_targets)
        target = (assign_labels, score)
        giou_loss = self.reduce_sum(self.giou_loss(pos_decode_bbox_pred, pos_decode_bbox_targets)) // pos_size * 0.3
        dfs_loss = self.reduce_sum(self.dfs_loss(pred_corners, target_corners, pos_inds))
        # qfl_loss = self.reduce_sum(self.qfl_loss(cls_scores, target, pos_inds)) // pos_size * 0.2
        qfl_loss = self.reduce_sum(self.qfl_loss(cls_scores, assign_labels, score, pos_inds))
        loss = giou_loss + dfs_loss + qfl_loss
        return loss


class TrainingWrapper(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__()
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

if __name__ == "__main__":
    backbone = shuffleNet()
    a = time.time()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    backboneII = ShuffleNetV2II()
    x = Tensor(np.random.randint(0, 255, (1, 3, 320, 320)), mstype.float32)
    # outs = backbone(x)
    # outs_III = backbone(x)
    # outs_II = backboneII(x)
    nanodet = NanoDetII(backbone, config)
    out = nanodet(x)
    net = NanoDetWithLossCell(nanodet, config)

    class GeneratDefaultGridCells:
        def __init__(self):
            feature_size = [[40, 40], [20, 20], [10, 10]]
            steps = [8, 16, 32]
            anchor_size = np.array([8,16,32], np.float32)
            feature_size = np.array(feature_size)
            strides = np.array(steps)
            self.default_multi_level_grid_cells = []
            # config.feature_size = [[40, 40], [20, 20], [10, 10]]
            for idex, feature_size in enumerate(feature_size):
                base_size = anchor_size[idex] / 320
                stride = strides[idex]
                h, w = feature_size
                x_range = (np.arange(w) + 0.5) * stride
                y_range = (np.arange(h) + 0.5) * stride
                # x_range = np.arange(feature_size[0])
                # y_range = np.arange(feature_size[1])
                y_feat, x_feat = np.meshgrid(x_range, y_range)

                # y_feat = (y_feat + 0.5) / h
                # x_feat = (x_feat + 0.5) / h

                y_feat, x_feat = y_feat.flatten(), x_feat.flatten()
                grid_cells = np.stack(
                    [
                        x_feat - 0.5 * stride,
                        y_feat - 0.5 * stride,
                        x_feat + 0.5 * stride,
                        y_feat + 0.5 * stride
                    ],
                    axis=-1
                )
                # grid_cells = np.stack(
                #     [
                #         x_feat - base_size / 2,
                #         y_feat - base_size / 2,
                #         x_feat + base_size / 2,
                #         y_feat + base_size / 2
                #     ],
                #     axis=-1
                # )

                self.default_multi_level_grid_cells.append(grid_cells)


    # nanodet list
    default_multi_level_grid_cells = GeneratDefaultGridCells().default_multi_level_grid_cells
    num_level_cells_list = [grid_cells.shape[0] for grid_cells in default_multi_level_grid_cells]
    mlvl_grid_cells = np.concatenate(default_multi_level_grid_cells, axis=0, dtype=np.float32)
    y1, x1, y2, x2 = np.split(mlvl_grid_cells[:, :4], 4, axis=-1)
    vol_anchors = (x2 - x1) * (y2 - y1)


    def nanodet_bboxes_encode(boxes):

        def bbox_overlaps(bbox):
            ymin = np.maximum(y1, bbox[0])
            xmin = np.maximum(x1, bbox[1])
            ymax = np.minimum(y2, bbox[2])
            xmax = np.minimum(x2, bbox[3])
            # 并行化运算
            w = np.maximum(xmax - xmin, 0.)
            h = np.maximum(ymax - ymin, 0.)

            inter_vol = h * w
            union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
            iou = inter_vol / union_vol
            return np.squeeze(iou)

        def atssAssign(gt_bboxes, gt_labels, overlaps):
            INF = 100000
            num_gt = gt_bboxes.shape[0]
            num_grid_cells = mlvl_grid_cells.shape[0]
            assigned_gt_inds = np.full(shape=(num_grid_cells,), fill_value=0, dtype=np.int32)
            gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
            gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
            gt_point = np.stack((gt_cx, gt_cy), axis=1)

            grid_cells_cx = (mlvl_grid_cells[:, 0] + mlvl_grid_cells[:, 2]) / 2.0
            grid_cells_cy = (mlvl_grid_cells[:, 1] + mlvl_grid_cells[:, 3]) / 2.0
            grid_cells_points = np.stack((grid_cells_cx, grid_cells_cy), axis=1)

            distances = grid_cells_points[:, None, :] - gt_point[None, :, :]
            distances = np.power(distances, 2)
            distances = np.sum(distances, axis=-1)
            distances = np.sqrt(distances)

            candidate_idxs = []
            start_idx = 0
            topk = 9
            for level, cells_per_level in enumerate(num_level_cells_list):
                end_idx = start_idx + cells_per_level
                distances_per_level = distances[start_idx:end_idx, :]
                selectable_k = min(topk, cells_per_level)
                topk_idxs_per_level = np.argsort(distances_per_level, axis=0)
                topk_idxs_per_level = topk_idxs_per_level[:selectable_k]
                candidate_idxs.append(topk_idxs_per_level + start_idx)
                start_idx = end_idx

            candidate_idxs = np.concatenate(candidate_idxs, axis=0)

            candidate_overlaps = overlaps[candidate_idxs, np.arange(num_gt)]
            overlaps_mean_per_gt = np.mean(candidate_overlaps, axis=0)
            overlaps_std_per_gt = np.std(candidate_overlaps, axis=0)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

            for gt_idx in range(num_gt):
                candidate_idxs[:, gt_idx] += gt_idx * num_grid_cells
            ep_bboxes_cx = np.reshape(grid_cells_cx, (1, -1)).repeat(num_gt, axis=0).reshape(-1)
            ep_bboxes_cy = np.reshape(grid_cells_cy, (1, -1)).repeat(num_gt, axis=0).reshape(-1)
            candidate_idxs = candidate_idxs.reshape(-1)

            l_ = ep_bboxes_cx[candidate_idxs].reshape(-1, num_gt) - gt_bboxes[:, 0]
            t_ = ep_bboxes_cy[candidate_idxs].reshape(-1, num_gt) - gt_bboxes[:, 1]
            r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].reshape(-1, num_gt)
            b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].reshape(-1, num_gt)
            # is_in_gts = np.stack([l_, t_, r_, b_], axis=1).min(axis=1) > 0.01
            # is_pos = is_pos & is_in_gts

            overlaps_inf = np.full_like(overlaps, -INF).T.reshape(-1)
            index = candidate_idxs.reshape(-1)[is_pos.reshape(-1)]
            overlaps_inf[index] = overlaps.T.reshape(-1)[index]
            overlaps_inf = overlaps_inf.reshape(num_gt, -1).T

            max_overlaps = np.max(overlaps_inf, axis=1)
            argmax_overlaps = np.argmax(overlaps_inf, axis=1)

            assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

            if gt_labels is not None:
                assigned_labels = np.full_like(assigned_gt_inds, -1, dtype=np.int32)
                pos_inds = np.nonzero(assigned_gt_inds > 0)[0].squeeze()
                if pos_inds.size > 0:
                    assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
            else:
                assigned_labels = None

            return num_gt, assigned_gt_inds, max_overlaps, assigned_labels

        def sample(assigned_gt_inds, gt_bboxes):
            pos_inds = np.nonzero(assigned_gt_inds > 0)[0].squeeze()
            neg_inds = np.nonzero(assigned_gt_inds == 0)[0].squeeze()
            pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.reshape(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
            return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

        def target_assign_single_img(pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds, gt_labels):
            num_grid_cells = mlvl_grid_cells.shape[0]

            bbox_targets = np.zeros_like(mlvl_grid_cells, np.float32)
            bbox_weights = np.zeros_like(mlvl_grid_cells, np.float32)
            assign_labels = np.full((num_grid_cells,), 80, dtype=np.int32)
            assign_labels_weights = np.zeros((num_grid_cells,), dtype=np.float32)

            if len(pos_inds) > 0:
                pos_bbox_targets = pos_gt_bboxes
                bbox_targets[pos_inds, :] = pos_bbox_targets
                bbox_weights[pos_inds, :] = 1.0

                if gt_labels is None:
                    assign_labels[pos_inds] = 0
                else:
                    assign_labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
                assign_labels_weights[pos_inds] = 1.0

            if len(neg_inds) > 0:
                assign_labels_weights[neg_inds] = 1.0

            return assign_labels, assign_labels_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        def bbox2distance(points, bbox, max_dis=None, eps=0.1):
            left = points[:, 0] - bbox[:, 0]
            top = points[:, 1] - bbox[:, 1]
            right = bbox[:, 2] - points[:, 0]
            bottom = bbox[:, 3] - points[:, 1]
            if max_dis is not None:
                left = np.clip(left, 0, max_dis - eps)
                top = np.clip(top, 0, max_dis - eps)
                right = np.clip(right, 0, max_dis - eps)
                bottom = np.clip(bottom, 0, max_dis - eps)
            return np.stack([left, top, right, bottom], -1)

        def grid_cells_to_center(grid_cells):
            cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
            cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
            return np.stack([cells_cx, cells_cy], axis=-1)

        gt_labels = []
        gt_bboxes = []
        overlaps = []
        for bbox in boxes:
            gt_labels.append(bbox[4])
            label = int(bbox[4])
            gt_bboxes.append(bbox[:4])
            overlap = bbox_overlaps(bbox)
            overlaps.append(overlap)
        overlaps = np.stack(overlaps, axis=-1)
        gt_labels = np.array(gt_labels)
        gt_bboxes = np.stack(gt_bboxes, 0)

        num_gt, assigned_gt_inds, max_overlaps, assigned_labels = atssAssign(gt_bboxes, gt_labels, overlaps)
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = sample(assigned_gt_inds, gt_bboxes)
        assign_labels, assign_labels_weights, bbox_targets, bbox_weights, pos_inds, neg_inds = \
            target_assign_single_img(pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds, gt_labels)

        grid_cells = mlvl_grid_cells.reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        assign_labels = assign_labels.reshape(-1)
        assign_labels_weights = assign_labels_weights.reshape(-1)
        # bg_class_ind = config.num_classes
        bg_class_ind = 80

        pos_inds = np.nonzero((assign_labels >= 0) & (assign_labels < bg_class_ind))[0]
        score = np.zeros(assign_labels.shape)

        pos_bbox_targets = bbox_targets[pos_inds]
        pos_grid_cells = grid_cells[pos_inds]

        pos_grid_cell_centers = grid_cells_to_center(pos_grid_cells)
        target_corners = bbox2distance(pos_grid_cell_centers, pos_bbox_targets, 7).reshape(-1)

        return pos_inds, pos_grid_cell_centers, pos_bbox_targets, target_corners, assign_labels


    def intersect(box_a, box_b):
        """Compute the intersect of two sets of boxes."""
        max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
        min_yx = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]


    def jaccard_numpy(box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes."""
        inter = intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[2] - box_b[0]) *
                  (box_b[3] - box_b[1]))
        union = area_a + area_b - inter
        return inter / union


    boxes = np.array([[50, 70, 200, 200, 32], [40, 90, 170, 100, 1]], np.float32)
    # boxes[:, [0, 2]] = boxes[:, [0, 2]] / 320
    # boxes[:, [1, 3]] = boxes[:, [1, 3]] / 320
    # boxes = np.array([[50, 70, 200, 200, 32], [40, 90, 170, 100, 1]])
    pos_inds, pos_grid_cell_center, pos_decode_bbox_targets, target_corners, assign_labels = nanodet_bboxes_encode(
        boxes)
    # x, pos_inds: Tensor, pos_grid_cell_center: Tensor, pos_decode_bbox_targets: Tensor, target_corners: Tensor, assign_labels: Tensor
    loss = net(x, Tensor(pos_inds[None]), Tensor(pos_grid_cell_center[None]), Tensor(pos_decode_bbox_targets[None]),
               Tensor(target_corners[None]), Tensor(assign_labels[None]))
    b = time.time()
    print("时间耗时%ds" % (b - a))
