import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore import Parameter
from src.model_utils.config import config


class DepthwiseConvModule(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
    ):
        super(DepthwiseConvModule, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            group=in_channels,
            has_bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, has_bias=False)
        self.dwnorm = nn.BatchNorm2d(in_channels)
        self.pwnorm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.init_weights()

    def construct(self, x):
        x = self.depthwise(x)
        x = self.dwnorm(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pwnorm(x)
        x = self.act(x)
        return x

    def init_weights(self):
        pass


class IntegralII(nn.Cell):
    def __init__(self, config):
        super(IntegralII, self).__init__()
        self.reg_max = config.reg_max
        self.softmax = P.Softmax(axis=-1)
        self.start = Tensor(0, mstype.float32)
        self.stop = Tensor(config.reg_max)
        linspace = Tensor([0, 1, 2, 3, 4, 5, 6, 7], mstype.float32)
        project = Parameter(default_input=linspace, name="project", requires_grad=False)
        self.distribution_project = nn.Dense(8, 1, weight_init=project, has_bias=False)
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, x):
        shape = self.shape(x)
        x = self.softmax(x.reshape(*shape[:-1], 4, 8))
        x = self.distribution_project(x).reshape(*shape[:-1], 4)
        return x

class Integral(nn.Cell):
    def __init__(self):
        super(Integral, self).__init__()
        self.softmax = P.Softmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.linspace = Tensor([[0, 1, 2, 3, 4, 5, 6, 7]], mstype.float32)
        self.matmul = P.MatMul(transpose_b=True)

    def construct(self, x):
        x = self.softmax(x.reshape(-1, 8))
        x = self.matmul(x, self.linspace).reshape(-1, 4)
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
        if self.max_shape is not None:
            x1 = C.clip_by_value(x1, Tensor(0), Tensor(self.max_shape[0]))
            y1 = C.clip_by_value(y1, Tensor(0), Tensor(self.max_shape[1]))
            x2 = C.clip_by_value(x2, Tensor(0), Tensor(self.max_shape[0]))
            y2 = C.clip_by_value(y2, Tensor(0), Tensor(self.max_shape[0]))
        return self.stack([x1, y1, x2, y2])


class ShuffleV2Block(nn.Cell):
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        branch_features = oup // 2
        if self.stride > 1:
            self.branch1 = nn.SequentialCell([
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, has_bias=False),
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
        return nn.Conv2d(i, o, kernel_size, stride, "pad", padding, group=i, has_bias=bias)

    def construct(self, x):
        if self.stride == 1:
            x1, x2 = P.Split(axis=1, output_num=2)(x)
            out = P.Concat(axis=1)((x1, self.branch2(x2)))
        else:
            out = P.Concat(axis=1)((self.branch1(x), self.branch2(x)))
        out = channel_shuffle(out, 2)
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    x = P.Reshape()(x, (batchsize, groups, channels_per_group, height, width))
    x = P.Transpose()(x, (0, 2, 1, 3, 4))
    x = P.Reshape()(x, (batchsize, -1, height, width))
    return x


class ShuffleNetV2(nn.Cell):
    def __init__(self, model_size='1.0x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)
        self.stage_repeats = [4, 8, 4]
        self.out_stages = (2, 3, 4)

        if model_size == "0.5x":
            _stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            _stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            _stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            _stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        output_channels = _stage_out_channels[0]

        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=output_channels, kernel_size=3, stride=2,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(num_features=output_channels, momentum=0.9),
            nn.ReLU(),
            # nn.LeakyReLU(),
        ])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        input_channels = output_channels
        output_channels = _stage_out_channels[1]
        self.stage2 = nn.SequentialCell([
            ShuffleV2Block(input_channels, output_channels, 2),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
        ])
        input_channels = output_channels
        output_channels = _stage_out_channels[2]
        self.stage3 = nn.SequentialCell([
            ShuffleV2Block(input_channels, output_channels, 2),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
        ])
        input_channels = output_channels
        output_channels = _stage_out_channels[3]
        self.stage4 = nn.SequentialCell([
            ShuffleV2Block(input_channels, output_channels, 2),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
            ShuffleV2Block(output_channels, output_channels, 1),
        ])

    def construct(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        C2 = self.stage2(x)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        return C2, C3, C4


def shuffleNet(model_size='1.0x'):
    return ShuffleNetV2(model_size=model_size)


class NanoDet(nn.Cell):
    def __init__(self, backbone, config, is_training=True):
        super(NanoDet, self).__init__()
        self.backbone = backbone
        feature_size = config.feature_size
        self.strides = [8, 16, 32]
        self.ConvModule = DepthwiseConvModule
        self.lateral_convs = nn.CellList()
        self.lateral_convs.append(nn.Conv2d(116, 96, kernel_size=1, stride=1, pad_mode='same', has_bias=True))
        self.lateral_convs.append(nn.Conv2d(232, 96, kernel_size=1, stride=1, pad_mode='same', has_bias=True))
        self.lateral_convs.append(nn.Conv2d(464, 96, kernel_size=1, stride=1, pad_mode='same', has_bias=True))
        self.P_upSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]))
        self.P_upSample2 = P.ResizeBilinear((feature_size[0], feature_size[0]))
        self.P_downSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]))
        self.P_downSample2 = P.ResizeBilinear((feature_size[2], feature_size[2]))
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=2)
        self.transpose = P.Transpose()
        self.slice = P.Slice()
        self._make_layer()

    def _build_shared_head(self):
        cls_convs = nn.SequentialCell()
        # reg_convs = nn.CellList()
        for i in range(2):
            cls_convs.append(
                self.ConvModule(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        return cls_convs

    def _make_layer(self):
        self.cls_convs = nn.CellList()
        # self.reg_convs = nn.CellList()
        for _ in self.strides:
            # cls_convs, reg_convs = self._build_shared_head()
            cls_convs = self._build_shared_head()
            self.cls_convs.append(cls_convs)
            # self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.CellList()
        self.gfl_reg = nn.CellList()
        for _ in self.strides:
            self.gfl_cls.append(
                nn.Conv2d(
                    in_channels=96,
                    out_channels=112,
                    kernel_size=1,
                    padding=0,
                    has_bias=True,
                )
            )
        for _ in self.strides:
            self.gfl_reg.append(
                nn.Conv2d(
                    in_channels=96,
                    out_channels=32,
                    kernel_size=1,
                    padding=0,
                    has_bias=True,
                )
            )

    def construct(self, inputs):
        C2, C3, C4 = self.backbone(inputs)
        # 对齐通道
        P4 = self.lateral_convs[2](C4)
        P3 = self.lateral_convs[1](C3)
        P2 = self.lateral_convs[0](C2)
        # top -> down
        P3 = self.P_upSample1(P4) + P3
        P2 = self.P_upSample2(P3) + P2
        # down -> top
        P3 = self.P_downSample1(P2) + P3
        P4 = self.P_downSample2(P3) + P4
        # s = self.cls_convs[0](P4)
        P4 = self.cls_convs[2](P4)
        P3 = self.cls_convs[1](P3)
        P2 = self.cls_convs[0](P2)

        P4 = self.gfl_cls[2](P4)
        P3 = self.gfl_cls[1](P3)
        P2 = self.gfl_cls[0](P2)

        P4 = self.reshape(P4, (-1, 112, 100))
        P3 = self.reshape(P3, (-1, 112, 400))
        P2 = self.reshape(P2, (-1, 112, 1600))
        preds = self.concat((P2, P3, P4))
        preds = self.transpose(preds, (0, 2, 1))

        cls_convs = self.slice(preds, (0, 0, 0), (-1, -1, 80))
        reg_convs = self.slice(preds, (0, 0, 80), (-1, -1, -1))
        return cls_convs, reg_convs

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

    def construct(self, pred: ms.Tensor, label, score, pos):
        # label, score = target
        pred_sigmoid = self.sigmoid(pred)
        scale_factor = pred_sigmoid
        zerolabel = self.zeros(pred.shape, ms.float32)
        loss = self.binary_cross_entropy_with_logits(pred, zerolabel) * self.pow(scale_factor, self.beta)
        pos_label = label[pos]
        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
        loss[pos, pos_label] = self.binary_cross_entropy_with_logits(pred[pos, pos_label], score[pos]) * self.pow(
            scale_factor.abs(), self.beta)
        loss = self.reduce_sum(loss, 1)
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

    def construct(self, pred, label, pos):
        dis_left = self.cast(label, ms.int32)
        dis_right = dis_left + 1
        weight_left = self.cast(dis_right, ms.float32) - label
        weight_right = label - self.cast(dis_left, ms.float32)
        dfl_loss = (
                self.cross_entropy(pred, dis_left) * weight_left
                + self.cross_entropy(pred, dis_right) * weight_right)
        dfl_loss = dfl_loss * self.loss_weight
        return dfl_loss


class GIouLossII(nn.Cell):
    def __init__(self, eps=1e-6, reduction='mena', loss_weight=2.0):
        super(GIouLossII, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.maximum = P.Maximum()
        self.minimum = P.Minimum()
        self.eps = Tensor(eps, ms.float32)
        self.value_zero = Tensor(0, ms.float32)

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
        self.integral = Integral()
        self.zeros = P.Zeros()
        self.distance2bbox = Distance2bbox()
        self.reshape = P.Reshape()
        self.bbox_overlaps = Overlaps()
        self.sigmoid = P.Sigmoid()
        self.max = P.ArgMaxWithValue(axis=1)

    # def construct(self, x, gt_meta):
    def construct(self, x, pos_inds: Tensor, pos_grid_cell_center: Tensor, pos_decode_bbox_targets: Tensor,
                  target_corners: Tensor, assign_labels: Tensor):
        cls_scores, bbox_preds = self.network(x)
        cls_scores = cls_scores.reshape(-1, 80)
        bbox_preds = bbox_preds.reshape(-1, 4 * (self.reg_max + 1))

        weight_targets = self.sigmoid(cls_scores)
        weight_targets = self.max(weight_targets)[1][pos_inds]

        pos_size = pos_inds.size
        pos_inds = pos_inds.squeeze()
        pos_grid_cell_center = pos_grid_cell_center.squeeze()
        pos_decode_bbox_targets = pos_decode_bbox_targets.squeeze()
        target_corners = target_corners.squeeze()
        assign_labels = assign_labels.squeeze()

        pos_bbox_pred = bbox_preds[pos_inds]
        pos_bbox_pred_corners = self.integral(pos_bbox_pred)
        pos_decode_bbox_pred = self.distance2bbox(pos_grid_cell_center, pos_bbox_pred_corners)
        pred_corners = self.reshape(pos_bbox_pred, (-1, self.reg_max + 1))
        # pred_corners = pos_bbox_pred.reshape(-1)
        # int64这里出现
        score = self.zeros(assign_labels.shape, ms.float32)
        temp = self.bbox_overlaps(pos_decode_bbox_pred, pos_decode_bbox_targets)
        score[pos_inds] = self.bbox_overlaps(pos_decode_bbox_pred, pos_decode_bbox_targets)
        # score[None][:, pos_inds] = self.bbox_overlaps(pos_decode_bbox_pred, pos_decode_bbox_targets)
        target = (assign_labels, score)
        giou_loss = self.reduce_sum(self.giou_loss(pos_decode_bbox_pred, pos_decode_bbox_targets))
        dfs_loss = self.reduce_sum(self.dfs_loss(pred_corners, target_corners, pos_inds))
        qfl_loss = self.reduce_sum(self.qfl_loss(cls_scores, assign_labels, score, pos_inds))
        loss = giou_loss + dfs_loss + qfl_loss
        return loss


class NanoDetWithLossCellII(nn.Cell):
    def __init__(self, network):
        super(NanoDetWithLossCellII, self).__init__()
        self.network = network

    # def construct(self, x, gt_meta):
    def construct(self, x):
        cls_scores, bbox_preds = self.network(x)
        return cls_scores, bbox_preds



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
    from mindspore.train.serialization import load_checkpoint, load_param_into_net
    import cv2 as cv

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    img = cv.imread('../000000007088.jpg')
    img = cv.resize(img, (320, 320))
    transpose = P.Transpose()
    img = Tensor(img, mstype.float32)
    # img = transpose(img,(2,0,1)).reshape(1,3,320,320)
    img = img.reshape(1, 3, 320, 320)

    # print(img[..., 2].shape)
    backbone = shuffleNet()
    nanodet = NanoDet(backbone,config)
    net = NanoDetWithLossCellII(nanodet)

    param_dict = load_checkpoint('../checkpoint.ckpt')
    # net = NanoDet(backbone, config)
    #
    # for item in net.parameters_and_names():
    #     print(item)
    # net.init_parameters_data()
    load_param_into_net(net, param_dict)
    # for item in net.parameters_and_names():
    #     # out = Tensor(item)
    #     # print(out.shape)
    #     print(item[1])
    net.set_train(False)
    cls_scores, bbox_preds = net(img)

    topk = P.TopK()
    integral = Integral()
    _, ind = topk(cls_scores,1)
    temp = ind.reshape(-1)
    bbox = integral(bbox_preds)


