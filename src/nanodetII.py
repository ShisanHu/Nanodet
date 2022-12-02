import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
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
        # self.act = nn.ReLU()
        self.act = nn.LeakyReLU(alpha=0.1)
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


class Integral(nn.Cell):
    def __init__(self):
        super(Integral, self).__init__()
        self.softmax = P.Softmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.linspace = Tensor([[0, 1, 2, 3, 4, 5, 6, 7]], mstype.float32)
        self.matmul = P.MatMul(transpose_b=True)

    def construct(self, x):
        x_shape = self.shape(x)
        x = self.reshape(x, (-1, 8))
        x = self.softmax(x)
        x = self.matmul(x, self.linspace)
        out_shape = x_shape[:-1] + (4,)
        x = self.reshape(x, out_shape)
        return x


class Distance2bbox(nn.Cell):
    def __init__(self):
        super(Distance2bbox, self).__init__()
        self.stack = P.Stack(-1)

    def construct(self, points, distance):
        y1 = points[..., 0] - distance[..., 1]
        x1 = points[..., 1] - distance[..., 0]
        y2 = points[..., 0] + distance[..., 3]
        x2 = points[..., 1] + distance[..., 2]
        return self.stack([y1, x1, y2, x2])


class BBox2Distance(nn.Cell):
    def __init__(self):
        super(BBox2Distance, self).__init__()
        self.stack = P.Stack(-1)

    def construct(self, points, bbox):
        left = points[..., 1] - bbox[..., 1]
        top = points[..., 0] - bbox[..., 0]
        right = bbox[..., 3] - points[..., 1]
        bottom = bbox[..., 2] - points[..., 0]
        left = C.clip_by_value(left, Tensor(0.0), Tensor(6.9))
        top = C.clip_by_value(top, Tensor(0.0), Tensor(6.9))
        right = C.clip_by_value(right, Tensor(0.0), Tensor(6.9))
        bottom = C.clip_by_value(bottom, Tensor(0.0), Tensor(6.9))
        return self.stack((left, top, right, bottom))


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
                # nn.ReLU(),
                nn.LeakyReLU(alpha=0.1),
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
            # nn.ReLU(),
            nn.LeakyReLU(alpha=0.1),
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
            # nn.ReLU(),
            nn.LeakyReLU(alpha=0.1),
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
            # nn.ReLU(),
            nn.LeakyReLU(alpha=0.1),
        ])

        self.pad = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), "CONSTANT")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

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
        self.tensor_summary = P.TensorSummary()

    def construct(self, x):
        self.tensor_summary("img_tensor", x)
        x = self.conv1(x)
        self.tensor_summary("backbone_conv1", x)
        x = self.pad(x)
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
        self.P_upSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]), half_pixel_centers=True)
        self.P_upSample2 = P.ResizeBilinear((feature_size[0], feature_size[0]), half_pixel_centers=True)
        self.P_downSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]), half_pixel_centers=True)
        self.P_downSample2 = P.ResizeBilinear((feature_size[2], feature_size[2]), half_pixel_centers=True)
        # self.P_upSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]))
        # self.P_upSample2 = P.ResizeBilinear((feature_size[0], feature_size[0]))
        # self.P_downSample1 = P.ResizeBilinear((feature_size[1], feature_size[1]))
        # self.P_downSample2 = P.ResizeBilinear((feature_size[2], feature_size[2]))
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=2)
        self.transpose = P.Transpose()
        self.slice = P.Slice()
        self._make_layer()
        self.tensor_summary = P.TensorSummary()

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
        self.tensor_summary("C3", C3)
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
        # P2, P3, P4 = inputs
        P2 = self.cls_convs[0](P2)
        P3 = self.cls_convs[1](P3)
        P4 = self.cls_convs[2](P4)

        P4 = self.gfl_cls[2](P4)
        P3 = self.gfl_cls[1](P3)
        P2 = self.gfl_cls[0](P2)

        self.tensor_summary("P3", P3)

        P4 = self.reshape(P4, (-1, 112, 100))
        P3 = self.reshape(P3, (-1, 112, 400))
        P2 = self.reshape(P2, (-1, 112, 1600))
        preds = self.concat((P2, P3, P4))
        preds = self.transpose(preds, (0, 2, 1))

        cls_scores = self.slice(preds, (0, 0, 0), (-1, -1, 80))
        bbox_preds = self.slice(preds, (0, 0, 80), (-1, -1, -1))
        return cls_scores, bbox_preds


class QualityFocalLoss(nn.Cell):
    def __init__(self, beta=2.0):
        super(QualityFocalLoss, self).__init__()
        self.sigmoid = P.Sigmoid()
        self.sigmiod_cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.pow = P.Pow()
        self.abs = P.Abs()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.beta = beta

    def construct(self, logits, label, score):
        # print(logits[0])
        logits_sigmoid = self.sigmoid(logits)
        label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        score = self.tile(self.expand_dims(score, -1), (1, 1, F.shape(logits)[-1]))
        label = label * score

        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        modulating_factor = self.pow(self.abs(label - logits_sigmoid), self.beta)
        qfl_loss = sigmiod_cross_entropy * modulating_factor
        return qfl_loss


class DistributionFocalLoss(nn.Cell):
    def __init__(self):
        super(DistributionFocalLoss, self).__init__()
        # self.loss_weight = loss_weight
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()

    def construct(self, pred, label):
        dis_left = self.cast(label, mstype.int32)
        dis_right = dis_left + 1
        weight_left = self.cast(dis_right, mstype.float32) - label
        weight_right = label - self.cast(dis_left, mstype.float32)
        dfl_loss = (
                self.cross_entropy(pred, dis_left) * weight_left
                + self.cross_entropy(pred, dis_right) * weight_right)
        # dfl_loss = dfl_loss * self.loss_weight
        return dfl_loss


class GIou(nn.Cell):
    """Calculating giou"""

    def __init__(self):
        super(GIou, self).__init__()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.concat = P.Concat(axis=1)
        self.mean = P.ReduceMean()
        self.div = P.RealDiv()
        self.eps = 0.000001

    def construct(self, box_p, box_gt):
        """construct method"""
        box_p_area = (box_p[..., 3:4] - box_p[..., 1:2]) * (box_p[..., 2:3] - box_p[..., 0:1])
        box_gt_area = (box_gt[..., 3:4] - box_gt[..., 1:2]) * (box_gt[..., 2:3] - box_gt[..., 0:1])
        y_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        x_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        x_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        yc_1 = self.min(box_p[..., 0:1], box_gt[..., 0:1])
        yc_2 = self.max(box_p[..., 2:3], box_gt[..., 2:3])
        xc_1 = self.min(box_p[..., 1:2], box_gt[..., 1:2])
        xc_2 = self.max(box_p[..., 3:4], box_gt[..., 3:4])
        c_area = (xc_2 - xc_1) * (yc_2 - yc_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        c_area = c_area + self.eps
        iou = self.div(self.cast(intersection, ms.float32), self.cast(union, ms.float32))
        res_mid0 = c_area - union
        res_mid1 = self.div(self.cast(res_mid0, ms.float32), self.cast(c_area, ms.float32))
        giou = iou - res_mid1
        giou = 1 - giou
        giou = C.clip_by_value(giou, -1.0, 1.0)
        giou = giou.squeeze(-1)
        return giou


class Iou(nn.Cell):
    def __init__(self):
        super(Iou, self).__init__()
        self.cast = P.Cast()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.div = P.RealDiv()
        self.eps = 0.000001

    def construct(self, box_p, box_gt):
        """construct method"""
        box_p_area = (box_p[..., 3:4] - box_p[..., 1:2]) * (box_p[..., 2:3] - box_p[..., 0:1])
        box_gt_area = (box_gt[..., 3:4] - box_gt[..., 1:2]) * (box_gt[..., 2:3] - box_gt[..., 0:1])
        y_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        x_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        x_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        iou = self.div(self.cast(intersection, ms.float32), self.cast(union, ms.float32))
        iou = iou.squeeze(-1)
        return iou


class NanoDetWithLossCell(nn.Cell):
    def __init__(self, network):
        super(NanoDetWithLossCell, self).__init__()
        self.network = network
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.less = P.Less()
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.zeros = P.Zeros()
        self.reshape = P.Reshape()
        self.sigmoid = P.Sigmoid()
        self.ones = P.Ones()
        self.iou = Iou()
        self.loss_bbox = GIou()
        self.loss_qfl = QualityFocalLoss()
        self.loss_dfl = DistributionFocalLoss()
        self.integral = Integral()
        self.distance2bbox = Distance2bbox()
        self.bbox2distance = BBox2Distance()

    # def construct(self, x, gt_meta):
    def construct(self, x, res_boxes, res_labels, res_center_priors, nums_match):
        cls_scores, bbox_preds = self.network(x)
        b = cls_scores.shape[0]
        cls_scores = self.cast(cls_scores, mstype.float32)
        bbox_preds = self.cast(bbox_preds, mstype.float32)

        mask = self.cast(self.less(-1, res_labels), mstype.float32)
        nums_match = self.reduce_sum(self.cast(nums_match, mstype.float32))
        mask_bbox = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        mask_bbox_preds = self.tile(self.expand_dims(mask, -1), (1, 1, 32))

        mask_bbox_pred_corners = self.integral(bbox_preds) * mask_bbox
        mask_grid_cell_centers = (res_center_priors / self.tile(
            self.expand_dims(res_center_priors[..., 2], -1), (1, 1, 4))) * mask_bbox
        mask_decode_bbox = res_boxes / self.tile(
            self.expand_dims(res_center_priors[..., 2], -1), (1, 1, 4))
        decode_bbox_pred = self.distance2bbox(mask_grid_cell_centers, mask_bbox_pred_corners)

        # loss_bbox
        loss_bbox = self.loss_bbox(decode_bbox_pred, mask_decode_bbox) * mask
        loss_bbox = self.reduce_sum(loss_bbox, -1)
        # loss_qfl
        score = self.ones(F.shape(res_labels), mstype.float32)
        score = score * self.iou(decode_bbox_pred, mask_decode_bbox) * mask
        loss_qfl = self.loss_qfl(cls_scores, res_labels, score)
        loss_qfl = self.reduce_sum(self.reduce_mean(loss_qfl, -1), -1)
        # loss_dfl
        pred_corners = self.reshape(bbox_preds * mask_bbox_preds, (-1, 8))
        target_corners = self.reshape(self.bbox2distance(mask_grid_cell_centers, mask_decode_bbox), (-1,))
        loss_dfl = self.loss_dfl(pred_corners, target_corners) * mask_bbox.reshape(-1)
        loss_dfl = self.reshape(loss_dfl, (-1, 2100, 4))
        loss_dfl = self.reduce_sum(self.reduce_mean(loss_dfl, -1), -1)
        loss = self.reduce_sum((loss_qfl + 2.0 * loss_bbox + 0.25 * loss_dfl) / nums_match)
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


class NanodetInferWithDecoder(nn.Cell):
    def __init__(self, network, center_priors, config):
        super(NanodetInferWithDecoder, self).__init__()
        self.network = network
        # self.distance2bbox = Distance2bbox(config.img_shape)
        self.distribution_project = Integral()
        self.center_priors = center_priors
        self.sigmoid = P.Sigmoid()
        self.expandDim = P.ExpandDims()
        self.tile = P.Tile()
        self.shape = P.Shape()
        self.stack = P.Stack(-1)

    def construct(self, x, max_shape):
        x_shape = self.shape(x)
        default_priors = self.expandDim(self.center_priors, 0)
        cls_preds, reg_preds = self.network(x)
        dis_preds = self.distribution_project(reg_preds) * self.tile(self.expandDim(default_priors[..., 2], -1),
                                                                     (1, 1, 4))
        bboxes = self.distance2bbox(default_priors[..., :2], dis_preds, max_shape)
        scores = self.sigmoid(cls_preds)
        # bboxes = self.tile(self.expandDim(bboxes, -2), (1, 1, 80, 1))
        return bboxes, scores

    def distance2bbox(self, points, distance, max_shape=None):
        y1 = points[..., 0] - distance[..., 1]
        x1 = points[..., 1] - distance[..., 0]
        y2 = points[..., 0] + distance[..., 3]
        x2 = points[..., 1] + distance[..., 2]
        if self.max_shape is not None:
            y1 = C.clip_by_value(y1, Tensor(0), Tensor(self.max_shape[0]))
            x1 = C.clip_by_value(x1, Tensor(0), Tensor(self.max_shape[1]))
            y2 = C.clip_by_value(y2, Tensor(0), Tensor(self.max_shape[0]))
            x2 = C.clip_by_value(x2, Tensor(0), Tensor(self.max_shape[1]))
        return self.stack([y1, x1, y2, x2])

# if __name__ == "__main__":
#     from mindspore.train.serialization import load_checkpoint, load_param_into_net
#     import cv2 as cv
#     import numpy as np
#
#     ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
#     img = cv.imread('../000000007088.jpg')
#     img = cv.resize(img, (320, 320))
#     transpose = P.Transpose()
#     img = Tensor(img, mstype.float32)
#     # img = transpose(img,(2,0,1)).reshape(1,3,320,320)
#     img = img.reshape(1, 3, 320, 320)
#
#     # print(img[..., 2].shape)
#     backbone = shuffleNet()
#     nanodet = NanoDet(backbone, config)
#     net = NanoDetWithLossCell(nanodet)
#     res_center_priors = Tensor(np.random.rand(1,2100, 4), mstype.float32)
#     # x, res_boxes, res_corners, res_labels, res_center_priors, nums_match
#     res_boxes = Tensor(np.random.rand(1, 2100, 4), mstype.float32)
#     res_labels = Tensor(np.random.randint(0,80,(1,2100)), mstype.int32)
#     nums_match = Tensor([24], mstype.float32)
#     loss = net(img, res_boxes, res_labels, res_center_priors, nums_match)
#
#
#
#     # net.set_train(False)
#     # # infor = NanodetInferWithDecoder(nanodet, center_priors, config)
#     # # infor(img)
#     # param_dict = load_checkpoint('../checkpoint.ckpt')
#     # # net = NanoDet(backbone, config)
#     # net.init_parameters_data()
#     # load_param_into_net(net, param_dict)
#     # # for item in net.parameters_and_names():
#     # #     out = Tensor(item[1])
#     # #     print(out)
#     # # import numpy as np
#     # # np.random.seed(1)
#     # # C2 = Tensor(np.random.rand(1, 96, 40, 40), mstype.float32)
#     # # C3 = Tensor(np.random.rand(1, 96, 20, 20), mstype.float32)
#     # # C4 = Tensor(np.random.rand(1, 96, 10, 10), mstype.float32)
#     # # inputs = (C2, C3, C4)
#     #
#     # cls_scores, bbox_preds = net(img)
#     # # out = net(inputs)
#     # # print(out[0])
#     # for item in net.parameters_and_names():
#     #     print(item[0])
#     # # net.init_parameters_data()
#     # # for item in net.parameters_and_names():
#     # #     # out = Tensor(item)
#     # #     # print(out.shape)
#     # #     print(item[1])
#     #
#     # # infor = NanodetInferWithDecoder(net, center_priors, config)
#     # # infor(img)
#     # topk = P.TopK()
#     # integral = Integral()
#     # # _, ind = topk(cls_scores,1)
#     # # temp = ind.reshape(-1)
#     # # bbox = integral(bbox_preds)
