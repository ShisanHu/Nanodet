import mindspore.nn as nn
import mindspore as ms
import mindspore.ops.operations as P
from mindspore.ops import composite as C
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from src.nanodetII import shuffleNet, NanoDet, NanoDetWithLossCellII
from src.model_utils.config import config

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

def get_single_level_center_point(featmap_size, stride):
    h, w = featmap_size
    x_range = (np.arange(w)+0.5) * stride
    y_range = (np.arange(h)+0.5) * stride
    y_feat, x_feat = np.meshgrid(x_range, y_range)
    y_feat, x_feat = y_feat.flatten(), x_feat.flatten()
    return Tensor(y_feat, dtype=mstype.float32), Tensor(x_feat, dtype=mstype.float32)


def apply_nms(cls_preds:Tensor, reg_preds:Tensor):
    # B = cls_preds.shape[0]
    input_shape = (320, 320)
    featmap_sizes = [(40, 40), (20, 20), (10, 10)]
    strides = [8, 16, 32]
    mlvl_center_priors = []
    distribution_project = Integral()
    distance2bbox = Distance2bbox(max_shape=input_shape)
    for i, stride in enumerate(strides):
        y, x = get_single_level_center_point(featmap_sizes[i], stride)
        step = P.Fill()(mstype.float32, (x.shape[0],), stride)
        proiors = P.Stack(axis=-1)([x, y, step, step])
        mlvl_center_priors.append(P.ExpandDims()(proiors, 0))
    center_priors = P.Concat(axis=1)(mlvl_center_priors)
    dis_preds = distribution_project(reg_preds) * center_priors[..., 2, None]
    bboxes = distance2bbox(center_priors[..., :2], dis_preds)
    scores = P.Sigmoid()(cls_preds)

    score, bbox = scores, bboxes
    padding = P.Zeros()((score.shape[0], 1), mstype.float32)
    score = P.Concat(axis=1)([score, padding])


if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    x = Tensor(np.random.randint(0, 255, (1, 3, 320, 320)), mstype.float32)
    backbone = shuffleNet()
    nanodet = NanoDet(backbone,config)
    net = NanoDetWithLossCellII(nanodet)

    cls_preds, reg_preds = net(x)
    apply_nms(cls_preds, reg_preds)