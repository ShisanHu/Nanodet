# check passed
import mindspore.nn as nn
from mindspore.common.initializer import initializer, HeNormal, HeUniform, Constant, XavierUniform, Normal

def kaiming_init(
    module:nn.Conv2d, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]

    if distribution == "uniform":
        uniform_init = HeUniform(negative_slope=a, mode=mode, nonlinearity=nonlinearity)
        module.weight_init = uniform_init

    else:
        normal_init = HeNormal(negative_slope=a, mode=mode,nonlinearity=nonlinearity)
        module.weight_init = normal_init

    if hasattr(module, "bias") and module.bias is not None:
        constant = Constant(value=bias)
        module.has_bias = True
        module.bias_init = constant

def xavier_init(module:nn.Conv2d, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]

    if distribution == "uniform":
        xavierUniform_init = XavierUniform(gain = gain)
        module.weight_init = xavierUniform_init
    else:
        # 未有相应算子
        # nn.init.xavier_normal_(module.weight, gain=gain)
        pass

    if hasattr(module, "bias") and module.bias is not None:
        constant = Constant(value=bias)
        module.has_bias = True
        module.bias_init = constant

def normal_init(module, mean=0, std=1, bias=0):
    normal = Normal(sigma=std , mean=mean)
    module.weight_init = normal
    if hasattr(module, "bias") and module.bias is not None:
        constant = Constant(value=bias)
        module.has_bias = True
        module.bias_init = constant

def constant_init(module, val, bias=0):

    if hasattr(module, "weight") and module.weight is not None:
        constant = Constant(value=val)
        module.weight_init = constant

    if hasattr(module, "bias") and module.bias is not None:
        constant = Constant(value=bias)
        module.has_bias = True
        module.bias_init = constant

