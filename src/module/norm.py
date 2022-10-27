# import torch.nn as nn
import mindspore.nn as nn

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "SyncBN": ("bn", nn.SyncBatchNorm),
    "GN": ("gn", nn.GroupNorm),
    # and potentially 'SN'
}

def build_norm_layer(cfg, num_features, postfix=""):
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop("type")
    print(layer_type)

    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)

    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        if layer_type == "SyncBN" and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.get_parameters():
        param.requires_grad = requires_grad

    return name, layer
