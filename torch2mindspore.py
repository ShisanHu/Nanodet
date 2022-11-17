import torch
import mindspore
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

def pytorch2mindspore(ckpt_name='nanodet_m.ckpt'):
    par_dict = torch.load(ckpt_name, map_location=torch.device('cpu'))
    weights = []
    for k, v in par_dict['state_dict'].items():
        print(k)
        # print(v)
        param_dict = {}
    #     param_name = k.replace('model.', 'network.')
    #
    #     param_name = param_name.replace('running_mean', 'moving_mean')
    #     param_name = param_name.replace('running_var', 'moving_variance')
    #
    #     param_name = param_name.replace('.fpn.', '.')
    #     param_name = param_name.replace('.conv.', '.')
    #     param_name = param_name.replace('.head.', '.')
    #
    #     param_name = param_name.replace('.conv.', '.')
    #     param_name = param_name.replace('.head.', '.')
    #
    #     param_name = param_name.replace('weight', 'gamma')
    #     param_name = param_name.replace('bias', 'beta')
    #
    #     param_name = param_name.replace('backbone.conv1.0.gamma', 'backbone.conv1.0.weight')
    #     param_name = param_name.replace('backbone.stage2.0.branch1.0.gamma', 'backbone.stage2.0.branch1.0.weight')
    #     param_name = param_name.replace('backbone.stage2.0.branch1.2.gamma', 'backbone.stage2.0.branch1.2.weight')
    #     param_name = param_name.replace('backbone.stage2.0.branch2.0.gamma', 'backbone.stage2.0.branch2.0.weight')
    #     param_name = param_name.replace('backbone.stage2.0.branch2.3.gamma', 'backbone.stage2.0.branch2.3.weight')
    #     param_name = param_name.replace('backbone.stage2.0.branch2.5.gamma', 'backbone.stage2.0.branch2.5.weight')
    #     param_name = param_name.replace('bias', 'beta')
    #     param_name = param_name.replace('bias', 'beta')
    #     param_name = param_name.replace('bias', 'beta')
    #     param_name = param_name.replace('bias', 'beta')
    #     param_name = param_name.replace('bias', 'beta')
    #     param_name = param_name.replace('bias', 'beta')
    #     param_name = param_name.replace('bias', 'beta')
    #     param_name = param_name.replace('bias', 'beta')
    #     param_name = param_name.replace('bias', 'beta')
    #     param_name = param_name.replace('bias', 'beta')
    #
    #
    #
    #     param_dict['name'] = param_name
    #     param_dict['data'] = Tensor(v.numpy())
    #
    #     weights.append(param_dict)
    #
    # save_checkpoint(weights, "checkpoint.ckpt")


if __name__ == "__main__":
    pytorch2mindspore()
    # par_dict = mindspore.load_checkpoint('checkpoint.ckpt')
    # for item in par_dict.items():
    #     print(item)