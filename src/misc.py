from functools import partial
import mindspore.ops.operations as P

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def images_to_levels(target, num_level_anchors):
    stack = P.Stack(axis=0)
    target = stack(target)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0) if target[:, start:end].shape[0] == 1 else target[:, start:end])
        start = end
    return level_targets