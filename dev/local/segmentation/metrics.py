#AUTOGENERATED! DO NOT EDIT! File to edit: dev/12_segmentation_metrics.ipynb (unless otherwise specified).

__all__ = ['foreground_acc', 'dice', 'iou', 'multilabel_dice', 'multilabel_iou', 'sigmoid_multilabel_dice',
           'sigmoid_multilabel_iou', 'sigmoid_dice_novoid']

#Cell
from fastai.vision import *
from fastai.metrics import foreground_acc, dice

#Cell
def iou(input: torch.Tensor, targs: torch.Tensor, **kwargs)->Rank0Tensor:
    "Binary IOU"
    return dice(input, targs, iou=True, **kwargs)

#Cell
def multilabel_dice(input:Tensor, targs:Tensor, c:int, iou:bool=False,
                    mean=True, eps:float=1e-8, sigmoid:bool=False, threshold:float=0.5)->Rank0Tensor:
    "Batch/Dataset Mean Dice"
    if sigmoid:
        sigmoid_input     = input.sigmoid()
        thresholded_input = sigmoid_input > threshold
        _, indices        = torch.max(sigmoid_input, dim=1);
        indices          += 1
        values, _         = torch.max(thresholded_input, dim=1)
        input             = (values.float()*indices.float()).view(-1)
    else:
        input             = input.argmax(dim=1, keepdim=True).view(-1)

    targs = targs.view(-1)
    res = []
    for ci in range(c):
        # float() fail for fp16 - nan
        _input, _targs = input == ci, targs == ci
        intersect = (_input * _targs).sum().float()
        union = (_input+_targs).sum().float()
        if not iou: res.append((2. * intersect / union if union > 0 else union.new([1.]).squeeze()))
        else: res.append(intersect / (union-intersect+eps))
    res = torch.tensor(res).to(input.device)
    if not mean: return res
    else: return res.mean()

#Cell
def multilabel_iou(input: torch.Tensor, targs: torch.Tensor, c)->Rank0Tensor:
    "Batch/Dataset Mean IOU"
    return multilabel_dice(input, targs, c=c, iou=True)

#Cell
def sigmoid_multilabel_dice(input, target, c, threshold):
    return multilabel_dice(input, target, c=c, threshold=threshold, sigmoid=True)

#Cell
def sigmoid_multilabel_iou(input, target, c, threshold):
    return multilabel_iou(input, target, c=c, threshold=threshold, sigmoid=True, iou=True)

#Cell
def _dice(input:Tensor, targs:Tensor, iou:bool=False, eps:float=1e-8,
          reduce:bool=True)->Rank0Tensor:
    "Dice coefficient metric for probas and binary target."
#     warn("Warning union=0->0")
    n = targs.shape[0]
    input = input.view(n,-1).float()
    targs = targs.view(n,-1).float()
    intersect = (input * targs).sum(dim=1).float()
    union = (input+targs).sum(dim=1).float()
    if not iou: l = 2. * intersect / union
    else: l = intersect / (union-intersect+eps)
#     l[union == 0.] = 1.
    l[union == 0.] = 0.
    if reduce: return l.mean()
    else: return l

#Cell
def _to_sigmoid_input(logits, threshold=0.5):
    "convert logits to preds with sigmoid and thresh (void=0)"
    sigmoid_input = logits.sigmoid()
    thresholded_input = sigmoid_input > threshold

    _, indices = torch.max(sigmoid_input, dim=1)
    indices += 1
    values, _ = torch.max(thresholded_input, dim=1)
    preds = (values.float()*indices.float())
    return preds

#Cell
def sigmoid_dice_novoid(input:Tensor, target:Tensor, threshold:float=0.5,
                        macro:bool=True)->Rank0Tensor:
    "macro: mean of per class dice, micro: mean of per image mean dice"
    c = input.size(1)
    preds = _to_sigmoid_input(input, threshold)
    if macro:
        res = [_dice(preds==ci, target==ci) for ci in range(1, c+1)]
        return torch.mean(tensor(res))
    else:
        res = [_dice(preds==ci, target==ci, reduce=False) for ci in range(1, c+1)]
        return torch.stack(res).mean(0).mean()