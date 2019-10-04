#AUTOGENERATED! DO NOT EDIT! File to edit: dev/12_segmentation_metrics.ipynb (unless otherwise specified).

__all__ = ['foreground_acc', 'dice', 'iou', 'multilabel_dice', 'multilabel_iou', 'sigmoid_multilabel_dice',
           'sigmoid_multilabel_iou']

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
        sigmoid_input = input.sigmoid()
        thresholded_input = sigmoid_input > threshold
        _, indices = torch.max(sigmoid_input, dim=1)
        values, _ = torch.max(thresholded_input, dim=1)
        input = (values.float()*indices.float()).view(-1)
    else:
        input = input.argmax(dim=1, keepdim=True).view(-1)

    targs = targs.view(-1)
    res = []
    for ci in range(c):
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