from fastai.vision import * 

__all__ = ["foreground_acc", "iou", "multilabel_dice", "multilabel_iou", "dice"]

def foreground_acc(input, target, void_code)->Rank0Tensor:
    "Accuracy excluding background pixels"
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def iou(input: torch.Tensor, targs: torch.Tensor)->Rank0Tensor:
    "Binary IOU"
    return dice(input, targs, iou=True)

def multilabel_dice(input:Tensor, targs:Tensor, c, iou:bool=False, mean=True)->Rank0Tensor:
    "Dataset Mean Dice"
    input = input.argmax(dim=1, keepdim=True).view(-1)
    targs = targs.view(-1)
    res = []
    for ci in range(c):
        _input, _targs = input == ci, targs == ci
        intersect = (_input * _targs).sum().float()
        union = (_input+_targs).sum().float()
        if not iou: res.append((2. * intersect / union if union > 0 else union.new([1.]).squeeze()))
        else: res.append(intersect / (union-intersect+1.0))
    res = torch.tensor(res).to(input.device)
    if not mean: return res
    else: return res.mean()

def multilabel_iou(input: torch.Tensor, targs: torch.Tensor, c)->Rank0Tensor:
    "Dataset Mean IOU"
    return multilabel_dice(input, targs, c=c, iou=True)
