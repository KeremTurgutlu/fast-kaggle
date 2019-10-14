#AUTOGENERATED! DO NOT EDIT! File to edit: dev/15_segmentation_tta.ipynb (unless otherwise specified).

__all__ = []

#Cell
from fastai.vision import *

#Cell
from fastai.basic_train import _loss_func2activ
def _seg_tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid) -> Iterator[List[Tensor]]:
    "Computes the outputs for non-flip and flip_lr augmented inputs"
    dl = learn.dl(ds_type)
    ds = dl.dataset
    old = ds.tfms
    try:
        pbar = master_bar(range(2))
        for i in pbar:
            tfm = [] # to remove random crop resize aug
            if i: tfm.append(flip_lr(p=1.))
            ds.tfms = tfm
            yield get_preds(learn.model, dl, pbar=pbar, activ=_loss_func2activ(learn.loss_func))[0]
    finally: ds.tfms = old

# flip_lr TTA preds
def _seg_TTA(seg_learn, ds_type=DatasetType.Valid):
    "Takes average of original and flip_lr"
    orig_preds, flip_lr_preds = list(_seg_tta_only(seg_learn, ds_type))
    flip_lr_preds = torch.stack([torch.flip(o, dims=[-1]) for o in flip_lr_preds], dim=0)
    avg_preds = (orig_preds + flip_lr_preds)/2
    return avg_preds