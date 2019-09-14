__all__ = ["dice_loss", "lovasz_softmax", "lovasz_hinge"]

from fastai.vision import *
from .lovasz_loss import *


def dice_loss(input, target):
    "binary dice loss"
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))