#AUTOGENERATED! DO NOT EDIT! File to edit: dev/11_segmentation_losses_mulitlabel.ipynb (unless otherwise specified).

__all__ = ['lovasz_softmax', 'cross_entropy', 'LabelBinarizer', 'bce_with_logits_flat_loss', 'fscore_dice_loss',
           'bce_dice_loss']

#Cell
from fastai.vision import *
from .lovasz_loss import *

#Cell
cross_entropy = CrossEntropyFlat(axis=1)

#Cell
class LabelBinarizer(Module):
    "converts a 2d tensor target label to 3d binary representation"
    def __init__(self, exclude_zero=True):
        self.exclude_zero = exclude_zero

    def forward(self, input, target):
        c, dtype = input.size(1), input.dtype
        trange = torch.arange(1 if self.exclude_zero else 0, c+1)
        trange = trange[None,...,None,None].to(target.device)
        return (target == trange).float()

#Cell
class _BCEWithLogitsFlatLoss(Module):
    def __init__(self, **kwargs):
        self.label_binarizer = LabelBinarizer(**kwargs)
        self.loss_fn = BCEWithLogitsFlat()
    def forward(self, input, target):
        return self.loss_fn(input, self.label_binarizer(input, target))

#Cell
bce_with_logits_flat_loss = _BCEWithLogitsFlatLoss(exclude_zero=True)

#Cell
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/losses.py

def _f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


class _DiceLoss(nn.Module):
    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps
        self.label_binarizer = LabelBinarizer()

    def forward(self, input, target):
        target = self.label_binarizer(input, target)
        return 1 - _f_score(input, target, beta=1., eps=self.eps,
                            threshold=None, activation=self.activation)


class _BCEDiceLoss(_DiceLoss):
    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.label_binarizer = LabelBinarizer()

    def forward(self, input, target):
        dice = super().forward(input, target)
        target = self.label_binarizer(input, target)
        bce = self.bce(input, target)

        return dice + bce

#Cell
fscore_dice_loss  = _DiceLoss()

#Cell
bce_dice_loss = _BCEDiceLoss()