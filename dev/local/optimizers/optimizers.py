#AUTOGENERATED! DO NOT EDIT! File to edit: dev/04_optimizers_optimizers.ipynb (unless otherwise specified).

__all__ = ['get_opt_func']

#Cell
from fastai.vision import *
from . import *

#Cell
def get_opt_func(opt, mom=0.99, alpha=0.9, eps=1e-8):
    "return optimizer function"
    if   opt=='adam':        opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='radam':       opt_func = partial(RAdam, betas=(mom,alpha), eps=eps)
    elif opt=='novograd':    opt_func = partial(Novograd, betas=(mom,alpha), eps=eps)
    elif opt=='rms':         opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd':         opt_func = partial(optim.SGD, momentum=mom)
    elif opt=='ranger':      opt_func = partial(Ranger,  betas=(mom,alpha), eps=eps)
    elif opt=='ralamb':      opt_func = partial(Ralamb,  betas=(mom,alpha), eps=eps)
    elif opt=='rangerlars':  opt_func = partial(RangerLars,  betas=(mom,alpha), eps=eps)
    elif opt=='lookahead':   opt_func = partial(LookaheadAdam, betas=(mom,alpha), eps=eps)
    elif opt=='lamb':        opt_func = partial(Lamb, betas=(mom,alpha), eps=eps)
    return opt_func