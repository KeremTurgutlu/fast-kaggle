#AUTOGENERATED! DO NOT EDIT! File to edit: dev/02_scheduler.ipynb (unless otherwise specified).

__all__ = ['fit_warm_cosanneal']

#Cell
from fastai.vision import *
from fastai.callbacks.general_sched import *

#Cell
def fit_warm_cosanneal(learn:Learner, num_epoch:int, lr:float=defaults.lr, annealing_start:float=0.2)->None:
    "cos annealing with constant warmup"
    n = len(learn.data.train_dl)
    anneal_start = int(n*num_epoch*annealing_start)
    phase0 = TrainingPhase(anneal_start).schedule_hp('lr', lr/100) # warmup
    phase1 = TrainingPhase(n*num_epoch - anneal_start).schedule_hp('lr', lr, anneal=annealing_cos)
    phases = [phase0, phase1]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks = [cb for cb in learn.callbacks if cb.__class__ != GeneralScheduler]
    learn.callbacks.append(sched)
    learn.fit(num_epoch)