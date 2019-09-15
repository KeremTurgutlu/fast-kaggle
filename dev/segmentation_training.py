# Always keeps this in cell index position: 1
from fastai.vision import *
from fastai.distributed import *
from fastai.script import *
from fastai.utils.mem import *

from local.segmentation.dataset import *
from local.segmentation import metrics
from local.segmentation import losses
from local.distributed import *
from local.optimizers import *

# https://stackoverflow.com/questions/8299270/ultimate-answer-to-relative-python-imports
@call_parse
def main(    
    PATH:Param("Path which have data", str)="",
    IMAGES:Param("images folder path name", str)="images",
    MASKS:Param("mask folder path name", str)="masks",
    CODES:Param("codes.txt with pixel codes", str)="",
    TRAIN:Param("train.txt with training image names", str)="",
    VALID:Param("valid.txt with validation image names", str)=None,
    TEST:Param("test.txt with test image names", str)=None,
    sample_size:Param("", int)=None,
    bs:Param("Batch size", int)=80,
    size:Param("Image size", int)=224,
    imagenet_pretrained:Param("Use imagenet weights for DynamicUnet", int)=1,
    max_lr:Param("Learning Rate", float)=3e-3,
    model_name:Param("Model name for save", str)="mybestmodel",
    epochs:Param("""Number of max epochs to train""", int)=10,
    tracking_metric:Param("""Which metric to use for tracking and evaluation""", str)="dice",
    void_name:Param("""Background class name""", str)=None,
    loss_function:Param("""Loss function for training""", str)="crossentropy",
    opt:Param("""Optimizer for training""", str)=None,
    arch_name:Param("""Architecture backbone for training""", str)="resnet34",
    
    EXPORT_PATH:Param("""Where to export trained model""", str)=".",
    
    gpu:Param("GPU to run on, can handle multi gpu", str)=None):
    
    """
    For Multi GPU Run: python ../fastai/fastai/launch.py {--gpus=0123} ./training.py {--your args}
    For Single GPU Run: python ./training.py {--your args}
    bs: 80 size: 224 , bs: 320 size: 112 
    """
        
    # Setup init
    gpu = setup_distrib(gpu)
    
    # Args
    if not gpu: print(f"Print args here: ")
        
    # Get data
    PATH = Path(PATH)
    ssdata = SemanticSegmentationData(PATH, IMAGES, MASKS, CODES, TRAIN, VALID, TEST, sample_size, bs, size)
    data = ssdata.get_data()
    if imagenet_pretrained: data.normalize(imagenet_stats)
    else: data.normalize()   
    
    # learn - models: 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50',
    arch = getattr(models, arch_name)
    if not gpu: print(f"Training with arch: {arch}")
    learn = unet_learner(data, arch = arch, pretrained = True)
    learn.path, learn.model_dir = Path(EXPORT_PATH), 'models'

    # metric
    metric = getattr(metrics, tracking_metric)
    if not gpu: print(f"Tracking metric: {metric}")
    if tracking_metric in ["multilabel_dice", "multilabel_iou"]: metric = partial(metric, c=learn.data.c)
    if tracking_metric == "foreground_acc": 
        void_code = np.where(learn.data.classes == void_name)[0].item()
        metric = partial(metric, void_code=void_code)
    learn.metrics = [metric]
    
    # loss
    loss = getattr(losses, loss_function, None)
    if loss: learn.loss_func = loss 
    if not gpu: print(f"Training with loss: {learn.loss_func}")

    # callbacks
    learn.callback_fns.append(partial(SaveDistributedModelCallback, monitor=tracking_metric, 
                                      mode="max", name=model_name, gpu=gpu))
        
    # optimizer / scheduler
    alpha=0.99; mom=0.9; eps=1e-8
    
    if   opt=='adam' : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='radam' : opt_func = partial(RAdam, betas=(mom,alpha), eps=eps)
    elif opt=='novograd' : opt_func = partial(Novograd, betas=(mom,alpha), eps=eps)
    elif opt=='rms'  : opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd'  : opt_func = partial(optim.SGD, momentum=mom)
    elif opt=='ranger'  : opt_func = partial(Ranger,  betas=(mom,alpha), eps=eps)
    elif opt=='ralamb'  : opt_func = partial(Ralamb,  betas=(mom,alpha), eps=eps)
    elif opt=='rangerlars'  : opt_func = partial(RangerLars,  betas=(mom,alpha), eps=eps)
    elif opt=='lookahead'  : opt_func = partial(LookaheadAdam, betas=(mom,alpha), eps=eps)
    elif opt=='lamb'  : opt_func = partial(Lamb, betas=(mom,alpha), eps=eps)
    if opt: learn.opt_func = opt_func

    # distributed
    if (gpu is not None) & (num_distrib()>1): learn.to_distributed(gpu)
    
    # to_fp16 
    learn.to_fp16()
    
    # train
    if not gpu: print(f"Starting training with max_lr: {max_lr}")
    if imagenet_pretrained:
        if not gpu: print("Training with transfer learning")
        # stage-1
        learn.freeze_to(-1)
        learn.fit_one_cycle(epochs, max_lr)
        
        # load model hack
        best_init = learn.save_distributed_model_callback.best
        learn.callback_fns = [cb_fn for cb_fn in learn.callback_fns if cb_fn.func == Recorder]
        learn.callback_fns.append(partial(SaveDistributedModelCallback, monitor=tracking_metric, name=model_name, best_init=best_init))

        # stage-2
        lrs = slice(max_lr/100,max_lr/4)
        learn.freeze_to(-2)
        learn.fit_one_cycle(epochs, lrs, pct_start=0.8)
        
        # load model hack
        best_init = learn.save_distributed_model_callback.best
        learn.callback_fns = [cb_fn for cb_fn in learn.callback_fns if cb_fn.func == Recorder]
        learn.callback_fns.append(partial(SaveDistributedModelCallback, monitor=tracking_metric, name=model_name, best_init=best_init))

        # stage-3
        lrs = slice(max_lr/100,max_lr/4)
        learn.unfreeze()
        learn.fit_one_cycle(epochs, lrs, pct_start=0.8)
    else:
        if not gpu: print("Training from scratch")
        learn.fit_one_cycle(epochs, max_lr)
    
        
    # save test preds 
    if TEST:
        preds, targs = learn.get_preds(DatasetType.Test)
        fnames = list(data.test_ds.items)
        try_save({"fnames":fnames, 
                  "preds":to_cpu(preds),
                  "targs":to_cpu(targs)}, path=Path(EXPORT_PATH), file="raw_preds.pkl")
    
    # to_fp32 + export learn
    learn.to_fp32()    
    learn.load(model_name) # load best saved model
    if not gpu: print(f"Exporting model to: {EXPORT_PATH}")
    learn.export(f"{model_name}_export.pkl")