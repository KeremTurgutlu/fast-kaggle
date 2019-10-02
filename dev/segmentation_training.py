# Always keeps this in cell index position: 2
from fastai.vision import *
from fastai.distributed import *
from fastai.script import *
from fastai.utils.mem import *

from local.segmentation.dataset import *
from local.segmentation.models import *
from local.segmentation import metrics
from local.segmentation import losses_binary, losses_multilabel
from local.callbacks import *
from local.optimizers import *

# https://stackoverflow.com/questions/8299270/ultimate-answer-to-relative-python-imports
@call_parse
def main(    
    # data
    PATH:Param("Path which have data", str)="",
    IMAGES:Param("images folder path name", str)="images",
    MASKS:Param("mask folder path name", str)="masks",
    CODES:Param("codes.txt with pixel codes", str)="",
    TRAIN:Param("train.txt with training image names", str)="",
    VALID:Param("valid.txt with validation image names", str)=None,
    TEST:Param("test.txt with test image names", str)=None,
    suffix:Param("suffix for label filenames", str)=".png",
    sample_size:Param("", int)=None,
    bs:Param("Batch size", int)=80,
    size:Param("Image size", int)=224,
    imagenet_pretrained:Param("Whether to normalize with inet stats", int)=1,
    
    # model
    modelname:Param("Model name from segmentation.models", str)="resdunet18",
    modelconfig:Param("JSON dictionary of model config", str)="{}",
    
    # metric
    tracking_metric:Param("Which metric to use for tracking and evaluation", str)="dice",
    void_name:Param("Background class name", str)=None,
    
    # train
    loss_function:Param("Loss function for training", str)="cross_entropy",
    opt:Param("Optimizer for training", str)=None,
    max_lr:Param("Learning Rate", float)=3e-3,
    epochs:Param("Number of max epochs to train", int)=10,
    
    # modelexports
    EXPORT_PATH:Param("Where to export trained model", str)=".",
    
    gpu:Param("GPU to run on, can handle multi gpu", str)=None):
    
    """
    For Multi GPU Run: python ../fastai/fastai/launch.py {--gpus=0123} ./training.py {--your args}
    For Single GPU Run: python ./training.py {--your args}
    """
        
    # Setup init
    gpu = setup_distrib(gpu)
    
    # Args
    if not gpu: print(f"Print args here: ")
        
    # data
    PATH = Path(PATH)
    try: VALID = float(VALID)
    except: passzx
    ssdata = SemanticSegmentationData(PATH, IMAGES, MASKS, CODES, TRAIN,
                                      VALID, TEST, sample_size, bs, size, suffix)
    data = ssdata.get_data()
    if imagenet_pretrained: data.normalize(imagenet_stats)
    else: data.normalize()   
    
    # model
    model, split_fn, copy_conf = get_model(modelname, data, eval(modelconfig))
    if not gpu: print(f"Training with model: {modelname} and config: {copy_conf}")
        
    # learn
    learn = Learner(data, model)
    learn = learn.split(split_fn)
    pretrained = copy_conf["pretrained"]
    if pretrained: learn.freeze()
    learn.path, learn.model_dir = Path(EXPORT_PATH), 'models'

    # metric
    metric = getattr(metrics, tracking_metric)
    if not gpu: print(f"Training with metric: {metric}")
    if tracking_metric in ["multilabel_dice", "multilabel_iou"]: metric = partial(metric, c=learn.data.c)
    if tracking_metric == "foreground_acc": 
        void_code = np.where(learn.data.classes == void_name)[0].item()
        metric = partial(metric, void_code=void_code)
    learn.metrics = [metric]
    
    # loss
    try: loss = getattr(losses_binary, loss_function)
    except: loss = getattr(losses_multilabel, loss_function)  
    learn.loss_func = loss 
    if not gpu: print(f"Training with loss: {learn.loss_func}")

    # callbacks
    save_cb = SaveDistributedModelCallback(learn, tracking_metric, "max", name=f"best_of_{modelname}", gpu=gpu)
    csvlog_cb = CSVDistributedLogger(learn, 'training_log', append=True, gpu=gpu)
    nan_cb = TerminateOnNaNCallback()
    cbs = [save_cb, csvlog_cb, nan_cb]
        
    # optimizer / scheduler
    alpha, mom, eps = 0.99, 0.9, 1e-8
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
        learn.fit_one_cycle(epochs, max_lr, callbacks=cbs)

        # stage-2
        lrs = slice(max_lr/100,max_lr/4)
        learn.freeze_to(-2)
        learn.fit_one_cycle(epochs, lrs, pct_start=0.8, callbacks=cbs)
 
        # stage-3
        lrs = slice(max_lr/100,max_lr/4)
        learn.unfreeze()
        learn.fit_one_cycle(epochs, lrs, pct_start=0.8, callbacks=cbs)
    else:
        if not gpu: print("Training from scratch")
        learn.fit_one_cycle(epochs, max_lr, callbacks=cbs)
        
    # modelexports
    if not nan_cb.isnan:
        if TEST: dtypes = ["Valid", "Test"]
        else: dtypes = ["Valid"]
        for dtype in dtypes:
            if not gpu: print(f"Generating Raw Predictions for {dtype}...")
            ds_type = getattr(DatasetType, dtype)
            preds, targs = learn.get_preds(ds_type)
            ds = learn.data.test_ds if dtype == "Test" else learn.data.valid_ds
            fnames = list(ds.items)
            try_save({"fnames":fnames, "preds":to_cpu(preds), "targs":to_cpu(targs)},
                     path=Path(EXPORT_PATH), file=f"{dtype}_raw_preds.pkl")
            if not gpu: print(f"Done.")
    else:
        if not gpu: print(f"Skipping Predictions due to NaN.")
