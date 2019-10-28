# Always keeps this in cell index position: 2
from fastai.vision import *
from fastai.distributed import *
from fastai.script import *
from fastai.utils.mem import *

from local.segmentation.dataset import *
from local.segmentation.models import *
from local.segmentation import metrics
from local.segmentation import losses_binary, losses_multilabel
from local.segmentation import tta
from local.callbacks import *
from local.optimizers.optimizers import *


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
    size:Param("Image size", str)="(224,224)",
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
    
    # TTA
    tta_updown:Param("Whether to do TTA with updown", int)=0,
    
    gpu:Param("GPU to run on, can handle multi gpu", str)=None):
    
    """
    For Multi GPU Run: python ../fastai/fastai/launch.py {--gpus=0123} ./training.py {--your args}
    For Single GPU Run: python ./training.py {--your args}
    """
        
    # Setup init
    gpu = setup_distrib(gpu)
    gpu_rank = rank_distrib()
    
    # Args
    if not gpu_rank: print(f"Print args here: ")
        
    # data
    PATH = Path(PATH)
    try: VALID = float(VALID)
    except: pass
    size = eval(size)
    ssdata = SemanticSegmentationData(PATH, IMAGES, MASKS, CODES, TRAIN,
                                      VALID, TEST, sample_size, bs, size, suffix)
    data = ssdata.get_data()
    if imagenet_pretrained: data.normalize(imagenet_stats)
    else: data.normalize()   
    
    # model
    model, split_fn, copy_conf = get_model(modelname, data, eval(modelconfig))
    if not gpu_rank: print(f"Training with model: {modelname} and config: {copy_conf}")
        
    # learn
    learn = Learner(data, model)
    learn = learn.split(split_fn)
    pretrained = copy_conf["pretrained"]
    if pretrained: learn.freeze()
    learn.path, learn.model_dir = Path(EXPORT_PATH), 'models'

    # loss
    try: loss = getattr(losses_binary, loss_function)
    except: loss = getattr(losses_multilabel, loss_function)  
    learn.loss_func = loss 
    if not gpu_rank: print(f"Training with loss: {learn.loss_func}")
        
    # metric
    metric = getattr(metrics, tracking_metric)
    if not gpu_rank: print(f"Training with metric: {metric}")
    if tracking_metric in ["multilabel_dice", "multilabel_iou"]: metric = partial(metric, c=learn.data.c)
    if tracking_metric == "foreground_acc": 
        void_code = np.where(learn.data.classes == void_name)[0].item()
        metric = partial(metric, void_code=void_code)
    learn.metrics = [metric]

    # callbacks
    save_cb = SaveDistributedModelCallback(learn, tracking_metric, "max", name=f"best_of_{modelname}", gpu=gpu_rank)
    csvlog_cb = CSVDistributedLogger(learn, 'training_log', append=True, gpu=gpu_rank)
    nan_cb = TerminateOnNaNCallback()
    cbs = [save_cb, csvlog_cb, nan_cb]
        
    # optimizer / scheduler
    if opt: 
        opt_func = get_opt_func(opt) # TODO: alpha, mom, eps
        learn.opt_func = opt_func
    if not gpu_rank: print(f"Starting training with opt_func: {learn.opt_func}")

    # distributed
    if (gpu is not None) & (num_distrib()>1): learn.to_distributed(gpu)
    
    # to_fp16 
    learn.to_fp16()
    
    # train
    if not gpu_rank: print(f"Starting training with max_lr: {max_lr}")
    if imagenet_pretrained:
        if not gpu_rank: print("Training with transfer learning")
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
        if not gpu_rank: print("Training from scratch")
        learn.fit_one_cycle(epochs, max_lr, callbacks=cbs)
        
    # preds - https://github.com/NVIDIA/apex/issues/515
    # model export
    if (not nan_cb.isnan) and (not gpu_rank):
        learn.load(f"best_of_{modelname}") # load best model
        dtypes = ["Valid", "Test"] if TEST else ["Valid"]
        for dtype in dtypes:
            ds_type = getattr(DatasetType, dtype)
            ds = learn.data.test_ds if dtype == "Test" else learn.data.valid_ds
            fnames = list(ds.items)
            # raw preds
            if not gpu_rank: print(f"Generating Raw Predictions for {dtype}...")
            preds, targs = learn.get_preds(ds_type)
            try_save({"fnames":fnames,
                      "preds":to_cpu(preds),
                      "targs":to_cpu(targs)},
                      path=Path(EXPORT_PATH), file=f"{dtype}_raw_preds.pkl")
            # TTA preds
            if not gpu_rank: print(f"Generating TTA Predictions for {dtype}...")
            preds = learn.segTTA(ds_type, updown=tta_updown)
            try_save({"fnames":fnames,
                      "preds":to_cpu(preds),
                      "targs":to_cpu(targs)},
                      path=Path(EXPORT_PATH), file=f"{dtype}_tta_preds.pkl")
            
            if not gpu_rank: print(f"Done.")
        if not gpu_rank: print(f"Exporting learn...")
        learn.export(f"best_of_{modelname}_export.pkl")
    else:
        if not gpu_rank: print(f"Skipping Predictions due to NaN.")