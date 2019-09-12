from fastai.vision import *
from dev_utils import *


# input and output paths
PATH = Path(input_dataset[0]['path'])
EXPORT_PATH = Path(output_dataset[0]['path'])

# parse ml parameters
ml_params = parse_ml_params(ml_task_params)

# pass on to script if not None
CODES = ml_params.get("CODES")
TRAIN = ml_params.get("TRAIN")
VALID = ml_params.get("VALID")
TEST = ml_params.get("TEST")
sample_size = ml_params.get("sample_size")
bs = ml_params.get("bs", 16)
size = ml_params.get("size", 224)
imagenet_pretrained = ml_params.get("imagenet_pretrained", 1)
max_lr = ml_params.get("max_lr", 3e-3)
model_name = ml_params.get("model_name", "mybestmodel")
epochs = ml_params.get("epochs", 10)
tracking_metric = ml_params.get("tracking_metric", "dice")
void_name = ml_params.get("void_name", "Void")
loss_function = ml_params.get("loss_function", "crossentropy")
opt = ml_params.get("opt", None)

multi_gpu = ml_params.get("multi_gpu", False)
print(); print(ml_params); print()


# training command args
if "multi_gpu" in ml_params: ml_params.pop("multi_gpu")
if "IterativeDevelopment" in ml_params: ml_params.pop("IterativeDevelopment")
ml_params["PATH"], ml_params["EXPORT_PATH"] = PATH, EXPORT_PATH
ngpus = torch.cuda.device_count()
# root = "/home/code-base/runtime" # use for local dev
root = "/"
if ngpus == 1: base_training_cmd = ["python", f"{root}/app/python/dev/dev/training.py"]
elif ngpus == 4: base_training_cmd = ["python", "/opt/conda/lib/python3.7/site-packages/fastai/launch.py",
                            "--gpus=0123", f"{root}/app/python/dev/dev/training.py"]
elif ngpus == 8: base_training_cmd = ["python", "/opt/conda/lib/python3.7/site-packages/fastai/launch.py",
                            "--gpus=01234567", f"{root}/app/python/dev/dev/training.py"]
training_cmd = base_training_cmd + [f"--{k}={v}" for k,v in ml_params.items() if v]

print(); print(training_cmd); print()
res = run_command(training_cmd)
print(res)
