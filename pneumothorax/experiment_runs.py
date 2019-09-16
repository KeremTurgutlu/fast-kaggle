import fastai; fastai.__version__
from fastai.vision import *
import sys; sys.path.append("../dev")
from local.script import run_command

PATH = Path("/home/turgutluk/data/siim_acr_pneu/")
IMAGES, MASKS, CODES, TRAIN, VALID, TEST = \
    "train/images_1024", "train/masks_1024", "codes.txt", "train.txt", 0.1, "test.txt"
bs = 1
size = 1024
epochs = 1
VALID = 0.1

for i in range(1,2):
    print(f"EXPERIMENT {i}")
    run_command(f"""
    python {Path(fastai.__file__).parent}/launch.py 
    --gpus=0123457 /home/turgutluk/git/fast-kaggle/dev/segmentation_training.py \
    --PATH={PATH} \
    --IMAGES={IMAGES} \
    --MASKS={MASKS} \
    --CODES={CODES} \
    --TRAIN={TRAIN} \
    --VALID={VALID} \
    --TEST={TEST} \
    --bs={bs} \
    --size={size} \
    --imagenet_pretrained=1 \
    --max_lr=3e-3 \
    --model_name=bestmodel \
    --epochs={epochs} \
    --tracking_metric=dice \
    --void_name=Void \
    --loss_function=xentropy \
    --opt=radam
    --EXPORT_PATH=./experiment_exports/experiment{i}
    """)