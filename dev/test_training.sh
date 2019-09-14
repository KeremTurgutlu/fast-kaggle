#!/bin/sh

#-------------------------------
# SEMANTIC SEGMENTATION
#-------------------------------

# single gpu - multilabel: CAMVID Dataset
python ./semantic_segmentation_training.py \
    --PATH=/home/turgutluk/.fastai/data/camvid_tiny \
    --CODES=codes.txt \
    --TRAIN=train.txt \
    --VALID=valid.txt \
    --TEST=test.txt \
    --bs=32 \
    --size=224 \
    --imagenet_pretrained=1 \
    --max_lr=3e-3 \
    --model_name=mybestmodel \
    --epochs=20 \
    --tracking_metric=foreground_acc \
    --void_name=Void \
    --loss_function=xentropy \
    --opt=radam

# # multi gpu
# python /opt/conda/lib/python3.7/site-packages/fastai/launch.py --gpus=0123 /home/code-base/runtime/app/python/dev/dev/training.py \
#     --PATH=$SENSEI_USERSPACE_SELF/camvid \
#     --CODES=codes.txt \
#     --TRAIN=train.txt \
#     --VALID=valid.txt \
#     --TEST=test.txt \
#     --bs=32 \
#     --size=224 \
#     --imagenet_pretrained=1 \
#     --max_lr=3e-3 \
#     --model_name=mybestmodel \
#     --epochs=5 \
#     --tracking_metric=foreground_acc \
#     --void_name=Void \
#     --loss_function=lovasz_softmax \
#     --opt=radam \
#     --arch_name=resnet18
