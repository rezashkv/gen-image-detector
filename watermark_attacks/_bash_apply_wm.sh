#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method stegaStamp \
--dataset imagenet \
--data-dir ./images/imagenet/org \
--out-dir ./images/imagenet/stegaStamp \

CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method dwtDct \
--dataset imagenet \
--data-dir ./images/imagenet/org \
--out-dir ./images/imagenet/dwtDct \

CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method dwtDctSvd \
--dataset imagenet \
--data-dir ./images/imagenet/org \
--out-dir ./images/imagenet/dwtDctSvd \

CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method watermarkDM \
--dataset imagenet \
--data-dir ./images/imagenet/org \
--out-dir ./images/imagenet/watermarkDM \

CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method rivaGan \
--dataset imagenet \
--data-dir ./images/imagenet/org \
--out-dir ./images/imagenet/rivaGan \

CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method treeRing \
--dataset imagenet \
--data-dir ./images/imagenet/org \
--out-dir ./images/imagenet/treeRing \

