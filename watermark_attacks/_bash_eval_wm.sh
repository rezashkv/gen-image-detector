#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python evaluate_watermark.py \
--wm-method stegaStamp \
--attack diffpure \
--dataset imagenet \
--data-dir images/imagenet/stegaStamp \
--org-data-dir images/imagenet/org \
--out-fname diffpure \
--save-images \


CUDA_VISIBLE_DEVICES=0 python evaluate_watermark.py \
--wm-method dwtDct \
--attack diffpure \
--dataset imagenet \
--data-dir images/imagenet/dwtDct \
--org-data-dir images/imagenet/org \
--out-fname diffpure \
--save-images \

CUDA_VISIBLE_DEVICES=0 python evaluate_watermark.py \
--wm-method dwtDctSvd \
--attack diffpure \
--dataset imagenet \
--data-dir images/imagenet/dwtDctSvd \
--org-data-dir images/imagenet/org \
--out-fname diffpure \
--save-images \

CUDA_VISIBLE_DEVICES=0 python evaluate_watermark.py \
--wm-method treeRing \
--attack diffpure \
--dataset imagenet \
--data-dir images/imagenet/treeRing \
--org-data-dir images/imagenet/org \
--out-fname diffpure \
--save-images \

CUDA_VISIBLE_DEVICES=0 python evaluate_watermark.py \
--wm-method watermarkDM \
--attack diffpure \
--dataset imagenet \
--data-dir images/imagenet/watermarkDM \
--org-data-dir images/imagenet/org \
--out-fname diffpure \
--save-images \

CUDA_VISIBLE_DEVICES=0 python evaluate_watermark.py \
--wm-method rivaGan \
--attack diffpure \
--dataset imagenet \
--data-dir images/imagenet/rivaGan \
--org-data-dir images/imagenet/org \
--out-fname diffpure \
--save-images \
