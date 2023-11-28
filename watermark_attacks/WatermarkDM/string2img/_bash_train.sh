#!/bin/bash



CUDA_VISIBLE_DEVICES=0 python train_imagenet.py \
--data_dir 'path_to_image_dataset' \
--image_resolution 256 \
--output_dir ./_output/imagenet_64_adv_03 \
--bit_length 64 \
--batch_size 64 \
--num_epochs 100 \

