#!/bin/bash

python train.py \
--vq_codebook_size 10 \
--checkpoint_path ../resource/checkpoint/codebook10.pth \
--randomize_sync \
--tags codebook10