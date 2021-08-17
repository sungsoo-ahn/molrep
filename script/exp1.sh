#!/bin/bash

python train.py \
--vq_codebook_size 50 \
--checkpoint_path ../resource/checkpoint/codebook50.pth \
--randomize_sync \
--tags codebook50 randomize