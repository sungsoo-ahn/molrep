#!/bin/bash

python train.py \
--vq_codebook_size 100 \
--checkpoint_path ../resource/checkpoint/codebook100.pth \
--randomize_sync \
--tags codebook100 randomize