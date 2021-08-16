#!/bin/bash

python train.py \
--vq_codebook_size 10 \
--deterministic_src \
--deterministic_tgt \
--checkpoint_path ../resource/checkpoint/codebook10_sync.pth \
--tags codebook10 randomize_sync