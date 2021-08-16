#!/bin/bash

python train.py \
--vq_codebook_size 50 \
--deterministic_src \
--deterministic_tgt \
--checkpoint_path ../resource/checkpoint/codebook50_sync.pth \
--tags codebook50 randomize_sync