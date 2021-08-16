#!/bin/bash

python train.py \
--continuous \
--deterministic_src \
--deterministic_tgt \
--checkpoint_path ../resource/checkpoint/continuous.pth \
--tags continuous deterministic_src deterministic_tgt