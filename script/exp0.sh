#!/bin/bash

python train.py \
--randomize_src \
--randomize_tgt \
--checkpoint_path ../resource/checkpoint/heteroencoder.pth \
--tags heteroencoder