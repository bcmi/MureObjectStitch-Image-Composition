#!/bin/bash

python scripts/inference.py \
--outdir results \
--testdir examples/example1 \
--num_samples 5 \
--sample_steps 50 \
--gpu 0
