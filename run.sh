#!/usr/bin/env bash


RESULTS_ROOT="./results"

python main.py  --model-name "t5" \
                --log-file "${RESULTS_ROOT}/log.txt"  \
                --dataset 'datasets/europarl/de-en/en_de.df'

