#!/usr/bin/env bash

python main.py  --model-name "t5-small" \
                --log-file "./results/t5/thruthfullqa_en_ro.txt"  \
                --dataset './datasets/truthfullqa/en_ro.df' \
                --lang-pair "en-ro" 

python main.py  --model-name "t5-base" \
                --log-file "./results/t5/thruthfullqa_en_fr.txt"  \
                --dataset './datasets/truthfullqa/en_fr.df' \
                --lang-pair "en-ro" 





