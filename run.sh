#!/usr/bin/env bash

python main.py  --model-name "curie" \
                --log-file "./results/openai_models/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en" \
                --few-shot


python main.py  --model-name "babbage" \
                --log-file "./results/openai_models/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en" \
                --few-shot


python main.py  --model-name "ada" \
                --log-file "./results/openai_models/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en" \
                --few-shot





