#!/usr/bin/env bash


python main.py  --model-name "text-babbage-001" \
                --log-file "./results/openai/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "en-de" 

python main.py  --model-name "text-curie-001" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "en-de" 

python main.py  --model-name "text-davinci-001" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "en-de" 



