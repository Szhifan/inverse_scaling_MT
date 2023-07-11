#!/usr/bin/env bash

# python main.py  --model-name "ada" \
#                 --log-file "./results/openai_models/thruthfullqa_de_en.txt"  \
#                 --dataset './datasets/truthfullqa/en_de.df' \
#                 --few-shot True \
#                 --lang-pair "de-en"
python main.py  --model-name "babbage" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot True \
                --lang-pair "de-en"
python main.py  --model-name "curie" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot True \
                --lang-pair "de-en"
python main.py  --model-name "davinci" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot True \
                --lang-pair "de-en"

python main.py  --model-name "text-ada-001" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot False \
                --lang-pair "de-en"
python main.py  --model-name "text-ada-001" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot False \
                --lang-pair "de-en"
python main.py  --model-name "text-ada-001" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot False \
                --lang-pair "de-en"
python main.py  --model-name "text-ada-001" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot False \
                --lang-pair "de-en"
python main.py  --model-name "text-ada-001" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot False \
                --lang-pair "de-en"
python main.py  --model-name "text-ada-001" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot False \
                --lang-pair "de-en"

python main.py  --model-name "text-ada-001" \
                --log-file "./results/openai_models/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --few-shot False \
                --lang-pair "de-en"

