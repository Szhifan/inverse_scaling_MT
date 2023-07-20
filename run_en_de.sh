#!/usr/bin/env bash
# no few shot for instructgpt!!!        
# python main.py  --model-name "text-davinci-002" \
#                 --log-file "./results/openai/thruthfullqa_en_de.txt"  \
#                 --dataset './datasets/truthfullqa/en_de.df' \
#                 --lang-pair "en-de" 
       
python main.py  --model-name "text-davinci-003" \
                --log-file "./results/openai/thruthfullqa_en_de.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "en-de" 


