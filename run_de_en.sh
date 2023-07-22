#!/usr/bin/env bash
#+====================READ BEFORE RUNNING=========
# no few shot for instructgpt!!! 
# check if commands are CORRECT!!!!!! 
#+====================READ BEFORE RUNNING=========      
python main.py  --model-name "ada" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en" \
                --few-shot
       
python main.py  --model-name "babbage" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en" \
                --few-shot

       
python main.py  --model-name "curie" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en" \
                --few-shot

python main.py  --model-name "davinci" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en" \
                --few-shot

python main.py  --model-name "text-ada-001" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en"    
python main.py  --model-name "text-babbage-001" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en"       
python main.py  --model-name "text-curie-001" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en" 

python main.py  --model-name "text-davinci-001" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en" 

python main.py  --model-name "text-davinci-002" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en"

python main.py  --model-name "text-davinci-003" \
                --log-file "./results/openai/thruthfullqa_de_en.txt"  \
                --dataset './datasets/truthfullqa/en_de.df' \
                --lang-pair "de-en"  