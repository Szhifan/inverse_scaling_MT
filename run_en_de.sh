#!/usr/bin/env bash

       
python main.py  --model-name "babbage" \
                --log-file "./results/openai/prompt2/thruthfullqa_en_de.txt"  \
                --lang-pair "en-de" \
                --few-shot

python main.py  --model-name "curie" \
                --log-file "./results/openai/prompt2/thruthfullqa_en_de.txt"  \
                --lang-pair "en-de" \
                --few-shot

python main.py  --model-name "davinci" \
                --log-file "./results/openai/prompt2/thruthfullqa_en_de.txt"  \
                --lang-pair "en-de" \
                --few-shot

python main.py  --model-name "text-ada-001" \
                --log-file "./results/openai/prompt2/thruthfullqa_en_de.txt"  \
                --lang-pair "en-de" \
   
python main.py  --model-name "text-babbage-001" \
                --log-file "./results/openai/prompt2/thruthfullqa_en_de.txt"  \
                --lang-pair "en-de" \
      
python main.py  --model-name "text-curie-001" \
                --log-file "./results/openai/prompt2/thruthfullqa_en_de.txt"  \
                --lang-pair "en-de" \

python main.py  --model-name "text-davinci-001" \
                --log-file "./results/openai/prompt2/thruthfullqa_en_de.txt"  \
                --lang-pair "en-de" \

python main.py  --model-name "text-davinci-002" \
                --log-file "./results/openai/prompt2/thruthfullqa_en_de.txt"  \
                --lang-pair "en-de" \

python main.py  --model-name "text-davinci-003" \
                --log-file "./results/openai/prompt2/thruthfullqa_en_de.txt"  \
                --lang-pair "en-de" \
