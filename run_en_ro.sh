#!/usr/bin/env bash

       
python main.py  --model-name "t5-small" \
                --log-file "./results/t5/prefix/thruthfullqa_en_ro.txt"  \
                --lang-pair "en-ro" 

python main.py  --model-name "t5-base" \
                --log-file "./results/t5/prefix/thruthfullqa_en_ro.txt"  \
                --lang-pair "en-ro" 
python main.py  --model-name "t5-large" \
                --log-file "./results/t5/prefix/thruthfullqa_en_ro.txt"  \
                --lang-pair "en-ro" 

python main.py  --model-name "t5-3b" \
                --log-file "./results/t5/prefix/thruthfullqa_en_ro.txt"  \
                --lang-pair "en-ro" 
            
       
python main.py  --model-name "flan-t5-small" \
                --log-file "./results/flan-t5/prefix/thruthfullqa_en_ro.txt"  \
                --lang-pair "en-ro" \
python main.py  --model-name "flan-t5-base" \
                --log-file "./results/flan-t5/prefix/thruthfullqa_en_ro.txt"  \
                --lang-pair "en-ro" \
python main.py  --model-name "flan-t5-large" \
                --log-file "./results/flan-t5/prefix/thruthfullqa_en_ro.txt"  \
                --lang-pair "en-ro" \

python main.py  --model-name "flan-t5-xl" \
                --log-file "./results/flan-t5/prefix/thruthfullqa_en_ro.txt"  \
                --lang-pair "en-ro" 