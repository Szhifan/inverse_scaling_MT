#!/bin/sh

for few_shot in "" "--few-shot"; do
for quantization in None 4-bits 8-bits; do
for chat in "" "-chat"; do
for src_lang in de fr ro ru; do
for model_size in 7b 13b 70b; do
      python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/thruthfullqa_${src_lang}_en.txt" --lang-pair "${src_lang}-en" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token
      python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/prefix/thruthfullqa_${src_lang}_en.txt" --lang-pair "${src_lang}-en" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token --prefix
done
done
done
done
done
