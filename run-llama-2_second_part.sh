#!/bin/sh

for chat in ""; do
for few_shot in ""; do
for model_size in 70b; do
for quantization in 4-bits 8-bits; do
for tgt_lang in de fr ro; do
      python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/thruthfullqa_en_${tgt_lang}.txt" --lang-pair "en-${tgt_lang}" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token
     python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/prefix/thruthfullqa_en_${tgt_lang}.txt" --lang-pair "en-${tgt_lang}" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token --prefix
done
done
done
done
done

for chat in ""; do
for few_shot in "--few-shot"; do
for model_size in 7b 13b 70b; do
for quantization in None 4-bits 8-bits; do
for tgt_lang in de fr ro; do
      python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/thruthfullqa_en_${tgt_lang}.txt" --lang-pair "en-${tgt_lang}" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token
     python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/prefix/thruthfullqa_en_${tgt_lang}.txt" --lang-pair "en-${tgt_lang}" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token --prefix
done
done
done
done
done

for chat in "-chat"; do
for few_shot in "" "--few-shot"; do
for model_size in 7b 13b 70b; do
for quantization in None 4-bits 8-bits; do
for tgt_lang in de fr ro; do
      python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/thruthfullqa_en_${tgt_lang}.txt" --lang-pair "en-${tgt_lang}" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token
     python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/prefix/thruthfullqa_en_${tgt_lang}.txt" --lang-pair "en-${tgt_lang}" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token --prefix
done
done
done
done
done

