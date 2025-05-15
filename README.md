# DTR
 The Official Implementation of the paper "Robustifying Vision-Language Models via Dynamic Token Reweighting" (DTR)

Please use the same environment as Llava (https://github.com/haotian-liu/LLaVA) to replicate the result.

A quick demo of DTR and its visualization of alpha is available in demo+Visualization.ipynb. 

To run a quick test on defending Llava-llama2 against HADES, please use:

```bash
CUDA_VISIBLE_DEVICES=0 python HADES_LLaVA_Llama2_with_DTR.py
```