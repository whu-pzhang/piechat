## Training

### Pretrain

```
NPROC_PER_NODE=8 xtuner train llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain --deepspeed deepspeed_zero2 --seed 1024
```

### Fine-tune

```
NPROC_PER_NODE=8 xtuner train llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune --deepspeed deepspeed_zero2 --seed 1024
```

### Single card

XTuner also supports single-card training for LLaVA-Llama-3-8B (Youth Edition), requiring only a single card with 20GB to complete the entire process of multi-modal training.

#### Pretrain (saved by default in `./work_dirs/llava_llama3_8b_instruct_quant_clip_vit_large_p14_336_e1_gpu1_pretrain/`)

```
xtuner train llava_llama3_8b_instruct_quant_clip_vit_large_p14_336_e1_gpu1_pretrain --deepspeed deepspeed_zero2 --seed 1024
```

#### Fine-tune (saved by default in `./work_dirs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e1_gpu1_finetune/`)

```
xtuner train llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_e1_gpu1_finetune --deepspeed deepspeed_zero2 --seed 1024
```


## Model convert(and Merge)


After training, we will obtain a set of weights (i.e., iter_xxx.pth), which are not in the universal HuggingFace format. We first need to convert them.

```
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH
# e.g., xtuner convert pth_to_hf llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune ./iter_39620.pth ./iter_39620_hf
```

At this point, we have obtained the relevant model (LLM or the corresponding LoRA). If you use the default configuration of LLaVA-Llama-3-8B, you will obtain the following file structure after converting. It includes the full-finetuned LLM weights, projector weights, and LoRA weights of the visual encoder.

```
├── llm_adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── README.md
├── projector
│   ├── config.json
│   ├── configuration_projector.py
│   ├── modeling_projector.py
│   └── model.safetensors
└── xtuner_config.py
```

## chat

We can achieve image-text question answering with the following command!

```
xtuner chat /data1/hf_hub/Meta-Llama-3-8B-Instruct \
    --visual-encoder openai/clip-vit-large-patch14-336 \
    --llava ./work_dirs/llava_llama3_8b_instruct_qlora_clip_large_p14_336_e1_gpu1_finetune/iter_9652_hf --prompt-template llama3_chat \
    --image 00012.jpg
```



