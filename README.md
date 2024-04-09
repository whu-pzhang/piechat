# GeoChat-Bench


## finetune

采用 qlora 微调策略，在 llava-internlm2 基础上进行微调。

### 训练流程

LLaVA 训练一共分为两步：对齐模块预训练、指令跟随微调。

预训练的 Projector 默认保存在 ./work_dirs/llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain，并且指令微调阶段将默认在此路径载入 Projector 权重 （iter_2181.pth）。

1. 对齐模块训练（默认保存在 ./work_dirs/）(本次未进行)

```
NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain --deepspeed deepspeed_zero2
```

2. 指令跟随微调（默认保存在 ./work_dirs/）

```
NPROC_PER_NODE=8 xtuner train piechat/configs/geochat_internlm2_7b_qlora_clip_vit_large_p14_336_lora_e1.py --deepspeed deepspeed_zero2
```

### 模型转换与合并

训练完成后，会会的一组权重文件（`iter_xxx.pth`），但这些文件并不是Huggingface格式，需要对齐进行转换。

```
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH
# e.g., convert pth_to_hf piechat/configs/geochat/geochat_internlm2_7b_qlora_clip_vit_large_p14_336_lora_e1.py work_dirs/geochat_internlm2_7b_qlora_clip_vit_large_p14_336_lora_e1/iter_4826.pth work_dirs/geochat_internlm2_7b_qlora_clip_vit_large_p14_336_lora_e1/iter_4826_hf
```

转换完成后的目录如下：

```
work_dirs/geochat_internlm2_7b_qlora_clip_vit_large_p14_336_lora_e1/iter_4826_hf
├── llm_adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── README.md
├── projector
│   ├── config.json
│   ├── configuration_projector.py
│   ├── modeling_projector.py
│   └── model.safetensors
├── visual_encoder_adapter
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── README.md
└── xtuner_config.py
```

之后，如果想要合并 LoRA 至LLM或者 CLIP-ViT中，使用下列命令：

```
# LLM
xtuner convert merge $LLM $LLM_ADAPTER $SAVE_PATH
# xtuner convert merge internlm/internlm2-chat-7b work_dirs/geochat_internlm2_7b_qlora_clip_vit_large_p14_336_lora_e1/iter_4826_hf/llm_adapter/ work_dirs/llava_internlm2_geochat_instruct_ft_qlora_1e/internlm2-chat-7b

# CLIP
xtuner convert merge $CLIP $CLIP_ADAPTER $SAVE_PATH --is-clip
# xtuner convert merge openai/clip-vit-large-patch14-336 work_dirs/geochat_internlm2_7b_qlora_clip_vit_large_p14_336_lora_e1/iter_4826_hf/visual_encoder_adapter work_dirs/llava_internlm2_geochat_instruct_ft_qlora_1e/clip-vit-large-p14-336 --is-clip
```

### 部署



## Scene classification

GeoChat 论文结果：

| Model     | Visual encoder     | LLM       | UCMerced | AID   |
| --------- | ------------------ | --------- | -------- | ----- |
| Qwen-VL   | CLIP ViT-G14       | Qwen-7B   | 62.90    | 52.60 |
| MiniGPTv2 | EVA-G              | LLaMA2-7B | 4.76     | 12.90 |
| LLaVA-1.5 | CLIP ViT-L14/336px | Vicuna-7B | 68.00    | 51.00 |
| GeoChat   | CLIP ViT-L14/504px | Vicuna-7B | 84.43    | 72.03 |

实测：

| Model     | Visual encoder     | LLM            | Schedule    | UCMerced  | AID       |
| --------- | ------------------ | -------------- | ----------- | --------- | --------- |
| LLaVA-1.5 | CLIP ViT-L14/336px | Vicuna-v1.5-7B | -           | 66.33     | 50.53     |
| LLaVA-1.6 | CLIP ViT-L14/336px | Vicuna-v1.5-7B | -           | 61.33     | 52.30     |
| GeoChat   | CLIP ViT-L14/504px | Vicuna-v1.5-7B | -           | **84.48** | **72.03** |
| LLaVA-1.5 | CLIP ViT-L14/336px | InternLM2-7B   | -           | 76.95     | 60.83     |
| LLaVA-1.5 | CLIP ViT-L14/336px | InternLM2-7B   | qlora_ft-1e | **91.76** | **70.73** |


## Visual Question Answering

### RSVQA-HR

| Model     | Visual encoder     | LLM            | Schedule    | Presence  | Comparison | Average Accuracy |
| --------- | ------------------ | -------------- | ----------- | --------- | ---------- | ---------------- |
| Qwen-VL   |                    |                | -           | 66.44     | 60.41      | 63.06            |
| LLaVA-1.5 |                    |                | -           | 69.83     | 67.29      | 68.40            |
| MiniGPTv2 |                    |                | -           | 40.79     | 50.91      | 46.46            |
| GeoChat   | CLIP ViT-L14/504px | Vicuna-v1.5-7B | -           | 58.45     | 83.19      | 72.30            |
| LLaVA-1.5 | CLIP ViT-L14/336px | InternLM2-7B   | -           | 60.23     | 69.28      | 65.37            |
| LLaVA-1.5 | CLIP ViT-L14/336px | InternLM2-7B   | qlora_ft-1e | **62.52** | **80.25**  | **72.59**        |

### RSVQA-LR

| Model     | Visual encoder     | LLM            | Schedule    | Presence  | Comparison | Rural/Urban | Average Accuracy |
| --------- | ------------------ | -------------- | ----------- | --------- | ---------- | ----------- | ---------------- |
| Qwen-VL   |                    |                | -           | 38.57     | 67.59      | 61.00       | 55.35            |
| LLaVA-1.5 |                    |                | -           | 55.46     | 68.20      | 59.00       | 62.77            |
| MiniGPTv2 |                    |                | -           | 55.16     | 55.22      | 39.00       | 54.96            |
| GeoChat   | CLIP ViT-L14/504px | Vicuna-v1.5-7B | -           | 91.09     | 90.33      | 94.00       | 90.70            |
| LLaVA-1.5 | CLIP ViT-L14/336px | InternLM2-7B   | -           | 68.53     | 67.17      | 64.00       | 66.57            |
| LLaVA-1.5 | CLIP ViT-L14/336px | InternLM2-7B   | qlora_ft-1e | **91.23** | **92.38**  | **96.00**   | **93.20**        |

## Visual Grounding


