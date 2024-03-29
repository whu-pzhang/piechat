# GeoChat-Bench


## Scene classification

GeoChat 论文结果：

| Model     | Visual encoder     | LLM       | UCMerced | AID   |
| --------- | ------------------ | --------- | -------- | ----- |
| Qwen-VL   | CLIP ViT-G14       | Qwen-7B   | 62.90    | 52.60 |
| MiniGPTv2 | EVA-G              | LLaMA2-7B | 4.76     | 12.90 |
| LLaVA-1.5 | CLIP ViT-L14/336px | Vicuna-7B | 68.00    | 51.00 |
| GeoChat   | CLIP ViT-L14/504px | Vicuna-7B | 84.43    | 72.03 |

实测：

| Model     | Visual encoder     | LLM            | UCMerced  | AID       |
| --------- | ------------------ | -------------- | --------- | --------- |
| LLaVA-1.5 | CLIP ViT-L14/336px | Vicuna-v1.5-7B | 66.33     | 50.53     |
| LLaVA-1.6 | CLIP ViT-L14/336px | Vicuna-v1.5-7B | 61.33     | 52.30     |
| GeoChat   | CLIP ViT-L14/504px | Vicuna-v1.5-7B | **84.48** | **72.03** |
| LLaVA-1.5 | CLIP ViT-L14/336px | InternLM2-7B   | 76.95     | 60.83     |


## Visual Question Answering

### RSVQA-HR

| Model     | Visual encoder     | LLM          | Presence | Comparison | Average Accuracy |
| --------- | ------------------ | ------------ | -------- | ---------- | ---------------- |
| Qwen-VL   |                    |              | 66.44    | 60.41      | 63.06            |
| LLaVA-1.5 |                    |              | 69.83    | 67.29      | 68.40            |
| MiniGPTv2 |                    |              | 40.79    | 50.91      | 46.46            |
| GeoChat   |                    |              | 58.45    | 83.19      | 72.30            |
| LLaVA-1.5 | CLIP ViT-L14/336px | InternLM2-7B | 60.23    | 69.28      | 65.37            |

### RSVQA-LR


