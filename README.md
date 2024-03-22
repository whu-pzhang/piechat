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

| Model     | Visual encoder     | LLM            | UCMerced | AID   |
| --------- | ------------------ | -------------- | -------- | ----- |
| LLaVA-1.5 | CLIP ViT-L14/336px | Vicuna-v1.5-7B | 66.33    | 50.53 |
| LLaVA-1.6 | CLIP ViT-L14/336px | Vicuna-v1.5-7B | 61.33    | 52.30 |
| GeoChat   | CLIP ViT-L14/504px | Vicuna-v1.5-7B |          |       |
| LLaVA-1.5 | CLIP ViT-L14/336px | InternLM2-7B   | 76.95    |       |