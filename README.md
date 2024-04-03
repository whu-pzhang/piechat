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


