# 遥感测绘多模态大模型

## 已有模型测试

| Model    | Image | Question | Answer | Date |
| -------- | ----- | -------- | ------ | ---- |
| ChatGLM4 |       |          |        |      |
| Qwen     |       |          |        |      |
| InternVL |       |          |        |      |




## 数据构建

构建多模态数据集基本上借助于现有公开数据集和商用大模型API，常见的构建方法有：

1. 利用纯语言大模型（GPT-4或ChatGPT），接受精心设计的文本prompt，来创建视觉内容的指令微调数据。典型例子有：LLaVA
2. 利用多模态大模型(GPT-4V等)，创建数据集。如ShareGPT4
3. SoM，借助已有数据集，打上标记后，再利用多模态大模型(GPT-4V)来生成高质量数据集。



## 参考

- [LLaVA](https://arxiv.org/abs/2304.08485)
- [ShareGPT4](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V)
- [SoM](https://github.com/microsoft/SoM)
