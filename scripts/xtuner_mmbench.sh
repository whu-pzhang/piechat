export CUDA_VISIBLE_DEVICES=0,1,2,3

model_path=internlm/internlm2-chat-7b
ve_path=openai/clip-vit-large-patch14-336
llava_path=xtuner/llava-internlm2-7b

xtuner mmbench ${model_path} \
  --visual-encoder ${ve_path} \
  --llava ${llava_path} \
  --prompt-template internlm2_chat \
  --data-path ./data/mmbench/MMBench_DEV_EN.tsv \
  --work-dir ./work_dirs/offical_llava-internlm2-7b/MMBench_DEV_EN
