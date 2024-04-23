import argparse
import os
import os.path as osp
import re
import sys
import math
import tqdm
import time
import json

import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

import torch
from torch.utils.data import Dataset
from mmengine.utils import mkdir_or_exist
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.utils.dl_utils import set_multi_processing
from peft import PeftModel
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from huggingface_hub import snapshot_download

from xtuner.dataset.utils import decode_base64_to_image, expand2square
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)

sys.path.append('.')

from piechat.utils import load, dump
from piechat.datasets import GeoSceneDataset

TORCH_DTYPE_MAP = dict(fp16=torch.float16,
                       bf16=torch.bfloat16,
                       fp32=torch.float32,
                       auto='auto')


@master_only
def master_print(msg):
    print(msg)


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument('model_name_or_path',
                        help='Hugging Face model name or path')
    parser.add_argument('--dataset-name',
                        default='AID',
                        choices=['AID', 'UCMerced', 'RESISC45'],
                        help='dataset name')
    parser.add_argument('--data-path', default=None, help='data path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument('--llava', default=None, help='llava name or path')
    parser.add_argument('--visual-encoder',
                        default=None,
                        help='visual encoder name or path')
    parser.add_argument('--visual-select-layer',
                        default=-2,
                        help='visual select layer')
    parser.add_argument('--image-folder', default=None, help='image')

    parser.add_argument('--prompt-template',
                        choices=PROMPT_TEMPLATE.keys(),
                        default=None,
                        help='Specify a prompt template')
    parser.add_argument('--stop-words',
                        nargs='+',
                        type=str,
                        default=[],
                        help='Stop words')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument('--bits',
                        type=int,
                        choices=[4, 8, None],
                        default=None,
                        help='LLM bits')
    parser.add_argument('--bot-name',
                        type=str,
                        default='BOT',
                        help='Name for Bot')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed for reproducible text generation')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.launcher != 'none':
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': rank if world_size > 1 else 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }

    with LoadWoInit():
        llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                   **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=True,
                                              encode_special_tokens=True)
    master_print(f'Load LLM from {args.model_name_or_path}')

    llava_path = snapshot_download(
        repo_id=args.llava) if not osp.isdir(args.llava) else args.llava

    # build visual_encoder
    if 'visual_encoder' in os.listdir(llava_path):
        assert args.visual_encoder is None, (
            "Please don't specify the `--visual-encoder` since passed "
            '`--llava` contains a visual encoder!')
        visual_encoder_path = osp.join(llava_path, 'visual_encoder')
    else:
        assert args.visual_encoder is not None, (
            'Please specify the `--visual-encoder`!')
        visual_encoder_path = args.visual_encoder
    with LoadWoInit():
        visual_encoder = CLIPVisionModel.from_pretrained(
            visual_encoder_path, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
        image_processor = CLIPImageProcessor.from_pretrained(
            visual_encoder_path)
    master_print(f'Load visual_encoder from {visual_encoder_path}')

    # load adapter
    if 'llm_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'llm_adapter')
        with LoadWoInit():
            llm = PeftModel.from_pretrained(llm,
                                            adapter_path,
                                            offload_folder=args.offload_folder)
        master_print(f'Load LLM adapter from {args.llava}')
    if 'visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
        visual_encoder = PeftModel.from_pretrained(
            visual_encoder, adapter_path, offload_folder=args.offload_folder)
        master_print(f'Load visual_encoder adapter from {args.llava}')

    # build projector
    projector_path = osp.join(llava_path, 'projector')
    with LoadWoInit():
        projector = AutoModel.from_pretrained(
            projector_path,
            torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype],
            trust_remote_code=True)
    master_print(f'Load projector from {args.llava}')

    projector.cuda()
    projector.eval()

    visual_encoder.cuda()
    visual_encoder.eval()

    llm.eval()

    stop_words = args.stop_words
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(tokenizer=tokenizer,
                                      stop_words=stop_words)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    # work_dir
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        save_dir = args.work_dir
    else:
        # use config filename as default work_dir
        save_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.data_path))[0])
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    save_dir = osp.join(save_dir, timestamp)

    if rank == 0:
        mkdir_or_exist(osp.abspath(save_dir))
        print('=======================================================')
        print(f'Dataset path: {osp.abspath(args.data_path)}\n'
              f'Results will be saved to {osp.abspath(save_dir)}')
        print('=======================================================')

        args_path = osp.join(save_dir, 'args.json')
        with open(args_path, 'w', encoding='utf-8') as f:
            json.dump(args.__dict__, f, indent=2)

    data_stem = osp.splitext(osp.basename(args.data_path))[0]
    results_xlsx_path = osp.join(save_dir,
                                 f'geochat_scene_{data_stem}_result.xlsx')
    results_json_path = osp.join(save_dir,
                                 f'geochat_scene_{data_stem}_result.json')

    #
    dataset = GeoSceneDataset(args.dataset_name, args.data_path,
                              args.image_folder)

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for i in tqdm.tqdm(per_rank_ids, desc=f'Rank {rank}'):
        data_sample = dataset[i]
        text = DEFAULT_IMAGE_TOKEN + '\n' + data_sample['question']

        if args.prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=1, bot_name=args.bot_name)
        else:
            prompt_text = text
        inputs = prompt_text

        image = expand2square(
            data_sample['image'],
            tuple(int(x * 255) for x in image_processor.image_mean))
        image = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0).to(visual_encoder.dtype)

        visual_outputs = visual_encoder(image, output_hidden_states=True)
        pixel_values = projector(
            visual_outputs.hidden_states[args.visual_select_layer][:, 1:])

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = tokenizer.encode(chunk)
            else:
                cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=llm, input_ids=ids, pixel_values=pixel_values)

        generate_output = llm.generate(**mm_inputs,
                                       generation_config=gen_config,
                                       streamer=None,
                                       bos_token_id=tokenizer.bos_token_id,
                                       stopping_criteria=stop_criteria)

        predict = tokenizer.decode(generate_output[0],
                                   skip_special_tokens=True).strip()

        cur_result = {}
        cur_result['question_id'] = data_sample.get('question_id')
        cur_result['question'] = data_sample.get('question')
        cur_result['answer'] = data_sample.get('answer')
        cur_result['prediction'] = predict
        cur_result['image_path'] = data_sample.get('image_path')
        cur_result['category'] = data_sample.get('answer')

        results.append(cur_result)

    results = collect_results(results, n_samples)

    if get_rank() == 0:
        results_df = pd.DataFrame(results)
        with pd.ExcelWriter(results_xlsx_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        if dataset.split == 'dev':
            results_dict = dataset.eval_result(results_df, show=True)
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2)
        else:
            print('All done!')

    return results_json_path


if __name__ == '__main__':
    main()
