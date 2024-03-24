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

TORCH_DTYPE_MAP = dict(fp16=torch.float16,
                       bf16=torch.bfloat16,
                       fp32=torch.float32,
                       auto='auto')


def load_jsonl(data_path):
    lines = open(data_path, 'r', encoding='utf-8').readlines()
    lines = [x.strip() for x in lines]
    if lines[-1] == '':
        lines = lines[:-1]
    data = [json.loads(x) for x in lines]
    return data


@master_only
def master_print(msg):
    print(msg)


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument('model_name_or_path',
                        help='Hugging Face model name or path')
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


class RSVQADataset(Dataset):

    def __init__(self, data_file, image_folder):
        self.data_file = data_file
        self.image_folder = image_folder
        self.df = load_jsonl(data_file)

    def load_image(self, image_file):
        if image_file.startswith('http://') or image_file.startswith(
                'https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        line = self.df[idx]
        question_id = line['question_id']
        image_file = osp.join(self.image_folder, line['image'])
        image = self.load_image(image_file)
        question = line['text']
        # gt = line['ground_truth']

        data = {
            'question_id': question_id,
            'image': image,
            'image_path': line['image'],
            'question': question,
            # 'ground_truth': gt,
        }

        return data

    @master_only
    def eval_result(self, result, show=True):

        def calc_acc(df, group='category'):
            assert group in ['overall', 'category']


def eval_metrics(data_path):
    base = [json.loads(q) for q in open(data_path, "r")]
    correct = 0
    incorrect = 0
    for answers in tqdm.tqdm(base):
        gt = answers['ground_truth'].lower()
        answer = answers['answer'].lower().strip()
        if gt == answer:
            correct += 1
        else:
            incorrect += 1

    print('correct:', correct)
    print('incorrect:', incorrect)
    print('Total:', correct + incorrect)
    print('Acc:', (correct / (correct + incorrect)))


def eval_model():
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
    results_jsonl_path = osp.join(save_dir, f'{data_stem}.jsonl')

    #
    dataset = SceneDataset(args.data_path, args.image_folder)

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for i in tqdm.tqdm(per_rank_ids, desc=f'Rank {rank}'):
        data_sample = dataset[i]
        text = DEFAULT_IMAGE_TOKEN + '\n' + data_sample['question']

        # if is_cn_string(text):
        #     text = text + '请直接回答选项字母。'
        # else:
        #     text = text + ("Answer with the option's letter from the "
        #                    'given choices directly.')
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
        cur_result['image_path'] = data_sample.get('image_path')
        cur_result['ground_truth'] = data_sample.get('ground_truth')
        cur_result['answer'] = predict

        results.append(cur_result)

    results = collect_results(results, n_samples)

    if get_rank() == 0:
        with jsonlines.open(results_jsonl_path, 'w') as writer:
            writer.write_all(results)

        print('All done!')

    return results_jsonl_path


if __name__ == '__main__':
    result_file = eval_model()
    eval_metrics(result_file)
