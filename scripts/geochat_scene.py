import argparse
import json
import os
import torch
import jsonlines
from tqdm import tqdm
from pathlib import Path

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device)

    #
    questions = [
        json.loads(q)
        for q in open(os.path.expanduser(args.question_file), "r")
    ]

    # output
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    qs_stem = Path(args.question_file).stem
    output_file = output_folder / f'{qs_stem}_{model_name}.jsonl'
    output_fp = jsonlines.open(output_file, 'w')

    for idx, qs in enumerate(tqdm(questions)):
        img_file = os.path.join(args.image_folder, qs['image'])
        image = load_image(img_file)
        image_size = image.size

        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [
                image.to(model.device, dtype=torch.float16)
                for image in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = qs['text']
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt,
                                          tokenizer,
                                          IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).to(
                                              model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        keywords = [stop_str]
        streamer = TextStreamer(tokenizer,
                                skip_prompt=True,
                                skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(input_ids,
                                        images=image_tensor,
                                        do_sample=False,
                                        temperature=args.temperature,
                                        max_new_tokens=args.max_new_tokens,
                                        use_cache=True)
        outputs = tokenizer.decode(output_ids[0],
                                   skip_special_tokens=True).strip()

        ans = dict(question_id=qs['question_id'],
                   image_id=qs['image'],
                   answer=outputs,
                   ground_truth=qs['ground_truth'])
        output_fp.write(ans)
    output_fp.close()

    return output_file


def eval_metrics(data_path):
    base = [json.loads(q) for q in open(data_path, "r")]
    correct = 0
    incorrect = 0
    for answers in tqdm(base):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file",
                        type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--output-folder", type=str, default=".")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    output_file = eval_model(args)
    eval_metrics(output_file)
