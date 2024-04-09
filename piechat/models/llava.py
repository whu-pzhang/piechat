from collections import OrderedDict

import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training

from xtuner.registry import BUILDER
from xtuner.model.modules import ProjectorConfig, ProjectorModel, dispatch_modules
from xtuner.model.utils import (LoadWoInit, find_all_linear_names,
                                get_peft_model_state_dict,
                                guess_load_checkpoint,
                                make_inputs_require_grad,
                                prepare_inputs_labels_for_multimodal,
                                traverse_dict)
from transformers import GenerationConfig
from xtuner.tools.utils import get_stop_criteria
from xtuner.model import LLaVAModel


class GLLaVAModel(LLaVAModel):

    def load_custom_weights(self, pretrained_pth):
        pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
        self.load_state_dict(pretrained_state_dict, strict=False)
        print(f'Load pretrained weight from {pretrained_pth}')

    def preparing_eval(self, eval_dataset, max_new_tokens=100):
        self.tokenizer = eval_dataset.tokenizer
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )
        self.gen_config = gen_config
        stop_words = []
        stop_words += eval_dataset.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(tokenizer=self.tokenizer,
                                          stop_words=stop_words)
        self.stop_criteria = stop_criteria

    def generate(self, data, data_samples=None):
        # data 是单张图片的数据
        data.pop('id', None)
        input_ids = data['input_ids'].unsqueeze(0).to(
            self.visual_encoder.device)
        data['input_ids'] = input_ids
        pixel_values = data['pixel_values'].unsqueeze(0).to(
            self.visual_encoder.device)

        visual_outputs = self.visual_encoder(pixel_values.to(
            self.visual_encoder.dtype),
                                             output_hidden_states=True)
        pixel_values = self.projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
        data['pixel_values'] = pixel_values
        mm_inputs = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria)

        predict = self.tokenizer.decode(generate_output[0],
                                        skip_special_tokens=True).strip()

        return predict
