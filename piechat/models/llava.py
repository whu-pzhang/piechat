from collections import OrderedDict

import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import GenerationConfig

from xtuner.registry import BUILDER
from xtuner.model.modules import ProjectorConfig, ProjectorModel, dispatch_modules
from xtuner.model.utils import (LoadWoInit, find_all_linear_names,
                                get_peft_model_state_dict,
                                guess_load_checkpoint,
                                make_inputs_require_grad,
                                prepare_inputs_labels_for_multimodal,
                                traverse_dict)

from xtuner.model.llava import LLaVAModel

from xtuner.tools.utils import get_stop_criteria


class GLLaVAModel(LLaVAModel):

    def generate(self, data, data_samples=None):
        pass
