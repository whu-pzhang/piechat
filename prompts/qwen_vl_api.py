import os
import os.path as osp
import time
import random as rd
import copy as cp
from abc import abstractmethod

import dashscope
from dashscope import MultiModalConversation

import time
import random as rd
from abc import abstractmethod
import logging


class BaseAPI:

    def __init__(self,
                 retry=10,
                 wait=3,
                 system_prompt=None,
                 verbose=True,
                 fail_msg='Failed to obtain answer via API.',
                 **kwargs):
        self.wait = wait
        self.retry = retry
        self.system_prompt = system_prompt
        self.kwargs = kwargs
        self.verbose = verbose
        self.fail_msg = fail_msg
        self.logger = get_logger('ChatAPI')
        if len(kwargs):
            self.logger.info(
                f'BaseAPI received the following kwargs: {kwargs}')
            self.logger.info('Will try to use them as kwargs for `generate`. ')

    @abstractmethod
    def generate_inner(self, inputs, **kwargs):
        self.logger.warning(
            'For APIBase, generate_inner is an abstract method. ')
        assert 0, 'generate_inner not defined'
        ret_code, answer, log = None, None, None
        # if ret_code is 0, means succeed
        return ret_code, answer, log

    def generate(self, inputs, **kwargs):
        input_type = None
        if isinstance(inputs, str):
            input_type = 'str'
        elif isinstance(inputs, list) and isinstance(inputs[0], str):
            input_type = 'strlist'
        elif isinstance(inputs, list) and isinstance(inputs[0], dict):
            input_type = 'dictlist'
        assert input_type is not None, input_type

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        for i in range(self.retry):
            try:
                ret_code, answer, log = self.generate_inner(inputs, **kwargs)
                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    return answer
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except:
                            self.logger.warning(
                                f'Failed to parse {log} as an http response. ')
                    self.logger.info(
                        f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}:')
                    self.logger.error(err)
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)

        return self.fail_msg if answer in ['', None] else answer


logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    except ImportError:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger


def proxy_set(s):
    for key in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
        os.environ[key] = s


class QwenVLWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'qwen-vl-plus',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 **kwargs):

        assert model in ['qwen-vl-plus', 'qwen-vl-max']
        self.model = model
        import dashscope
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        if key is None:
            key = os.environ.get('DASHSCOPE_API_KEY', None)
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        dashscope.api_key = key
        if proxy is not None:
            proxy_set(proxy)
        super().__init__(wait=wait,
                         retry=retry,
                         system_prompt=system_prompt,
                         verbose=verbose,
                         **kwargs)

    @staticmethod
    def build_msgs(msgs_raw, system_prompt=None):
        msgs = cp.deepcopy(msgs_raw)
        ret = []
        if system_prompt is not None:
            ret.append(dict(role='system', content=[dict(text=system_prompt)]))
        content = []
        for i, msg in enumerate(msgs):
            if osp.exists(msg):
                content.append(dict(image='file://' + msg))
            elif msg.startswith('http'):
                content.append(dict(image=msg))
            else:
                content.append(dict(text=msg))
        ret.append(dict(role='user', content=content))
        return ret

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        pure_text = True
        if isinstance(inputs, list):
            for pth in inputs:
                if osp.exists(pth) or pth.startswith('http'):
                    pure_text = False
        assert not pure_text
        messages = self.build_msgs(msgs_raw=inputs,
                                   system_prompt=self.system_prompt)
        gen_config = dict(max_output_tokens=self.max_tokens,
                          temperature=self.temperature)
        gen_config.update(self.kwargs)
        print(gen_config)

        try:
            response = MultiModalConversation.call(model=self.model,
                                                   messages=messages)
            if self.verbose:
                print(response)
            answer = response.output.choices[0]['message']['content'][0][
                'text']
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(err)
                self.logger.error(f'The input messages are {inputs}.')

            return -1, '', ''


class QwenVLAPI(QwenVLWrapper):

    def generate(self, image_path, prompt, dataset=None):
        return super(QwenVLAPI, self).generate([image_path, prompt])

    def interleave_generate(self, ti_list, dataset=None):
        return super(QwenVLAPI, self).generate(ti_list)


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    system_prompt = '''You are a powerful remote sensing and aerial image captioner.Please create detailed captions describing the contents of the given image.The caption annotation procedure follows the principles of:
(1): describing the image attributes, including satellite/aerial images, color/panchromatic images, and high/low resolution; 
(2): describing object attributes, including object quantity, color, material, shape, size, and spatial position (including absolute position in the image and relative position between objects); 
(3): generally, the annotation process involves first describing the overall scene of the image, followed by describing specific object. 
(4): Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Do not describe the contents by itemizing them in list form. Minimize aesthetic descriptions as much as possible.
'''

    qwen_api = QwenVLAPI(verbose=True, system_prompt=system_prompt)
    img_path = '00012.jpg'

    qwen_api.generate(img_path, '')
