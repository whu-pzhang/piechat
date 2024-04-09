import os
import os.path as osp
import copy as cp

from base_api import BaseAPI
from utils import proxy_set


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
            content = [(dict(text=system_prompt))]
            ret.append(dict(role='system', content=content))
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
        from dashscope import MultiModalConversation
        assert isinstance(inputs, str) or isinstance(inputs, list)
        pure_text = True
        if isinstance(inputs, list):
            for pth in inputs:
                if osp.exists(pth) or pth.startswith('http'):
                    pure_text = False
        assert not pure_text
        messages = self.build_msgs(msgs_raw=inputs,
                                   system_prompt=self.system_prompt)
        print(messages)
        gen_config = dict(max_output_tokens=self.max_tokens,
                          temperature=self.temperature)
        gen_config.update(self.kwargs)
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

    def generate(self, image_path):
        return super(QwenVLAPI, self).generate([image_path])


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    system_prompt = '''You are a powerful remote sensing and aerial image captioner.Please create detailed captions describing the contents of the given image.The caption annotation procedure follows the principles of:
(1): describing the image attributes, including satellite/aerial images, color/panchromatic images, and high/low resolution; 
(2): describing object attributes, including object quantity, color, material, shape, size, and spatial position (including absolute position in the image and relative position between objects); 
(3): generally, the annotation process involves first describing the overall scene of the image, followed by describing specific object. 
(4): Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Do not describe the contents by itemizing them in list form. Minimize aesthetic descriptions as much as possible.
'''

    qwen_api = QwenVLAPI(model='qwen-vl-max',
                         system_prompt=system_prompt,
                         verbose=True)
    img_path = '00012.jpg'

    qwen_api.generate(img_path)
