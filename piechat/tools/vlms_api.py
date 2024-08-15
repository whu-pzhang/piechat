import copy as cp

from vlmeval.api.base import BaseAPI
from vlmeval.api.qwen_vl_api import QwenVLWrapper as _QwenVLWrapper


class QwenVLWrapper(_QwenVLWrapper):

    def build_msgs(self, msgs_raw, system_prompt=None):
        msgs = cp.deepcopy(msgs_raw)
        ret = []
        if system_prompt is not None:
            content = [(dict(text=system_prompt))]
            ret.append(dict(role='system', content=content))
        content = []
        for msg in msgs:
            if msg['type'] == 'text':
                content.append(dict(text=msg['value']))
            elif msg['type'] == 'image':
                content.append(dict(image='file://' + msg['value']))
        ret.append(dict(role='user', content=content))
        return ret


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    system_prompt = '''You are a powerful remote sensing and aerial image captioner.Please create detailed captions describing the contents of the given image.The caption annotation procedure follows the principles of:
(1): describing the image attributes, including satellite/aerial images, color/panchromatic images, and high/low resolution; 
(2): describing object attributes, including object quantity, color, material, shape, size, and spatial position (including absolute position in the image and relative position between objects); 
(3): generally, the annotation process involves first describing the overall scene of the image, followed by describing specific object. 
(4): Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Do not describe the contents by itemizing them in list form. Minimize aesthetic descriptions as much as possible.
(5): Strictly answer in English.
'''

    msgs = ['00008.jpg']

    vlm_wrapper = QwenVLWrapper(model='qwen-vl-max',
                                system_prompt=system_prompt,
                                verbose=False)
    # inputs = qwen_api.preproc_content(msgs)
    # msg = qwen_api.build_msgs(inputs, system_prompt=system_prompt)
    # print(inputs)
    # print(msg)

    ret = vlm_wrapper.generate(msgs)
