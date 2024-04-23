import torch

from xtuner.utils import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from xtuner.dataset.utils import expand2square
from transformers import TextIteratorStreamer


def process_images(images, image_processor):
    new_images = []
    for image in images:
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean))
        image = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        new_images.append(image)

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt,
                          tokenizer,
                          image_token_index=IMAGE_TOKEN_INDEX,
                          return_tensors=None):
    prompt_chunks = [
        tokenizer(chunk).input_ids
        for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X))
                for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(
            prompt_chunks[0]
    ) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks,
                              [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def tokenizer_image_token_v2(prompt,
                             tokenizer,
                             image_token_index=IMAGE_TOKEN_INDEX,
                             return_tensors=None):
    chunk_encode = []
    for idx, chunk in enumerate(prompt.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
        # assert len(chunk_encode) == 2
    input_ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        input_ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            input_ids.append(image_token_index)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


if __name__ == '__main__':
    from transformers import AutoTokenizer
    model_name_or_path = "internlm/internlm2-chat-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              trust_remote_code=True,
                                              encode_special_tokens=True)
    prompt = "<image> Describe the image."

    ret1 = tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
    print(ret1)

    ret2 = tokenizer_image_token_v2(prompt, tokenizer, return_tensors='pt')
    print(ret2)
