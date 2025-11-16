"""
Processor class for LLAVA_QWEN2
"""

from builtins import NotImplementedError

from typing import Optional, Union, List
import torch
import numpy as np

import PIL

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput

from transformers import CLIPImageProcessor
from src.model.fastvlm.mm_utils import tokenizer_image_token, IMAGE_TOKEN_INDEX, expand2square
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from src.utils.basic_utils import print_master

def to_rgb(image: ImageInput):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def concat_images_vertical(images: list):
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = sum(heights)

    new_img = PIL.Image.new(images[0].mode, (max_width, total_height))
    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.size[1]
    return new_img


class LlavaQwen2Processor(ProcessorMixin):
    r"""
    Constructs a LlavaQwen2 processor which wraps a LlavaQwen2 image processor and a Qwen2 tokenizer into a single processor.
    Args:
        image_processor (, *optional*):
            The image processor is a required input.
        tokenizer (, *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """
    image_processor_class = "CLIPImageProcessor"
    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        super().__init__(image_processor, tokenizer)
    
    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[List, torch.Tensor]] = None,
        **kwargs
    ):
        tokenizer = self.tokenizer
        image_processor = self.image_processor

        if not isinstance(text, list):
            text = [text]
        
        input_ids = [tokenizer_image_token(t, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for t in text]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

        # attention mask
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        batch_images = []
        if images is not None:
            for imgs in images:
                if imgs is None or all(i is None for i in imgs):
                    # tạo placeholder image tensor
                    H, W = image_processor.crop_size['height'], image_processor.crop_size['width']
                    placeholder = torch.zeros(3, H, W)
                    batch_images.append(placeholder)
                else:
                    imgs = [to_rgb(i) for i in imgs]
                    # concat images in sample
                    imgs = concat_images_vertical([i for i in imgs if i is not None])
                    try:
                        img_tensor = image_processor.preprocess(
                            imgs, 
                            input_data_format="channels_last",
                            return_tensors='pt'
                        )['pixel_values'][0]
                    except Exception as e:
                        print_master("=== ERROR in image preprocessing ===")
                        print_master(f"Exception: {e}")
                        print_master(f"Type of imgs: {type(imgs)}")
                        if isinstance(imgs, PIL.Image.Image):
                            print_master(f"Image mode: {imgs.mode}")
                            print_master(f"Image size: {imgs.size}")
                            print_master(f"Image array shape: {np.array(imgs).shape}")
                        else:
                            print_master(f"Imgs content: {imgs}")
                        raise e  # terminate chương trình sau khi in debug info
                    batch_images.append(img_tensor)
            images = torch.stack(batch_images)
        else:
            images = None

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            }
        
        if images is not None:
            data["images"] = images

        return BatchFeature(data=data, tensor_type="pt")

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )