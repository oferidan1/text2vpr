import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import math
import time
import glob

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def load_intervl(model_path):
    # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
    # If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
    #path = 'OpenGVLab/InternVL3-8B'
    path = 'OpenGVLab/InternVL3-38B'
    #device_map = split_model('InternVL3-8B')
    device_map = split_model(path)
    #device_map = 'cuda:0'  # Use a single GPU for simplicity in this example
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    return model, tokenizer
    
def run_intern_vl(model, tokenizer, image_path):
    # set the max number of tiles in `max_num`
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # single-image single-round conversation
    image_paths = [image_path] * 1

    pixel_values = [load_image(image_path, max_num=12).to(torch.bfloat16).cuda() for image_path in image_paths]
    pixel_values = torch.cat(pixel_values, dim=0)

    question = '<image>\ndescribe all objects in this image from left to right in one line, including their attributes and colors, ignore dynamic objects like people and cars. in your response, use the format: object1, object2, object3, ...'
    question = '<image>\ndescribe this location for visual place recognition. Focus on: 1) Scene type and setting, 2) Distinctive landmarks and architecture, 3) Unique visual patterns/colors/textures, 4) Spatial layout, 5) Key identifying features that distinguish this place from similar locations. Be specific about permanent visual elements, avoid temporary objects like people, car ,weather and lighting conditions. the output is one line of text listing the items from left to right, separated by commas.'    
    for i in range(1):
        t1 = time.time()
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        t2 = time.time()
        print(f'Inference time: {t2 - t1:.2f} seconds')
        print(response)


def describe_all_images(image_folder):
    image_paths = glob.glob(f"{image_folder}/*.jpg")
    for image_path in image_paths:
        run_gemini(image_path)

    
if __name__ == '__main__':
    #image_path = 'images/dizengoff.webp'
    #path = 'OpenGVLab/InternVL3-8B'
    model_path = 'OpenGVLab/InternVL3-38B'
    
    model, tokenzier = load_intervl(model_path)
    
    image_path = 'images/@0543158.27@4180593.76@10@S@037.77166@-122.50995@M7mOh9X4Xw_OHp-DYe5hQg@@206@@@@201311@@.jpg'
    image_path = '/mnt/d/data/sf_xl/small/dummy/@0544204.32@4173406.33@10@S@037.70683@-122.49851@TYcjxIohRl--XFaR4OgdxA@@0@@@@201910@@.jpg'
    run_intern_vl(model, tokenzier, image_path)
    image_path = '/mnt/d/data/sf_xl/small/dummy/@0544204.34@4173404.11@10@S@037.70681@-122.49851@ubFEPvi18FX7ZwRKzwY6JA@@0@@@@201107@@.jpg'
    run_intern_vl(model, tokenzier, image_path)
    # image_folder = 'mnt/d/data/sf_xl/small/test'
    # describe_all_images(image_folder)