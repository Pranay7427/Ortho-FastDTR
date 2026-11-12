import os
import torch
import json
import argparse
import random
import io
import functools
import gc
import requests
import pandas as pd
import numpy as np
import cv2  # for resizing heatmap
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
##############################
# Global settings and device
##############################
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global alpha

##############################
# Helper Functions
##############################
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def travel_total_distance(B, C):
    displacement = B - C
    total_distance = torch.norm(displacement)
    return abs(total_distance)

def travel_orthogonal(A, B, C):
    displacement = B - C
    scalar_along_A = torch.dot(displacement, A)
    projection_along_A = scalar_along_A * A
    orthogonal_component = displacement - projection_along_A
    return torch.norm(orthogonal_component)

def travel_scalar(A, B, C):
    displacement = B - C
    scalar_travel = torch.dot(displacement, A)
    return scalar_travel

def project_vector(A, B, C):
    displacement = B - C
    scalar_proj = torch.dot(displacement, A)
    projection_vector = scalar_proj * A
    return projection_vector

safe_instruction = "You are a responsible AI and should refuse anything that is illegal or unethical. Here is the instruction: \n"

def get_harmless_instructions(model_name):
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)
    instructions = [example['instruction'] for example in dataset['train'] if example['input'].strip() == '']
    random.shuffle(instructions)
    instructions = instructions[:32]
    prompts = []
    from llava.conversation import conv_templates
    for inst in instructions:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], safe_instruction +inst)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
    return prompts

def get_harmful_text_instructions_hades(model_name):
    dataset = load_dataset("Monosail/HADES")['test']
    indices = random.sample(range(len(dataset)), 32)
    instructions = [dataset[i]['instruction'] for i in indices]
    prompts = []
    from llava.conversation import conv_templates
    for inst in instructions:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], safe_instruction + inst)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
    return prompts

def get_toks_and_images(model_name, new_inst, image, tokenizer_mm, image_processor):
    indices = [0]
    toks_list = []
    images_list = []
    sizes_list = []
    pil_images_list = []
    # Import constants and conversation template inside the function
    from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    for idx in indices:
        pil_img = image
        pil_images_list.append(pil_img)
        sizes_list.append(pil_img.size)  # (width, height)
        inst = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + new_inst
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], safe_instruction + inst)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        toks = tokenizer_image_token(prompt, tokenizer_mm, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
        toks_list.append(toks)
        image_tensor = image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"].half().to(DEVICE)
        images_list.append(image_tensor)
    max_len = max(toks.shape[1] for toks in toks_list)
    padded_toks_list = []
    for toks in toks_list:
        pad_length = max_len - toks.shape[1]
        if pad_length > 0:
            pad_tensor = torch.full((toks.shape[0], pad_length),
                                    tokenizer_mm.pad_token_id,
                                    dtype=toks.dtype,
                                    device=toks.device)
            padded_toks = torch.cat([toks, pad_tensor], dim=1)
        else:
            padded_toks = toks
        padded_toks_list.append(padded_toks)
    toks_batch = torch.cat(padded_toks_list, dim=0)
    images_batch = torch.cat(images_list, dim=0)
    clear_gpu_memory()
    return toks_batch, images_batch, sizes_list, pil_images_list

def get_harmful_multimodal_examples(model_name,tokenizer_mm, image_processor, num_examples: int = 32):
    dataset = load_dataset("Monosail/HADES")['test']
    indices = random.sample(range(len(dataset)), num_examples)
    toks_list = []
    images_list = []
    sizes_list = []
    pil_images_list = []
    from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    for idx in indices:
        item = dataset[idx]
        pil_img = item['image'].convert('RGB')
        pil_images_list.append(pil_img)
        sizes_list.append(pil_img.size)
        inst = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + item['instruction']
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], inst)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        toks = tokenizer_image_token(prompt, tokenizer_mm, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
        toks_list.append(toks)
        image_tensor = image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"].half().to(DEVICE)
        images_list.append(image_tensor)
    max_len = max(toks.shape[1] for toks in toks_list)
    padded_toks_list = []
    for toks in toks_list:
        pad_length = max_len - toks.shape[1]
        if pad_length > 0:
            pad_tensor = torch.full((toks.shape[0], pad_length),
                                    tokenizer_mm.pad_token_id,
                                    dtype=toks.dtype,
                                    device=toks.device)
            padded_toks = torch.cat([toks, pad_tensor], dim=1)
        else:
            padded_toks = toks
        padded_toks_list.append(padded_toks)
    toks_batch = torch.cat(padded_toks_list, dim=0)
    images_batch = torch.cat(images_list, dim=0)
    clear_gpu_memory()
    return toks_batch, images_batch, sizes_list, pil_images_list

def calculate_average(numbers):
    """
    Calculate the average of a list of numbers by summing them one by one and dividing by the length.
    
    Parameters:
        numbers (list of float): The list of numbers.
    
    Returns:
        float: The average of the numbers. Returns 0 if the list is empty.
    """
    total = 0
    count = 0
    for number in numbers:
        total += number
        count += 1
    # Check to avoid division by zero if the list is empty.
    if count == 0:
        return 0
    return total / count

def run_hidden_states(model, input_ids, images=None):
    outputs = model(input_ids, images=images, output_hidden_states=True)
    return outputs.hidden_states

##############################
# Hook function for scaling
##############################
def optimized_hook_fn_llama(module, input, output):
    global alpha
    # Stack the 256 learnable parameters into a tensor [256]
    params_tensor = torch.stack(alpha, dim=0)  # shape: [256]
    # Reshape to [1, 256, 1] for broadcasting with [1, 256, hidden_dim]
    params_tensor = params_tensor.view(1, 256, 1)
    return output * params_tensor.half()


def optimized_hook_fn_vicuna(module, input, output):
    global alpha
    # Stack the 256 learnable parameters into a tensor [256]
    params_tensor = torch.stack(alpha, dim=0)  # shape: [256]
    # Reshape to [1, 256, 1] for broadcasting with [1, 256, hidden_dim]
    params_tensor = params_tensor.view(1, 576, 1)
    return output * params_tensor.half()

##############################
# Generation function (unchanged)
##############################
def generate_response(tokenizer, model, image_processor, query, image, model_name, max_new_tokens=200):
    from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    qs = query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    from llava.conversation import conv_templates
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(DEVICE)
    input_ids = __import__("llava.mm_utils").mm_utils.tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    clear_gpu_memory()
    return output_ids

##############################
# Optimization of alpha
##############################
layer = 14
pos = -1
def optimize_alpha_for_input_llama(model_mm, harmful_toks_mm, harmful_images, target_direction, harmful_mean_act_text,
                             num_steps=100, lr=0.01, visualization=False):
    global alpha
    loss_history = []
    initial_loss = None
    # Initialize 256 learnable parameters for the image patches
    alpha = [torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, device=DEVICE)) for _ in range(256)]
    hs_mm_harmful = run_hidden_states(model_mm, harmful_toks_mm, images=harmful_images)
    harmful_mean_orig = hs_mm_harmful[layer+1][:, pos, :].mean(dim=0)
    optimizer = optim.Adam(alpha, lr=lr)
    for step in range(num_steps):
        optimizer.zero_grad()
        hs = run_hidden_states(model_mm, harmful_toks_mm, images=harmful_images)
        hidden = hs[layer+1][:, pos, :]
        hidden_mean = hidden.mean(dim=0)
        cos_sim = torch.dot(hidden_mean, target_direction)
        adv_loss = travel_scalar(target_direction, hidden_mean, harmful_mean_act_text)
        loss = adv_loss + 0.1 * travel_total_distance(hidden,harmful_mean_act_text)
        loss_value = adv_loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
        for param in alpha:
            param.data.clamp_(0.1, 1.0)
        if step % 5 == 0:
            print(f"Step {step}: avg_alpha={calculate_average(alpha).item():.3f}, loss={loss.item():.3f}")
        if visualization == False:
            if step == 0:
                initial_loss = loss_value
    
            loss_history.append(loss_value)
    
            # Stopping condition 1: less than 50% of initial loss
            if loss_value < 0.5 * initial_loss:
                print(f"Step {step}: loss {loss_value:.4f} < 50% of initial ({initial_loss:.4f}), stopping.")
                return alpha
    
            # Stopping condition 2: relative drop over last 5 steps < 1%
            if len(loss_history) > 5:
                old_loss = loss_history[-6]
                relative_drop = (old_loss - loss_value) / old_loss
                if relative_drop < 0.001:
                    print(f"Step {step}: relative drop {relative_drop*100:.2f}% over last 5 steps < 0.1%, stopping.")
                    return alpha
            
    return alpha

def optimize_alpha_for_input_vicuna(model_mm, harmful_toks_mm, harmful_images, target_direction, harmful_mean_act_text,
                             num_steps=100, lr=0.01, visualization=False):
    global alpha
    loss_history = []
    initial_loss = None
    # Initialize 256 learnable parameters for the image patches
    alpha = [torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, device=DEVICE)) for _ in range(256)]
    hs_mm_harmful = run_hidden_states(model_mm, harmful_toks_mm, images=harmful_images)
    harmful_mean_orig = hs_mm_harmful[layer+1][:, pos, :].mean(dim=0)
    optimizer = optim.Adam(alpha, lr=lr)
    for step in range(num_steps):
        optimizer.zero_grad()
        hs = run_hidden_states(model_mm, harmful_toks_mm, images=harmful_images)
        hidden = hs[layer+1][:, pos, :]
        hidden_mean = hidden.mean(dim=0)
        cos_sim = torch.dot(hidden_mean, target_direction)
        adv_loss = travel_scalar(target_direction, hidden_mean, harmful_mean_act_text)
        loss = adv_loss + 0.1 * travel_total_distance(hidden,harmful_mean_act_text)
        loss_value = adv_loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
        for param in alpha:
            param.data.clamp_(0.1, 1.0)
        if step % 5 == 0:
            print(f"Step {step}: avg_alpha={calculate_average(alpha).item():.3f}, loss={loss.item():.3f}")
        if visualization == False:
            if step == 0:
                initial_loss = loss_value
    
            loss_history.append(loss_value)
    
            # Stopping condition 1: less than 50% of initial loss
            if loss_value < 0.5 * initial_loss:
                print(f"Step {step}: loss {loss_value:.4f} < 50% of initial ({initial_loss:.4f}), stopping.")
                return alpha
    
            # Stopping condition 2: relative drop over last 5 steps < 1%
            if len(loss_history) > 5:
                old_loss = loss_history[-6]
                relative_drop = (old_loss - loss_value) / old_loss
                if relative_drop < 0.001:
                    print(f"Step {step}: relative drop {relative_drop*100:.2f}% over last 5 steps < 0.1%, stopping.")
                    return alpha
            
    return alpha