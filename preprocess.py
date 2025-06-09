from PIL import Image
from transformers import CLIPImageProcessor
from transformers import AutoTokenizer

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

import torch
import random
import json

clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_processor.do_center_crop = False
clip_processor.size = {"height": 224, "width": 224}  # Set the size to match the model input
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
tokenizer.pad_token = tokenizer.eos_token
anchor_xywh = torch.tensor([0.5, 0.5, 0.5, 0.5])  # fixed anchor

hico_text_prompts = json.load(open('prompts.json'))

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def xywh_to_xyxy(box):
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def get_union_box(box1, box2, size):
    x1 = min(box1[0], box2[0]) / size[0]
    y1 = min(box1[1], box2[1]) / size[1]
    x2 = max(box1[2], box2[2]) / size[0]
    y2 = max(box1[3], box2[3]) / size[1]
    return [x1, y1, x2, y2] 

def preprocess(sample, args=None):
    # random anchor version
    
    image, target = sample

    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    n_hois = len(target["hoi"])
    idx = random.randint(0, n_hois - 1)

    hoi = target["hoi"][idx]  
    box_h = target["boxes_h"][idx]  
    box_o = target["boxes_o"][idx]  
    if args.run_type in ['ablation_3', 'full']:
        context = target['human_attrib'][idx] + " "
        if args.run_type == 'full':
            context += target["context-llava"][idx]
    else:
        context = ""

    # union_xyxy = get_union_box(box_h, box_o, image.size)
    union_xyxy = get_union_box(box_h, box_h, image.size) # use human box as union box
    union_xywh = torch.tensor(xyxy_to_xywh(union_xyxy))

    # delta = union_xywh - anchor_xywh
    # random_t = random.uniform(0, 1)
    # new_anchor_xywh = anchor_xywh + delta * random_t
    # extra_label = (1 - random_t) * delta
    # # extra_label = union_xywh - 

    ## version : before 0608
    # anchor_xywh = torch.tensor([0.5, 0.5, 0.5, 0.5])
    # if random.random() < 0.2:
    #     rand_xys = torch.rand(2) * 0.5 + 0.25  # [0.25, 0.75]
    #     rand_whs = torch.rand(2) * 0.5 + 0.25  # [0.25, 0.75]
    #     anchor_xywh[0] = rand_xys[0]  # cx
    #     anchor_xywh[1] = rand_xys[1]  # cy
    #     anchor_xywh[2] = rand_whs[0]  # w
    #     anchor_xywh[3] = rand_whs[1]  # h
    # elif random.random() < 0.5:
    #     # anchor near the label box
    #     anchor_xywh[0] = union_xywh[0] + random.uniform(-0.1, 0.1) * union_xywh[2]  # cx
    #     anchor_xywh[1] = union_xywh[1] + random.uniform(-0.1, 0.1) * union_xywh[3]  # cy
    #     anchor_xywh[2] = union_xywh[2] * random.uniform(0.8, 1.2)  # w
    #     anchor_xywh[3] = union_xywh[3] * random.uniform(0.8, 1.2)  # h
    # else:
    #     pass

    # version : after 0608
    anchor_xywh = torch.tensor([0.5, 0.5, 1.0, 1.0]) # full image anchor
    if random.random() < 0.3:
        # anchor near the label box
        anchor_xywh[0] = union_xywh[0] + random.uniform(-0.1, 0.1) * union_xywh[2]  # cx
        anchor_xywh[1] = union_xywh[1] + random.uniform(-0.1, 0.1) * union_xywh[3]  # cy
        anchor_xywh[2] = union_xywh[2] * random.uniform(0.8, 1.2)  # w
        anchor_xywh[3] = union_xywh[3] * random.uniform(0.8, 1.2)  # h        
    else:
        ratio = random.random()
        # linear interpolation between full image anchor and union box anchor
        # interpolate in xyxy and then convert to xywh
        anchor_xyxy = xywh_to_xyxy(anchor_xywh)
        anchor_xyxy = [
            anchor_xyxy[0] * (1 - ratio) + union_xyxy[0] * ratio,
            anchor_xyxy[1] * (1 - ratio) + union_xyxy[1] * ratio,
            anchor_xyxy[2] * (1 - ratio) + union_xyxy[2] * ratio,
            anchor_xyxy[3] * (1 - ratio) + union_xyxy[3] * ratio
        ]
        anchor_xywh = torch.tensor(xyxy_to_xywh(anchor_xyxy))

    interaction_str = f"Locate a person in {hico_text_prompts[hoi.item()]}"
    output_str = "<answer>" # TODO : modify?
    if context != "":
        interaction_str += f". To be specific, {context.lower()}"
    # breakpoint()
    inputs = tokenizer(interaction_str, return_tensors="pt", padding="longest", truncation=True)
    labels = tokenizer(output_str, return_tensors="pt", padding="longest", truncation=True)["input_ids"].squeeze(0)
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "pixel_values": pixel_values,                      # [3, 224, 224]
        "input_ids": inputs["input_ids"].squeeze(0),       # [L]
        "current_anchor" : anchor_xywh,
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "label_texts": labels,                                  # [L]
        "label_coords": union_xywh                        # [4]
    }


def vlm_data_collator(features):
    return {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "input_ids": pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0),
        "label_texts": pad_sequence([f["label_texts"] for f in features], batch_first=True, padding_value=-100),
        "label_coords": torch.stack([f["label_coords"] for f in features]),  # shape: [B, 4],
        "current_anchor" : torch.stack([f['current_anchor'] for f in features])
    }