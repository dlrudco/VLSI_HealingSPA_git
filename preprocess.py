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

def get_union_box(box1, box2, size):
    x1 = min(box1[0], box2[0]) / size[0]
    y1 = min(box1[1], box2[1]) / size[1]
    x2 = max(box1[2], box2[2]) / size[0]
    y2 = max(box1[3], box2[3]) / size[1]
    return [x1, y1, x2, y2] 

def preprocess(sample):
    # random anchor version
    anchor_xywh = torch.tensor([0.5, 0.5, 0.5, 0.5])
    if random.random() < 0.2:
        rand_xys = torch.rand(2) * 0.5 + 0.25  # [0.25, 0.75]
        rand_whs = torch.rand(2) * 0.5 + 0.25  # [0.25, 0.75]
        anchor_xywh[0] = rand_xys[0]  # cx
        anchor_xywh[1] = rand_xys[1]  # cy
        anchor_xywh[2] = rand_whs[0]  # w
        anchor_xywh[3] = rand_whs[1]  # h
    image, target = sample

    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    n_hois = len(target["hoi"])
    idx = random.randint(0, n_hois - 1)

    hoi = target["hoi"][idx]  
    box_h = target["boxes_h"][idx]  
    box_o = target["boxes_o"][idx]  

    union_xyxy = get_union_box(box_h, box_o, image.size)
    union_xywh = torch.tensor(xyxy_to_xywh(union_xyxy))

    delta = union_xywh - anchor_xywh
    random_t = random.uniform(0, 1)
    new_anchor_xywh = anchor_xywh + delta * random_t
    extra_label = (1 - random_t) * delta
    # extra_label = union_xywh - anchor_xywh

    interaction_str = f"Locate {hico_text_prompts[hoi.item()]}, given the current anchor box :"
    output_str = "<answer>" # TODO : modify?

    inputs = tokenizer(interaction_str, return_tensors="pt", padding="longest", truncation=True)
    labels = tokenizer(output_str, return_tensors="pt", padding="longest", truncation=True)["input_ids"].squeeze(0)
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "pixel_values": pixel_values,                      # [3, 224, 224]
        "input_ids": inputs["input_ids"].squeeze(0),       # [L]
        "current_anchor" : new_anchor_xywh,
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "label_texts": labels,                                  # [L]
        "label_coords": extra_label                        # [4]
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