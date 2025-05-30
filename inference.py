import os
import json
import torch
import torchvision
from clip_t5_modelling import CLIPT5Config, CLIPT5Model
from safetensors.torch import load_file
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import CLIPImageProcessor
from transformers import AutoTokenizer
from dataset import HICODet, HICODetVLMDataset
from preprocess import preprocess, tokenizer

hico_text_prompts = json.load(open('prompts.json'))

def _to_list_of_tensor(x, dtype=None, device=None):
    return [torch.as_tensor(item, dtype=dtype, device=device) for item in x]

def _to_tuple_of_tensor(x, dtype=None, device=None):
    return tuple(torch.as_tensor(item, dtype=dtype, device=device) for item in x)

def _to_dict_of_tensor(x, dtype=None, device=None):
    return dict([(k, torch.as_tensor(v, dtype=dtype, device=device)) for k, v in x.items()])

def to_tensor(x, input_format='tensor', dtype=None, device=None):
    """Convert input data to tensor based on its format"""
    if input_format == 'tensor':
        return torch.as_tensor(x, dtype=dtype, device=device)
    elif input_format == 'pil':
        return torchvision.transforms.functional.to_tensor(x).to(
            dtype=dtype, device=device)
    elif input_format == 'list':
        return _to_list_of_tensor(x, dtype=dtype, device=device)
    elif input_format == 'tuple':
        return _to_tuple_of_tensor(x, dtype=dtype, device=device)
    elif input_format == 'dict':
        return _to_dict_of_tensor(x, dtype=dtype, device=device)
    else:
        raise ValueError("Unsupported format {}".format(input_format))
    
class ToTensor:
    """Convert to tensor"""
    def __init__(self, input_format='tensor', dtype=None, device=None):
        self.input_format = input_format
        self.dtype = dtype
        self.device = device
    def __call__(self, x):
        return to_tensor(x, 
            input_format=self.input_format,
            dtype=self.dtype,
            device=self.device
        )
    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'input_format={}'.format(repr(self.input_format))
        reprstr += ', dtype='
        reprstr += repr(self.dtype)
        reprstr += ', device='
        reprstr += repr(self.device)
        reprstr += ')'
        return reprstr
    
clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
tokenizer.pad_token = tokenizer.eos_token

@torch.no_grad()
def generate_offsets(model, image, text, anchor_xywh, tokenizer, clip_processor, device="cuda"):
    model.eval()
    model.to(device)

    # 1. 이미지 전처리
    pixel_values = clip_processor(image, return_tensors="pt")["pixel_values"].to(device)  # [1, 3, 224, 224]

    # 2. 텍스트 전처리
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=128
    )
    input_ids = inputs["input_ids"].to(device)              # [1, T]
    attention_mask = inputs["attention_mask"].to(device)    # [1, T]

    # 3. anchor box 준비 (normalized xywh)
    current_anchor = torch.tensor(anchor_xywh, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 4]

    # 4. 모델에 넣고 회귀값 추출
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        current_anchor=current_anchor,
        compute_additional_loss=True 
    )

    # 5. 회귀값 반환
    return outputs["regression_output"].squeeze(0).cpu()  # [K]



def offset_to_box(offset, anchor_xywh=[0.5, 0.5, 0.4, 0.4], img_size=(224, 224)):
    # anchor + offset = predicted union box (normalized xywh)
    anchor = torch.tensor(anchor_xywh)
    pred = anchor + offset

    # convert to xyxy (normalized)
    cx, cy, w, h = pred
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # scale to image size
    W, H = img_size
    return [int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)]


def get_gt_union_box(box_h, box_o):
    # box_h, box_o: xyxy in normalized coordinates
    x1 = min(box_h[0], box_o[0])
    y1 = min(box_h[1], box_o[1])
    x2 = max(box_h[2], box_o[2])
    y2 = max(box_h[3], box_o[3])
    
    return [int(x1), int(y1), int(x2), int(y2)]

def visualize_boxes_with_gt(image, anchor_box, pred_box, gt_box, title=""):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # anchor box (red)
    ax.add_patch(patches.Rectangle(
        (anchor_box[0], anchor_box[1]),
        anchor_box[2] - anchor_box[0],
        anchor_box[3] - anchor_box[1],
        linewidth=2, edgecolor='r', facecolor='none', label='Anchor'
    ))

    # predicted box (green)
    ax.add_patch(patches.Rectangle(
        (pred_box[0], pred_box[1]),
        pred_box[2] - pred_box[0],
        pred_box[3] - pred_box[1],
        linewidth=2, edgecolor='g', facecolor='none', label='Predicted'
    ))

    # GT union box (blue)
    ax.add_patch(patches.Rectangle(
        (gt_box[0], gt_box[1]),
        gt_box[2] - gt_box[0],
        gt_box[3] - gt_box[1],
        linewidth=2, edgecolor='b', facecolor='none', label='GT Union'
    ))

    ax.legend()
    plt.title(title)
    plt.axis('off')
    plt.savefig("visualization.png", bbox_inches='tight')


def main(args):
    config = CLIPT5Config(
        vision_model_name="openai/clip-vit-large-patch14",
        language_model_name="google/flan-t5-large"
    )
    model = CLIPT5Model(config, tokenizer, apply_lora=True)
    
    checkpoint = load_file(args.checkpoint, device="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    test_hicodet = HICODet(
                root=os.path.join('hicodet', "hico_20160224_det/images", 'train2015'),
                anno_file=os.path.join('hicodet', f"instances_train2015_curated.json"),
                target_transform=ToTensor(input_format='dict')
            )
    for iidx in range(len(test_hicodet)):
        image, target = test_hicodet.__getitem__(iidx)
        n_hois = len(target["hoi"])
        # for idx in range(n_hois):
        idx = 0

        hoi = target["hoi"][idx]  
        box_h = target["boxes_h"][idx]  
        box_o = target["boxes_o"][idx]  
        interaction_str = f"Locate {hico_text_prompts[hoi.item()]}, given the current anchor box : ?"
        anchor_xywh = torch.tensor([0.5, 0.5, 0.5, 0.5])
        
        with torch.no_grad():
            # cum_offset = torch.zeros(4, dtype=torch.float32)
            # new_anchor_xywh = anchor_xywh.clone()
            # for i in range(10):
            offset = generate_offsets(
                model=model,
                image=image,
                text=interaction_str,
                anchor_xywh=anchor_xywh,
                tokenizer=tokenizer,
                clip_processor=clip_processor,
                device="cuda"
            )
                # new_anchor_xywh = new_anchor_xywh + offset / 10
                # cum_offset += offset / 10
                # print(f"Predicted offset: {offset.numpy()}")
        print(interaction_str)
        # offset = cum_offset
        img_size = image.size

        # --- 예측된 union box (green)
        pred_box = offset_to_box(offset, anchor_xywh, img_size)

        # --- anchor box (red)
        anchor_box = offset_to_box(torch.zeros(4), anchor_xywh, img_size)

        # --- GT union box 계산 (blue)
        # HICODet target dict에서 하나 예시로 꺼낸다고 가정

        gt_box = get_gt_union_box(box_h, box_o)

        # --- 시각화
        visualize_boxes_with_gt(image, anchor_box, pred_box, gt_box, title="Anchor vs Predicted vs GT")
        breakpoint()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for CLIP-T5 model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
    main(args)
