import os
import json
import torch
from clip_t5_modelling import CLIPT5Config, CLIPT5Model
from safetensors.torch import load_file
from transformers import CLIPImageProcessor, AutoTokenizer
from torchvision.ops import box_iou
from dataset import HICODet
from preprocess import tokenizer
from tqdm import tqdm

def offset_to_box(offset, anchor_xywh=[0.5, 0.5, 0.4, 0.4], img_size=(224, 224)):
    cx, cy, w, h = torch.tensor(anchor_xywh) + offset
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    W, H = img_size
    return torch.tensor([x1 * W, y1 * H, x2 * W, y2 * H])

def get_gt_union_box(box_h, box_o):
    x1 = min(box_h[0], box_o[0])
    y1 = min(box_h[1], box_o[1])
    x2 = max(box_h[2], box_o[2])
    y2 = max(box_h[3], box_o[3])
    return torch.tensor([x1, y1, x2, y2])

@torch.no_grad()
def evaluate_clip_t5(args, model, dataset, hico_prompts, tokenizer, clip_processor, device="cuda", iou_threshold=0.5):
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    pbar = tqdm(range(len(dataset)), desc="Evaluating")
    for idx in pbar:
        image, target = dataset.__getitem__(idx)
        image_size = image.size
        n_hois = len(target["hoi"])
        if n_hois == 0:
            continue

        for hidx in tqdm(range(n_hois), leave=False, desc=f"Processing image {idx}"):
            hoi_idx = target["hoi"][hidx]
            text_prompt = f"Locate a person in {hico_prompts[hoi_idx]}"
            if args.run_type in ['ablation_3', 'full']:
                try:
                    context = target['human_attrib'][hidx] + " "
                except IndexError:
                    breakpoint()
                if args.run_type == 'full':
                    context += target["context-llava"][hidx]
            else:
                context = ""
            if context != "":
                text_prompt += f". To be specific, {context.lower()}"

            if args.run_type in ['baseline', 'ablation_1']:
                anchor = torch.zeros(4)
            else:   
                anchor = torch.tensor([0.5, 0.5, 0.5, 0.5])  # fixed anchor

            offset_acc = torch.zeros(4)
            anchor_now = anchor.clone()

            for _ in range(1):
                pixel_values = clip_processor(image, return_tensors="pt")["pixel_values"].to(device)
                text_inputs = tokenizer(text_prompt, return_tensors="pt", padding="longest", truncation=True, max_length=128)
                input_ids = text_inputs["input_ids"].to(device)
                attention_mask = text_inputs["attention_mask"].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    current_anchor=anchor_now.unsqueeze(0).to(device),
                    compute_additional_loss=True
                )
                offset = outputs["regression_output"].squeeze(0).cpu()
                offset_acc += offset
                anchor_now += offset
            pred_box = offset_to_box(offset_acc, anchor, image_size).unsqueeze(0)
            gt_box = get_gt_union_box(target["boxes_h"][hidx], target["boxes_h"][hidx]).unsqueeze(0)

            # MSE Loss
            mse = torch.nn.functional.l1_loss(pred_box, gt_box).item()
            total_loss += mse

            # IoU Accuracy
            iou = box_iou(pred_box, gt_box)[0, 0].item()
            total_correct += int(iou > iou_threshold)
            total_samples += 1
            pbar.set_postfix({
                "avg_l1": total_loss/total_samples if total_samples > 0 else 0,
                "correct": total_correct / total_samples if total_samples > 0 else 0,
                "total": total_samples
            })

    avg_mse = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_mse, accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for CLIP-T5 model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--run_type", type=str, default="train", choices=["baseline", "ablation_1", "ablation_2", "ablation_3", "full"], help="Type of run to perform")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hico_prompts = json.load(open("prompts.json"))
    
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    tokenizer.pad_token = tokenizer.eos_token

    config = CLIPT5Config(
        vision_model_name="openai/clip-vit-large-patch14",
        language_model_name="google/flan-t5-large"
    )
    model = CLIPT5Model(config, tokenizer, apply_lora=True, args=args)
    checkpoint = load_file(args.checkpoint, device="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    test_dataset = HICODet(
        root="hicodet/hico_20160224_det/images/test2015",
        anno_file="hicodet/instances_test2015_merged.json"
    )

    avg_mse, accuracy = evaluate_clip_t5(args, model, test_dataset, hico_prompts, tokenizer, clip_processor, device=device)
    print(f"[Test] MSE: {avg_mse:.4f}, Accuracy (IoU > 0.5): {accuracy:.4f}")
    with open(f'results_{args.run_type}.txt', 'w') as f:
        f.write(f"[Test] MSE: {avg_mse:.4f}, Accuracy (IoU > 0.5): {accuracy:.4f}\n")
        f.write(f"Run type: {args.run_type}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")