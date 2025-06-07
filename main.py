import os
import torch
import torchvision
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from preprocess import preprocess, vlm_data_collator, tokenizer
from clip_t5_modelling import CLIPT5Config, CLIPT5Model
from dataset import HICODet, HICODetVLMDataset
from safetensors.torch import load_file

from functools import partial

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
    
def main(args):
    config = CLIPT5Config(
        vision_model_name="openai/clip-vit-large-patch14",
        language_model_name="google/flan-t5-large"
    )
    model = CLIPT5Model(config, tokenizer, apply_lora=True, args=args)
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        state_dict = load_file(args.checkpoint, device="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No checkpoint provided, initializing model from scratch.")
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_hicodet = HICODet(
                root=os.path.join('hicodet', "hico_20160224_det/images", 'train2015'),
                anno_file=os.path.join('hicodet', f"instances_train2015_merged.json"),
                target_transform=ToTensor(input_format='dict')
            )
    _preprocess = partial(preprocess, args=args)
    dataset = HICODetVLMDataset(train_hicodet, _preprocess)

    training_args = TrainingArguments(
        output_dir=f"./checkpoints/{args.exp_name}",
        per_device_train_batch_size=16,
        dataloader_num_workers=8,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=30,
        weight_decay=5e-4,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=1000,
        save_total_limit=5,
        remove_unused_columns=False,
        fp16=False,
        report_to="wandb",
        run_name=args.exp_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=vlm_data_collator,
        tokenizer=None  # generation은 안 하므로 tokenizer 없이 가능
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(f"./checkpoints/{args.exp_name}")

def make_random_exp_name():
    import random
    import string
    random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"exp_{random_name}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CLIP-T5 model on HICO-DET dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name for logging")
    parser.add_argument("--run_type", type=str, default="train", choices=["baseline", "ablation_1", "ablation_2", "ablation_3", "full"], help="Type of run to perform")
    args = parser.parse_args()
    if args.exp_name == "":
        args.exp_name = make_random_exp_name()
    main(args)
