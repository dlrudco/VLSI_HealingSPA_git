import os
import torch
import torchvision
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from preprocess import preprocess, vlm_data_collator, tokenizer
from clip_t5_modelling import CLIPT5Config, CLIPT5Model
from dataset import HICODet, HICODetVLMDataset

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
    
def main():
    config = CLIPT5Config(
        vision_model_name="openai/clip-vit-large-patch14",
        language_model_name="google/flan-t5-large"
    )
    model = CLIPT5Model(config, tokenizer, apply_lora=True)

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_hicodet = HICODet(
                root=os.path.join('hicodet', "hico_20160224_det/images", 'train2015'),
                anno_file=os.path.join('hicodet', f"instances_train2015_curated.json"),
                target_transform=ToTensor(input_format='dict')
            )
    dataset = HICODetVLMDataset(train_hicodet, preprocess)

    training_args = TrainingArguments(
        output_dir="./checkpoints/clip-flant5_curated_tanh_nocrop_logloss",
        per_device_train_batch_size=32,
        dataloader_num_workers=8,
        gradient_accumulation_steps=1,
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
        run_name="clip-flant5_curated_tanh_nocrop_logloss",
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
    trainer.save_model("./checkpoints/clip-flant5_curated_tanh_nocrop_logloss")

if __name__ == "__main__":
    main()
