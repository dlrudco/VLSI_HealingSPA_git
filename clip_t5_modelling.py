import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, T5ForConditionalGeneration, CLIPVisionModel
from transformers import PretrainedConfig, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def xywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(dim=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

class CLIPT5Config(PretrainedConfig):
    model_type = "clip_t5"

    def __init__(self, vision_model_name="openai/clip-vit-large-patch14",
                 language_model_name="google/flan-t5-large",
                 image_proj_dim=1024,
                 **kwargs):
        super().__init__(**kwargs)
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.image_proj_dim = image_proj_dim

class LogAbsLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', beta=3.0):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.beta = beta

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        log_term = torch.log1p(self.beta*error + self.eps)  # log1p(x) = log(1 + x)
        if self.reduction == 'mean':
            return torch.mean(log_term)
        elif self.reduction == 'sum':
            return torch.sum(log_term)
        else:
            return log_term

import random

class CLIPT5Model(PreTrainedModel):
    config_class = CLIPT5Config  # 별도로 정의한 config class

    def __init__(self, config, tokenizer, apply_lora=False, iteration=3, args=None):
        super().__init__(config)
        self.iteration = iteration
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.vision_model_name)
        self.vision_encoder.requires_grad_(False)
        self.tokenizer = tokenizer
        self.special_tokens = None
        self.args = args
        vision_dim = self.vision_encoder.config.hidden_size  # 1024 for CLIP-L/14
        if apply_lora:
            self.language_model = load_lora_t5_model(config.language_model_name)
        else:
            self.language_model = T5ForConditionalGeneration.from_pretrained(config.language_model_name, attn_implementation="eager",config={"tensor_parallel": 1},)
        t5_d_model = self.language_model.config.d_model

        self.image_proj = nn.Linear(vision_dim, t5_d_model)
        # self.anchor_proj = nn.Linear(4, t5_d_model)
        self.anchor_proj = nn.Linear(t5_d_model, t5_d_model)
        self.anchor_position_embeds = self.get_sinusoidal_positional_embedding(224, t5_d_model) # pixel level indicator.

        # self.offset_token_embeddings = nn.Embedding(4, t5_d_model)  # For the anchor token
        if self.args.run_type in ['baseline', 'ablation_1']:
            self.regression_head = nn.Sequential(nn.Linear(t5_d_model, 1), nn.Sigmoid())  # Regression head for predicting offsets
        else:
            self.regression_head = nn.Sequential(nn.Linear(t5_d_model, 1), nn.Tanh())  # Regression head for predicting offsets
        # self.loss_func = LogAbsLoss(reduction='sum')  # Logarithmic absolute loss for regression
        self.loss_func = nn.L1Loss(reduction='sum')

        self.image_positional_embedding = self.get_sinusoidal_positional_embedding(257, t5_d_model)  # 256 patches + 1 CLS token
        self.position_proj = nn.Linear(t5_d_model, t5_d_model)

    def get_sinusoidal_positional_embedding(self, l: int, d: int) -> torch.Tensor:
        pe = torch.zeros(l, d)
        position = torch.arange(0, l).unsqueeze(1)  # shape: (l, 1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))  # shape: (d//2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        return pe
    
    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        label_texts=None,
        label_coords=None,               # [B, K]
        compute_additional_loss=False,
        return_dict=True,
        current_anchor=None,
    ):
        text_embeds = self.language_model.encoder.embed_tokens(input_ids)  # [B, T, d_model]
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            vision_embeds = vision_outputs.last_hidden_state  # [B, 257, 1024]
            if self.args.run_type == 'baseline':
                vision_embeds = vision_embeds[:, 0, :].unsqueeze(1)  # only use the CLS token
        projected_image_embeds = self.image_proj(vision_embeds)  # [B, 257, d_model]
        projected_image_embeds = projected_image_embeds + \
            self.position_proj(self.image_positional_embedding.unsqueeze(0).to(projected_image_embeds.device))  # [B, 257, d_model]
        image_attention_mask = torch.ones(projected_image_embeds.shape[:2], dtype=torch.long).to(projected_image_embeds.device) 
        
        ce_loss = None # TODO : perhaps add a cross-entropy loss for the text labels later
        total_loss = 0.
        if self.args.run_type not in ['baseline', 'ablation_1']:
            patch_anchor = current_anchor
            # cur_iteration = self.iteration if self.training else 1
            # for _ in range(cur_iteration):
            self.anchor_position_embeds = self.anchor_position_embeds.to(projected_image_embeds.device)
            patch_anchor = (patch_anchor * 223).round().long().clamp(0, 223).to(projected_image_embeds.device)  # Convert to pixel coordinates in the range [0, 223], (B, 4)
            patch_anchor = self.anchor_position_embeds[patch_anchor].to(projected_image_embeds.device)  # [B, 4, d_model]
            projected_anchor_embeds = self.anchor_proj(patch_anchor)  # [B, 4, d_model]
            anchor_attention_mask = torch.ones(projected_anchor_embeds.shape[:2], dtype=torch.long).to(projected_anchor_embeds.device)
            
            with torch.no_grad():
                visual_patch = []
                xy_patch = xywh_to_xyxy(current_anchor)  # Convert from xywh to xyxy
                for bidx in range(projected_image_embeds.size(0)):
                    patch_x1 = int(max(0., xy_patch[bidx, 0] * pixel_values.size(3)))
                    patch_x2 = int(min(pixel_values.size(3), max(xy_patch[bidx, 2] * pixel_values.size(3),patch_x1+1)))
                    patch_y1 = int(max(0., xy_patch[bidx, 1] * pixel_values.size(2)))
                    patch_y2 = int(min(pixel_values.size(2), max(xy_patch[bidx, 3] * pixel_values.size(2),patch_y1+1)))
                    patch_chunk = pixel_values[bidx, :,
                        patch_y1:patch_y2, patch_x1:patch_x2]
                    resized_patch = F.interpolate(
                        patch_chunk.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                    )
                    visual_patch.append(resized_patch.squeeze(0))

                visual_patch = torch.stack(visual_patch, dim=0)  # [B, 3, H, W]
                patch_outputs = self.vision_encoder(pixel_values=visual_patch)
                patch_embeds = patch_outputs.last_hidden_state
                patch_cls = patch_embeds[:, 0, :].unsqueeze(1)  # [B, 1, d_model]
            projected_patch_embeds = self.image_proj(patch_cls)  # [B, 1, d_model]
            patch_attention_mask = torch.ones(projected_patch_embeds.shape[:2], dtype=torch.long).to(projected_patch_embeds.device)    

            fused_inputs = torch.cat([projected_image_embeds, text_embeds, projected_patch_embeds, projected_anchor_embeds], dim=1) # [B, 257 + T + 1 + 1, d_model]
            fused_attention_mask = torch.cat([image_attention_mask, attention_mask, patch_attention_mask, anchor_attention_mask], dim=1) # [B, 257 + T + 1 + 1]
        else:
            fused_inputs = torch.cat([projected_image_embeds, text_embeds], dim=1)
            fused_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        encoder_outputs = self.language_model.encoder(
            inputs_embeds=fused_inputs,
            attention_mask=fused_attention_mask,
            return_dict=return_dict,
        )

        B = fused_inputs.size(0)
        K = label_coords.shape[-1] if label_coords is not None else 4
        
        # start_token_id = self.language_model.config.decoder_start_token_id or self.language_model.config.pad_token_id
        # decoder_input_ids = torch.full((B, K), start_token_id, dtype=torch.long).to(fused_inputs.device)
        # decoder_inputs_embeds = self.language_model.decoder.embed_tokens(decoder_input_ids)
        # decoder_inputs_embeds = self.offset_token_embeddings(torch.arange(K, device=fused_inputs.device)).unsqueeze(0).repeat(B,1,1)  # [K, d_model]
        # Step 6. Decoder
        special_tokens = ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<extra_id_3>"]
        token_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)
        decoder_input_ids = torch.tensor(token_ids).unsqueeze(0).repeat(B, 1).to(fused_inputs.device)  # [B, K]
        decoder_inputs_embeds = self.language_model.decoder.embed_tokens(decoder_input_ids)
        decoder_outputs = self.language_model.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=fused_attention_mask,
            return_dict=return_dict,
        )
        
        sequence_output = decoder_outputs.last_hidden_state

        # lm_logits = self.language_model.lm_head(decoder_outputs.last_hidden_state)  # [B, T, vocab_size] # for now, we just use the sequence output directly
        # TODO : when we also add text target, we can use the lm_head to get the logits

        # if label_texts is not None:
        #     ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        #     ce_loss = ce_loss_fn(lm_logits.view(-1, lm_logits.size(-1)), label_texts.view(-1))
        # else:
        #     ce_loss = None

        pred_offsets = self.regression_head(sequence_output).squeeze(-1)  # [B, K]
        if label_coords is not None:
            # regression_loss = F.l1_loss(pred_offsets, label_coords, reduction='sum') / B  # Mean Squared Error Loss
            # breakpoint()
            if self.args.run_type in ['baseline', 'ablation_1']:
                # predict the xywh coordinates directly
                xy_label_coords = xywh_to_xyxy(label_coords)  # Convert from xywh to xyxy
                xy_pred = xywh_to_xyxy(pred_offsets)  # Convert from xywh to xyxy
                regression_loss = self.loss_func(xy_pred, xy_label_coords) / B
            else:
                # predict the offsets from the current anchor
                xy_label_coords = xywh_to_xyxy(label_coords)  # Convert from xywh to xyxy
                xy_pred = xywh_to_xyxy(pred_offsets+current_anchor)  # Convert from xywh to xyxy
                regression_loss = self.loss_func(xy_pred, xy_label_coords) / B  # Logarithmic absolute loss
            if ce_loss is not None:
                total_loss = ce_loss + regression_loss
            else:
                total_loss = regression_loss
        else:
            total_loss = ce_loss
            

        return {
            "loss": total_loss,
            "regression_output": pred_offsets,
        }



def load_lora_t5_model(model_name: str):
    # 1. Quantization 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32
    )

    # 2. Base model 로드
    base_model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )

    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=[
            "q", "k", "v",               # Attention heads
            "o",                         # Output projection
            "wi", "wo",                  # Feed-forward
        ]
    )

    lora_model = get_peft_model(base_model, lora_config)

    print("LoRA applied:")
    train_p, tot_p = lora_model.get_nb_trainable_parameters()
    print(f'Trainable Params :      {train_p/1e6:.2f}M / {tot_p/1e6:.2f}M')
    print(f'Percentage :         {100*train_p/tot_p:.2f}%')

    return lora_model
