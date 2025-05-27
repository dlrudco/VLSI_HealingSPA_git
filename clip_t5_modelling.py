import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, T5ForConditionalGeneration, CLIPVisionModel
from transformers import PretrainedConfig, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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


class CLIPT5Model(PreTrainedModel):
    config_class = CLIPT5Config  # 별도로 정의한 config class

    def __init__(self, config, apply_lora=False):
        super().__init__(config)
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.vision_model_name)
        self.vision_encoder.requires_grad_(False)
        vision_dim = self.vision_encoder.config.hidden_size  # 1024 for CLIP-L/14
        if apply_lora:
            self.language_model = load_lora_t5_model(config.language_model_name)
        else:
            self.language_model = T5ForConditionalGeneration.from_pretrained(config.language_model_name, attn_implementation="eager",config={"tensor_parallel": 1},)
        t5_d_model = self.language_model.config.d_model

        self.image_proj = nn.Linear(vision_dim, t5_d_model)
        self.anchor_proj = nn.Linear(4, t5_d_model)

        self.regression_head = nn.Sequential(nn.Linear(t5_d_model, 1), nn.Tanh())  # Regression head to predict offsets

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
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_embeds = vision_outputs.last_hidden_state  # [B, 257, 1024]
        projected_image_embeds = self.image_proj(vision_embeds)  # [B, 257, d_model]
        image_attention_mask = torch.ones(projected_image_embeds.shape[:2], dtype=torch.long).to(projected_image_embeds.device)
        
        projected_anchor_embeds = self.anchor_proj(current_anchor).unsqueeze(1)  # [B, 1, d_model]
        anchor_attention_mask = torch.ones(projected_anchor_embeds.shape[:2], dtype=torch.long).to(projected_anchor_embeds.device)

        text_embeds = self.language_model.encoder.embed_tokens(input_ids)  # [B, T, d_model]

        fused_inputs = torch.cat([projected_image_embeds, text_embeds, projected_anchor_embeds], dim=1) # [B, 257 + T + 1, d_model]
        fused_attention_mask = torch.cat([image_attention_mask, attention_mask, anchor_attention_mask], dim=1) # [B, 257 + T + 1]

        encoder_outputs = self.language_model.encoder(
            inputs_embeds=fused_inputs,
            attention_mask=fused_attention_mask,
            return_dict=return_dict,
        )

        B = fused_inputs.size(0)
        K = label_coords.shape[-1] if label_coords is not None else 4
        
        start_token_id = self.language_model.config.decoder_start_token_id or self.language_model.config.pad_token_id
        decoder_input_ids = torch.full((B, K), start_token_id, dtype=torch.long).to(fused_inputs.device)
        decoder_inputs_embeds = self.language_model.decoder.embed_tokens(decoder_input_ids)

        # Step 6. Decoder
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
        ce_loss = None # TODO : perhaps add a cross-entropy loss for the text labels later
        total_loss = None
        pred_offsets = self.regression_head(sequence_output).squeeze(-1)  # [B, K]
        if label_coords is not None:
            regression_loss = F.mse_loss(pred_offsets, label_coords, reduction='sum') / B  # Mean Squared Error Loss

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
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
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
