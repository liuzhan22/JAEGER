# ./models/SALMONN3D.py

import logging
import re
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Union
from transformers.utils import ModelOutput
from dataclasses import dataclass
from peft import LoraConfig, TaskType, get_peft_model
from .qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from utils.logger import print_trainable_parameters, is_main_process

@dataclass
class CausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class SALMONN3D(nn.Module):
    def __init__(
        self,
        model_path="",
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        low_resource=False,
        device_8bit=0,
        trainable_modules=None,
        **kwargs # Ignore unused args
    ):
        super().__init__()
        self.trainable_modules = trainable_modules if trainable_modules is not None else []
        # 1. Load Processor
        logging.info(f'Loading Qwen2.5-Omni Processor from {model_path}')
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True, use_safetensors=True)

        # 2. Load Model
        logging.info(f'Loading Qwen2.5-Omni Thinker Model from {model_path}')
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "use_safetensors": True,
            "attn_implementation": "flash_attention_2",
            "ignore_mismatched_sizes": True, # because of concating foa embeddings, audio_tower.ln_post, audio_tower.proj size will be different
        }
        
        if low_resource:
            load_kwargs.update({"load_in_8bit": True, "device_map": {"": device_8bit}})
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            load_kwargs["device_map"] = {"": local_rank}
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

        # 3. Freeze & Unfreeze Logic
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # 4. Setup LoRA
        if lora:
            target_modules_regex = r"model\.layers\..*?(" + "|".join(lora_target_modules) + r")$"
            logging.info(f"LoRA Target Regex: {target_modules_regex}")
            target_llm_decoder = getattr(self.model, "model", None)
            if target_llm_decoder is not None:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules_regex,
                )
                self.model = get_peft_model(self.model, peft_config)
                if is_main_process():
                    logging.info("--- LoRA Status ---")
                    self.model.print_trainable_parameters()
            else:
                logging.warning("LoRA requested but target module not found.")

        if self.trainable_modules:
            logging.info(f"Unfreezing modules matching: {self.trainable_modules}")
            for name, param in self.model.named_parameters():
                if any(keyword in name for keyword in self.trainable_modules):
                    param.requires_grad = True

        print_trainable_parameters(self.model)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        raw_wav: Optional[torch.FloatTensor] = None,
        raw_wav_lens: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Standardized HF forward signature.
        DataCollator has already unpacked samples into these arguments.
        """
        
        # Pack arguments into the dictionary structure expected by Qwen2.5-Omni
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if labels is not None:
            inputs["labels"] = labels
        if input_features is not None:
            inputs["input_features"] = input_features
        if feature_attention_mask is not None:
            inputs["feature_attention_mask"] = feature_attention_mask
        if raw_wav is not None:
            inputs["raw_wav"] = raw_wav
            inputs["raw_wav_lens"] = raw_wav_lens
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            inputs["image_grid_thw"] = image_grid_thw
        if video_grid_thw is not None:
            inputs["video_grid_thw"] = video_grid_thw
        if point_clouds is not None:
            inputs["point_clouds"] = point_clouds

        # Forward
        outputs = self.model(**inputs)

        loss = outputs.loss

        # fix for mean loss
        if loss is not None and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                loss = loss / world_size
                
        logits = outputs.logits

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
    
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        Wrapper specifically to handle Qwen2.5's multiple EOS tokens logic automatically.
        """
        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = [151643, 151645] # <|endoftext|> 151643, <|im_end|> 151645
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = 151643 # <|endoftext|>
        return self.model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Forward the call to the underlying model to support gradient checkpointing (memory saving).
        """
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    @classmethod
    def from_config(cls, config):
        # Simplified factory method
        model_path = config.get("model_path") or config.get("qwen_path", "")
        
        return cls(
            model_path=model_path,
            lora=config.get("lora", True),
            lora_rank=config.get("lora_rank", 64),
            lora_alpha=config.get("lora_alpha", 128),
            lora_dropout=config.get("lora_dropout", 0.05),
            low_resource=config.get("low_resource", False),
            device_8bit=config.get("device_8bit", 0),
        )