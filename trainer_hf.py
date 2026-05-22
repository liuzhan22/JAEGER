# ./trainer_hf.py

import argparse
import os
import logging
import sys
import torch
import json
import time
import random
import numpy as np
from transformers import Trainer, TrainingArguments
from transformers import set_seed

# Import your modules
from config import Config
from models.SALMONN3D import SALMONN3D
from dataset import SALMONN3DDataset
from data_collator import SALMONNDataCollator
from utils.logger import setup_logger, now, is_main_process
from safetensors.torch import save_file, load_file

def parse_args():
    parser = argparse.ArgumentParser(description='HF Trainer for SALMONN3D')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser.parse_args()

class SALMONNTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        """
        Overridden save_model to only save parameters with requires_grad=True.
        This significantly reduces checkpoint size for LoRA/PEFT training.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Unwrap the model (handle DDP wrapper i.e., model.module)
        model_to_save = self.model
        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module

        trainable_state_dict = {
            k: v.cpu() for k, v in model_to_save.named_parameters() if v.requires_grad
        }

        save_file(trainable_state_dict, os.path.join(output_dir, "model.safetensors"))

        if hasattr(model_to_save, "peft_config"):
            model_to_save.peft_config.save_pretrained(output_dir)
        elif hasattr(model_to_save, "config"):
            model_to_save.config.save_pretrained(output_dir)

        if hasattr(model_to_save, "processor"):
            model_to_save.processor.save_pretrained(output_dir)

        logging.info(f"[Custom Save] Checkpoint saved to {output_dir} (Trainable params only, {len(trainable_state_dict)} keys).")

def main():
    # 1. Load Configuration
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    timestamp = now()

    base_output_dir = run_config.get("output_dir", "./output")
    exp_dir = os.path.join(base_output_dir, timestamp)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    base_run_name = run_config.get("wandb_run_name", "salmonn3d_exp")
    full_run_name = f"{base_run_name}_{timestamp}"

    if "wandb_project" in run_config:
        os.environ["WANDB_PROJECT"] = run_config.wandb_project
    os.environ["WANDB_DIR"] = exp_dir
    # os.environ["WANDB_MODE"] = "offline"

    setup_logger(output_dir=exp_dir, log_filename=f"train_{timestamp}.log")

    logging.info(f"Experiment Directory: {exp_dir}")
    cfg.pretty_print()

    # 2. Set Seed
    set_seed(run_config.seed)

    # 3. Build Model
    model = SALMONN3D(
        model_path=model_config.qwen_path,
        lora=model_config.lora,
        lora_rank=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        low_resource=model_config.get("low_resource", False),
        device_8bit=model_config.get("device_8bit", 0),
        trainable_modules=model_config.get("trainable_modules", []),
    )
    
    # Optional: Load initial checkpoints for audio and visual towers if specified
    curr_sd = model.state_dict()
    audio_ckpt = model_config.get("audio_init_ckpt", "")
    visual_ckpt = model_config.get("visual_init_ckpt", "")

    if audio_ckpt and os.path.exists(audio_ckpt):
        logging.info(f"Loading initial audio tower weights from {audio_ckpt}")
        state_dict = {k: v for k, v in load_file(audio_ckpt).items() if k in curr_sd and "audio_tower" in k and v.shape == curr_sd[k].shape}
        if not state_dict:
            raise ValueError("No matching keys found for audio tower in the provided checkpoint.")
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded {len(state_dict)} keys into audio tower.")

    if visual_ckpt and os.path.exists(visual_ckpt):
        logging.info(f"Loading initial visual tower weights from {visual_ckpt}")
        state_dict = {k: v for k, v in load_file(visual_ckpt).items() if k in curr_sd and "visual" in k and v.shape == curr_sd[k].shape}
        if not state_dict:
            raise ValueError("No matching keys found for visual tower in the provided checkpoint.")
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded {len(state_dict)} keys into visual tower.")

    # 4. Load Multi-Prompt Dictionary
    prompt_dict = None
    if model_config.get("multi_prompt", False) and model_config.get("prompt_path", ""):
        logging.info(f"Loading multi training prompts from {model_config.prompt_path}")
        try:
            with open(model_config.prompt_path, "r", encoding="utf-8") as f:
                prompt_dict = json.load(f)
        except Exception as e:
            logging.error(f"Error loading multi training prompt file: {e}")

    # 5. Prepare Datasets
    tasks = data_config.get("tasks", None)
    enable_audio = data_config.get("enable_audio", True)
    enable_visual = data_config.get("enable_visual", True)
    logging.info(f"Modality Config - Audio: {enable_audio}, Visual: {enable_visual}")
    if tasks:
        logging.info(f"Filtering datasets by tasks: {tasks}")
    train_dataset = SALMONN3DDataset(data_config.train_ann_path, tasks=tasks, enable_audio=enable_audio, enable_visual=enable_visual)
    eval_dataset = SALMONN3DDataset(data_config.valid_ann_path, tasks=tasks, enable_audio=enable_audio, enable_visual=enable_visual) if data_config.get("valid_ann_path", None) else None
    
    # 6. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=exp_dir,
        overwrite_output_dir=True,
        
        # Training Strategy
        num_train_epochs=run_config.optims.max_epoch,
        per_device_train_batch_size=run_config.batch_size_train,
        per_device_eval_batch_size=run_config.batch_size_eval,
        gradient_accumulation_steps=run_config.accum_grad_iters,
        
        # Optimization
        learning_rate=run_config.optims.peak_lr,
        weight_decay=run_config.optims.weight_decay,
        warmup_steps=run_config.optims.warmup_steps,
        lr_scheduler_type="cosine",
        
        # Precision
        bf16=True if torch.cuda.is_bf16_supported() else False, 
        fp16=not torch.cuda.is_bf16_supported(),
        
        # Logging & Saving
        logging_dir=os.path.join(run_config.output_dir, "logs"),
        logging_steps=run_config.log_freq,
        report_to="wandb" if "wandb_project" in run_config else "none",
        run_name=full_run_name,
        save_strategy=run_config.get("save_strategy", "steps"),
        save_steps=run_config.get("save_steps", 50),
        eval_strategy=run_config.get("eval_strategy", "steps"),
        eval_steps=run_config.get("eval_steps", 50),
        save_total_limit=2,
        dataloader_drop_last=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        
        # DDP & Hardware
        dataloader_num_workers=run_config.num_workers,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False, 
    )

    # 7. Initialize Data Collator
    # Pass the loaded prompt_dict here
    data_collator = SALMONNDataCollator(
        processor=model.processor,
        model_config=model_config,
        prompt_dict=prompt_dict
    )

    # 8. Initialize Trainer
    trainer = SALMONNTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 9. Train
    if run_config.evaluate:
        logging.info("Starting Evaluation...")
        metrics = trainer.evaluate()
        logging.info(metrics)
    else:
        logging.info("Starting Training...")
        resume_ckpt = model_config.get("ckpt", None)
        load_weights_only = model_config.get("load_weights_only", False)

        if resume_ckpt and os.path.exists(resume_ckpt):
            if load_weights_only:
                logging.info(f"Loading model weights only from {resume_ckpt}")
                
                # Identify weight file
                if os.path.isdir(resume_ckpt):
                    weight_path = os.path.join(resume_ckpt, "model.safetensors")
                else:
                    weight_path = resume_ckpt

                # Load state dict
                if weight_path.endswith(".safetensors"):
                    state_dict = load_file(weight_path)
                else:
                    state_dict = torch.load(weight_path, map_location="cpu")
                
                ckpt_keys = list(state_dict.keys())
                model_keys = list(model.state_dict().keys())

                # Apply weights to model
                msg = model.load_state_dict(state_dict, strict=False)
                logging.info(f"Weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
                
                if msg.missing_keys:
                    logging.warning(f"Total Missing: {len(msg.missing_keys)}")
                    logging.warning(f"First 10 missing keys: {msg.missing_keys[:10]}")
                if msg.unexpected_keys:
                    logging.warning(f"Total Unexpected: {len(msg.unexpected_keys)}")
                    logging.warning(f"First 10 unexpected keys: {msg.unexpected_keys[:10]}")
                    
                # Start training from step 0
                trainer.train()
            else:
                # Full resume (weights + optimizer + scheduler)
                if os.path.isdir(resume_ckpt) and os.path.exists(os.path.join(resume_ckpt, "trainer_state.json")):
                    logging.info(f"Resuming full training state from {resume_ckpt}")
                    trainer.train(resume_from_checkpoint=resume_ckpt)
                else:
                    logging.warning("No valid trainer_state.json found, training from scratch.")
                    trainer.train()
        else:
            trainer.train()
        
        # Save final model
        logging.info(f"Saving model to {exp_dir}")
        trainer.save_model(exp_dir)

        # Save processor for inference convenience
        if hasattr(model, "processor"):
            model.processor.save_pretrained(exp_dir)

if __name__ == "__main__":
    main()