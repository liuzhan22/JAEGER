# ./utils/logger.py

import logging
import os
import sys
import warnings
import transformers
import torch.distributed as dist
from datetime import datetime

def is_main_process():
    env_rank= os.environ.get("RANK", None)
    if env_rank is not None:
        return int(env_rank) == 0
    return True

def now():
    return datetime.now().strftime("%Y%m%d%H%M")

def human_format(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)

def setup_logger(output_dir=None, log_filename=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if is_main_process() and output_dir and log_filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_filename)
        handlers.append(logging.FileHandler(log_path, mode="w"))

    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
        force=True
    )
    warnings.filterwarnings("ignore", message=".*Some weights of the model checkpoint.*talker.*")
    if is_main_process():
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

def print_trainable_parameters(model):
    if not is_main_process():
        return

    trainable_params = 0
    all_param = 0
    
    logging.info("="*60)
    logging.info("GLOBAL PARAMETER INSPECTION")
    logging.info("="*60)
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logging.info(f"[Trainable] {name} | {human_format(param.numel())}")
    
    logging.info("-" * 60)
    logging.info(f"Total Trainable: {human_format(trainable_params)} | Ratio: {100 * trainable_params / all_param:.4f}%")
    logging.info(f"Total Parameters: {human_format(all_param)}")
    logging.info("="*60 + "\n")