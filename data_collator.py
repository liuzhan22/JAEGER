# ./data_collator.py

import torch
import numpy as np
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class SALMONNDataCollator:
    processor: Any
    model_config: Any
    prompt_dict: Optional[Dict[str, List[str]]] = None
    padding: bool = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a list of dataset samples into a batch.
        """
        input_ids_list = []
        labels_list = []
        audio_features_list = []
        feature_attention_mask_list = []
        
        # Lists for custom FOA encoder inputs
        raw_wav_tensor_list = []
        raw_wav_lens_list = []

        image_features_list = []
        image_grid_thw_list = []
        point_cloud_list = []

        # Default frame: <|im_start|>system...<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n
        prompt_template = self.model_config.get("prompt_template", "{}")
        end_sym = self.model_config.get("end_sym", "<|im_end|>")

        for feature in features:
            # 1. Prepare Target (Ground Truth) from dataset 'text' field
            target_str = feature.get("text", "")
            if target_str is None:
                target_str = ""

            # 2. Prepare Instruction (Prompt) from prompt_dict based on 'task'
            task_name = feature.get("task", None)
            instruction = ""
            
            if self.prompt_dict and task_name in self.prompt_dict:
                template = random.choice(self.prompt_dict[task_name])
                # logging.info(f"Using prompt for task '{task_name}': {instruction}")
                if task_name == "visual_grounding":
                    obj_cats = feature.get("obj_category", [])
                    # Convert list ["a", "b", "c"] -> string "a, b, and c"
                    cat_str = "objects" # Fallback
                    if isinstance(obj_cats, list) and len(obj_cats) > 0:
                        # Deduplicate while preserving order if needed, but usually dataset is clean
                        if len(obj_cats) == 1:
                            cat_str = obj_cats[0]
                        else:
                            cat_str = ", ".join(obj_cats[:-1]) + f", and {obj_cats[-1]}"
                    elif isinstance(obj_cats, str):
                        cat_str = obj_cats
                    instruction = template.format(obj_category=cat_str)
                elif task_name == "multi_spatial_doa":
                    gender_raw = feature.get("gender", "")
                    gender_map = {"M": "male", "F": "female"}
                    gender_str = gender_map.get(gender_raw, "unknown")
                    instruction = template.format(gender=gender_str)
                elif task_name == "dual_source":
                    gender_str = feature.get("gender", "unknown")
                    instruction = template.format(gender=gender_str)
                elif task_name in ["multi_spatial_doa_vgg", "dual_source_vgg"]:
                    category_str = feature.get("category", "unknown")
                    instruction = template.format(category=category_str)
                elif task_name == "spatial_doa_vgg":
                    instruction = template
                elif task_name == "dual_source_more_cand":
                    gender_str = feature.get("gender", "unknown")
                    num_speakers = feature.get("num_speakers", 0)
                    ordinals = []
                    for i in range(1, num_speakers + 1):
                        if i == 1:
                            ordinals.append("1st")
                        elif i == 2:
                            ordinals.append("2nd")
                        elif i == 3:
                            ordinals.append("3rd")
                        else:
                            ordinals.append(f"{i}th")
                    ordinal_list = ", ".join(ordinals)
                    try:
                        instruction = template.format(gender=gender_str, ordinals=ordinal_list)
                    except KeyError:
                        instruction = template.format(gender=gender_str)
                elif task_name == "single_source_more_cand":
                    num_speakers = feature.get("num_speakers", 0)
                    ordinals = []
                    for i in range(1, num_speakers + 1):
                        if i == 1:
                            ordinals.append("1st")
                        elif i == 2:
                            ordinals.append("2nd")
                        elif i == 3:
                            ordinals.append("3rd")
                        else:
                            ordinals.append(f"{i}th")
                    ordinal_list = ", ".join(ordinals)
                    try:
                        instruction = template.format(ordinals=ordinal_list)
                    except KeyError:
                        instruction = template
                elif task_name == "starss23":
                    gender_raw = feature.get("gender", "")
                    gender_map = {"M": "male", "F": "female"}
                    gender_str = gender_map.get(gender_raw, "unknown")
                    instruction = template.format(gender=gender_str)
                else:
                    instruction = template
            else:
                # Fallback instruction if task/dict is missing
                instruction = "Describe the audio."

            # 3. Dynamic Modality Prefix Construction
            modality_prefix = ""
            
            # Check for Video/Image
            rgb_image = feature.get("rgb_image", None)
            point_cloud = feature.get("point_cloud", None)
            if rgb_image is not None:
                modality_prefix += "<|vision_bos|><|IMAGE|><|vision_eos|>\n"

            # Check for Audio
            raw_wav = feature.get("raw_wav")
            if raw_wav is not None:
                modality_prefix += "<|audio_bos|><|AUDIO|><|audio_eos|>\n"

            # 4. Construct Full Prompt
            # Content = [Modality Tokens] + [Instruction]
            full_user_content = modality_prefix + instruction
            
            # Apply ChatML template
            prompt_text = prompt_template.format(full_user_content)

            # 5. Construct Full Training Text (Prompt + Target + EOS)
            if target_str:
                full_train_text = f"{prompt_text}{target_str}{end_sym}"
                # print(f"Full Train Text: {full_train_text}")
            else:
                full_train_text = prompt_text

            # 6. Process Audio Data (Numpy -> Tensor)
            if isinstance(raw_wav, torch.Tensor):
                wav_np = raw_wav.cpu().numpy()
            else:
                wav_np = raw_wav

            ch0 = None
            foa = None

            if wav_np is not None:
                # Handle multi-channel
                if isinstance(wav_np, np.ndarray) and wav_np.ndim == 2:
                    if wav_np.shape[0] > wav_np.shape[1]:
                        wav_np = wav_np.T # Ensure [C, T]
                    ch0 = wav_np[0] # Channel 0 for Whisper/Qwen-Audio Encoder
                    foa = wav_np    # All channels for FOA Encoder
                else:
                    ch0 = wav_np
                    foa = None # Single channel has no FOA info

            # 7. Processor Tokenization
            # Qwen Processor handles special tokens and mel extraction
            processor_inputs = {
                "text": [full_train_text],
                "return_tensors": "pt",
                "padding": True
            }
            if ch0 is not None:
                processor_inputs["audio"] = [ch0]
            if rgb_image is not None:
                processor_inputs["images"] = [rgb_image]

            inputs = self.processor(**processor_inputs)
            # print(inputs)

            # Collect Input IDs
            input_ids = inputs.input_ids[0]
            input_ids_list.append(input_ids)

            # Collect Mel Features
            if hasattr(inputs, "input_features") and inputs.input_features is not None:
                audio_features_list.append(inputs.input_features[0])
                if hasattr(inputs, "feature_attention_mask") and inputs.feature_attention_mask is not None:
                    feature_attention_mask_list.append(inputs.feature_attention_mask[0])

            # Collect Raw Wav (for internal FOA encoder)
            if foa is not None:
                raw_wav_lens_list.append(foa.shape[-1])
                raw_wav_tensor = torch.from_numpy(foa) 
                raw_wav_tensor_list.append(raw_wav_tensor)

            # Collect Visual Features
            if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
                image_features_list.append(inputs.pixel_values)
                if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                    image_grid_thw_list.append(inputs.image_grid_thw[0])

                if point_cloud is not None:
                    if isinstance(point_cloud, np.ndarray):
                        point_cloud = torch.from_numpy(point_cloud)
                    point_cloud_list.append(point_cloud)

            # 8. Create Labels (Masking the Prompt)
            if target_str:
                labels = input_ids.clone()
                # Tokenize only the target part to know its length
                # set add_special_tokens=False to avoid adding extra BOS/EOS
                target_tokens = self.processor.tokenizer(target_str + end_sym, add_special_tokens=False).input_ids
                # print(f"Target Tokens: {target_tokens}")
                target_len = len(target_tokens)
                
                # Mask everything except the target part
                if target_len < len(labels):
                    labels[:-target_len] = -100
                else:
                    # In case target occupies full length (rare/edge case)
                    pass 
                
                labels_list.append(labels)

        # --- Batch Assembly ---
        
        # Pad Input IDs
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        batch_attention_mask = batch_input_ids.ne(self.processor.tokenizer.pad_token_id).long()

        batch_data = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask
        }

        # Pad Labels
        if labels_list:
            batch_labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=-100
            )
            batch_data["labels"] = batch_labels

        # Stack Audio Features
        if audio_features_list:
            batch_data["input_features"] = torch.stack(audio_features_list)
            if feature_attention_mask_list:
                batch_data["feature_attention_mask"] = torch.stack(feature_attention_mask_list)

        # Pad and Stack Raw Wavs
        if raw_wav_tensor_list:
            max_len = max(w.shape[-1] for w in raw_wav_tensor_list)
            padded_wavs = []
            for wav in raw_wav_tensor_list:
                # wav shape: [4, T]
                current_len = wav.shape[-1]
                if current_len < max_len:
                    pad_amount = max_len - current_len
                    # Pad last dimension
                    wav = torch.nn.functional.pad(wav, (0, pad_amount), value=0)
                padded_wavs.append(wav)
            
            batch_data["raw_wav"] = torch.stack(padded_wavs)
            batch_data["raw_wav_lens"] = torch.tensor(raw_wav_lens_list, dtype=torch.long)

        # Stack Visual Features
        if image_features_list:
            batch_data["pixel_values"] = torch.cat(image_features_list, dim=0)
            if image_grid_thw_list:
                batch_data["image_grid_thw"] = torch.stack(image_grid_thw_list)

        if point_cloud_list:
            batch_data["point_clouds"] = torch.stack(point_cloud_list)

        return batch_data