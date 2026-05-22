# ./dataset.py

import argparse
import json
import os
import re
import copy

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import numpy as np
import librosa
import logging

class BBox:
    def __init__(self, class_name, x, y, z, l, w, h, index=0):
        self.class_name = class_name
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.l = float(l)
        self.w = float(w)
        self.h = float(h)
        self.index = index

    def process_simple(self):
        # convert to cm and clip to int
        self.x_bin = int(self.x * 100)
        self.y_bin = int(self.y * 100)
        self.z_bin = int(self.z * 100)
        
        self.l_bin = int(self.l * 100)
        self.w_bin = int(self.w * 100)
        self.h_bin = int(self.h * 100)

    def shift_scale(self, shift, scale):
        self.x = (self.x - shift[0]) * scale
        self.y = (self.y - shift[1]) * scale
        self.z = (self.z - shift[2]) * scale

        self.l = self.l * scale
        self.w = self.w * scale
        self.h = self.h * scale

    @staticmethod
    def _quantize(val, min_val, max_val, num_bins):
        """Helper method to normalize and discretize a value."""
        norm = (val - min_val) / (max_val - min_val)
        norm = max(0.0, min(1.0, norm))
        return int(norm * (num_bins - 1))

    def discretize(self, num_bins=1000, world_max=2.0, scale_max=0.625):
        # range: [-world_max, world_max] -> [0, num_bins]
        self.x_bin = self._quantize(self.x, -world_max, world_max, num_bins)
        self.y_bin = self._quantize(self.y, -world_max, world_max, num_bins)
        self.z_bin = self._quantize(self.z, -world_max, world_max, num_bins)
        
        # scale: [0, scale_max] -> [0, num_bins]
        self.l_bin = self._quantize(self.l, 0, scale_max, num_bins)
        self.w_bin = self._quantize(self.w, 0, scale_max, num_bins)
        self.h_bin = self._quantize(self.h, 0, scale_max, num_bins)

    def to_string(self):
        return f"bbox_{self.index}=Bbox({self.class_name}, {self.x_bin}, {self.y_bin}, {self.z_bin}, {self.l_bin}, {self.w_bin}, {self.h_bin})"

def parse_bbox_text(text):
    bboxes = []

    pattern = r"bbox_(\d+)=Bbox\(([^,]+),\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\)"
    matches = re.findall(pattern, text)
    
    for m in matches:
        idx, cls, x, y, z, l, w, h = m
        bboxes.append(BBox(cls, x, y, z, l, w, h, index=idx))
    return bboxes

class SALMONN3DDataset(Dataset):
    def __init__(self, ann_path, tasks=None, target_resolution=(672, 378), enable_audio=True, enable_visual=True):
        super().__init__()
        self.target_resolution = target_resolution # (width, height)
        self.enable_audio = enable_audio
        self.enable_visual = enable_visual
        if ann_path is None:
            self.annotation = []
        else:
            with open(ann_path, "r") as f:
                data = json.load(f)
            # expect a top-level "annotation" field
            self.annotation = data.get("annotation", data)

        if tasks:
            original_len = len(self.annotation)
            self.annotation = [ann for ann in self.annotation if ann.get("task", "") in tasks]
            filtered_len = len(self.annotation)
            logging.info(f"Filtered annotations by tasks {tasks}: {original_len} -> {filtered_len}")

    def __len__(self):
        return len(self.annotation)

    def _depth_to_point_cloud(self, depth_img, hfov, vfov):
        """
        Convert a depth image to a point cloud.
        hfov, vfov: in degrees
        +x: right
        +y: up
        +z: backward
        """
        if depth_img.size != self.target_resolution:
            depth_img = depth_img.resize(self.target_resolution, Image.NEAREST)
        w, h = depth_img.size
        depth = np.array(depth_img).astype(np.float32)
        # Convert depth map to meter
        depth = depth / 255 * 10.0

        # Camera intrinsics
        fx = (w / 2) / np.tan(np.deg2rad(hfov) / 2)
        fy = (h / 2) / np.tan(np.deg2rad(vfov) / 2)
        cx = w / 2
        cy = h / 2

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z_cam = -depth
        x_cam = (u - cx) * depth / fx
        y_cam = -(v - cy) * depth / fy
        point_cloud = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [H, W, 3]
        # print center slice of point cloud for debugging
        # print("Center slice of point cloud:", point_cloud[h//2-5:h//2+5, w//2-5:w//2+5, :])
        point_cloud = point_cloud.reshape(-1, 3)  # [H*W, 3]
        return point_cloud

    def __getitem__(self, index):
        while True:
            ann = self.annotation[index]
            base_path = ann["path"]
            basename = os.path.basename(base_path)
            is_valid_sample = True

            # --- Audio Loading ---
            raw_wav = None
            if self.enable_audio:
                wav_path = os.path.join(base_path, "foa_ls.wav")
                # wav_path = base_path
                if os.path.exists(wav_path):
                    try:
                        audio, sr = librosa.load(wav_path, sr=16000, mono=False)
                        max_length = 13 * 16000
                        if audio.shape[-1] > max_length:
                            audio = audio[:, :max_length]
                        if audio.ndim == 1:
                            waveform = audio.astype(np.float32)
                        else:
                            if audio.shape[0] > audio.shape[1]:
                                audio = audio.T # (T, C) -> (C, T)
                            if audio.shape[0] != 4:
                                raise ValueError(f"Expected 1 or 4 channels, got shape {audio.shape} for {wav_path}")
                            waveform = audio.astype(np.float32)  # [4, T]
                        if is_valid_sample:
                            raw_wav = torch.from_numpy(waveform)
                    except Exception as e:
                        logging.warning(f"Error loading audio file {wav_path}: {e}")
                        is_valid_sample = False
                elif os.path.exists(base_path):
                    # Fallback: try loading base_path directly if it's a wav file (for starss23)
                    try:
                        audio, sr = librosa.load(base_path, sr=16000, mono=False)
                        max_length = 15 * 16000
                        if audio.shape[-1] > max_length:
                            audio = audio[:, :max_length]
                        if audio.ndim == 1:
                            waveform = audio.astype(np.float32)
                        else:
                            if audio.shape[0] > audio.shape[1]:
                                audio = audio.T # (T, C) -> (C, T)
                            if audio.shape[0] != 4:
                                raise ValueError(f"Expected 1 or 4 channels, got shape {audio.shape} for {base_path}")
                            waveform = audio.astype(np.float32)  # [4, T]
                        if is_valid_sample:
                            raw_wav = torch.from_numpy(waveform)
                    except Exception as e:
                        logging.warning(f"Error loading audio file {base_path}: {e}")
                        is_valid_sample = False
                else:
                    logging.warning(f"Audio enabled but file not found: {wav_path}")
                    is_valid_sample = False

            # --- Visual Loading ---
            rgb_image = None
            point_cloud = None
            if self.enable_visual and is_valid_sample:
                # rgb_path = os.path.join(base_path, basename + "_rgb.png")
                rgb_path = os.path.join(base_path, "rgb.png")
                if os.path.exists(rgb_path):
                    try:
                        rgb_image = Image.open(rgb_path).convert("RGB")
                        if rgb_image.size != self.target_resolution:
                            rgb_image = rgb_image.resize(self.target_resolution, Image.BILINEAR)
                        # dep_path = os.path.join(base_path, basename + "_depth_viz.png")
                        dep_path = os.path.join(base_path, "depth.png")
                        if os.path.exists(dep_path):
                            depth_img = Image.open(dep_path)
                            hfov = ann.get("hfov", 90.0)
                            vfov = ann.get("vfov", 58.72)
                            point_cloud = self._depth_to_point_cloud(depth_img, hfov, vfov)
                        else:
                            logging.warning(f"Depth image not found for {rgb_path}, skipping point cloud generation.")
                            is_valid_sample = False
                    except Exception as e:
                        logging.warning(f"Error loading visual data for {rgb_path}: {e}")
                        is_valid_sample = False
                else:
                    logging.warning(f"Visual enabled but RGB image not found: {rgb_path}")
                    is_valid_sample = False
                    
            if not is_valid_sample:
                index = (index + 1) % len(self.annotation)
                continue
            
            break

        # --- Text & Meta info ---
        text = ann.get("text", "")
        task = ann.get("task", "")
        obj_category = ann.get("obj_category", [])
        gender = ann.get("gender", "")
        
        # Normalize Point Cloud and Transform BBoxes if applicable
        if point_cloud is not None:
            try:
                # 1. Filter valid points
                valid_mask = ~np.isnan(point_cloud).any(axis=1)
                pc_valid = point_cloud[valid_mask]

                if len(pc_valid) > 0:
                    # 2. Calculate Stats
                    # shift = np.min(pc_valid, axis=0)
                    # pc_centered = pc_valid - shift
                    # distances = np.linalg.norm(pc_centered, axis=1)
                    # avg_dist = np.mean(distances) + 1e-6
                    # scale = 1.0 / avg_dist

                    # # 3. Apply to Point Cloud (In-place update)
                    # # Note: We must update the original point_cloud, handling NaNs
                    # point_cloud = (point_cloud - shift) * scale
                    
                    # 4. Transform Text BBoxes (Only if text contains bbox info)
                    if "bbox_" in text:
                        bbox_objs = parse_bbox_text(text)
                        new_texts = []
                        for bbox in bbox_objs:
                            # bbox.shift_scale(shift, scale)
                            # # Using 2.0 max to cover potentially large scenes
                            # bbox.discretize(num_bins=1000, world_max=2.0, scale_max=2.0)
                            bbox.process_simple() # Or: use simple cm conversion
                            new_texts.append(bbox.to_string())
                        
                        if len(new_texts) > 0:
                            text = "\n".join(new_texts)
            
            except Exception as e:
                logging.warning(f"3D Normalization error for {base_path}: {e}")
                # Fallback: keep original text/pcd or skip based on preference

        return {
            "raw_wav": raw_wav,
            "rgb_image": rgb_image,
            "point_cloud": point_cloud,
            "text": text,
            "task": task,
            "id": base_path,
            "obj_category": obj_category,
            "gender": gender,
        }

if __name__ == "__main__":
    # --- Debugging Block ---
    parser = argparse.ArgumentParser(description="Debug SALMONN3DDataset components")
    parser.add_argument("--dep_img", type=str, required=True, help="Path to the depth image for debugging")
    parser.add_argument("--hfov", type=float, default=90.0, help="Horizontal FOV in degrees")
    parser.add_argument("--vfov", type=float, default=58.72, help="Vertical FOV in degrees")
    args = parser.parse_args()

    depth_img = Image.open(args.dep_img)
    dataset = SALMONN3DDataset(ann_path=None)
    point_cloud = dataset._depth_to_point_cloud(depth_img, args.hfov, args.vfov)
    print(f"Generated point cloud shape: {point_cloud.shape}")  