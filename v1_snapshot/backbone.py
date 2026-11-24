"""Frozen DINOv3 ViT-B/16 wrapper: uint8 frames -> (patch tokens, CLS)."""
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel

from common import BB_DIR, DIM, IMG


class FrozenDINOv3:
    def __init__(self, path=BB_DIR, device="cuda", dtype=torch.float16):
        self.model = AutoModel.from_pretrained(path, torch_dtype=dtype).to(device).eval()
        pre = json.load(open(os.path.join(path, "preprocessor_config.json")))
        self.mean = torch.tensor(pre["image_mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(pre["image_std"], device=device).view(1, 3, 1, 1)
        self.device, self.dtype = device, dtype
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, frames_u8):
        """frames_u8: uint8 tensor [B, H, W, 3] (any size, will resize) ->
        (patch [B, 196, DIM] fp16-cpu, cls [B, DIM] fp16-cpu)"""
        x = frames_u8.to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).to(self.dtype) / 255.0
        x = F.interpolate(x, size=(IMG, IMG), mode="bicubic", align_corners=False,
                          antialias=True)
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x).last_hidden_state  # [B, 1+reg+196, DIM]
        n_patch = (IMG // 16) ** 2
        cls, patch = out[:, 0], out[:, -n_patch:]           # layout: CLS, regs, patches
        assert patch.shape[1] == n_patch, patch.shape
        return patch.to("cpu", torch.float16), cls.to("cpu", torch.float16)
