"""
Backbones for binary OCT classification (MS vs control).

Two choices, selected by name:
  - "resnet"   ResNet50, ImageNet weights, 512x1024 input (the original model)
  - "retfound" RETFound ViT-L/16, MAE-pretrained on retinal images, 224x448

Everything in the federation — server reference weights, both clients, and
inference.py — builds its model through build_model() with the same arguments.
FedAvg validates state_dict name, order and shape between server and client, so
a second construction path is a second chance for them to disagree.

The RETFound checkpoint ships in HuggingFace ViTModel key naming with a square
14x14 positional embedding. _remap_hf_to_timm renames it to timm's layout and
_interpolate_pos_embed resizes the grid to the non-square target. Ported from
MS_circle/train_circle.py, whose loader is correct; the training recipe there is
what was broken, and that lives in train.py.
"""
import os
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# Single source of truth for the backbone names accepted anywhere: run-config,
# the /join response, and the inference CLI.
BACKBONES = ("resnet", "retfound")

# 224x448 -> a 14x28 grid = 392 tokens. Keeps the B-scans' 2:1 aspect at
# roughly RETFound's native 224 scale; 512x1024 would be 2048 tokens and ~30x
# the attention cost.
RETFOUND_INPUT_HW = (224, 448)
RETFOUND_MODEL = "vit_large_patch16_224"
RETFOUND_BLOCKS = 24


def get_resnet50_binary(pretrained: bool = True) -> nn.Module:
    """
    ResNet50 with Dropout and final FC layer adapted for 2 classes (binary classification).
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    return model


# ---------------------------------------------------------------- retfound

def _remap_hf_to_timm(checkpoint_model: dict) -> dict:
    """HuggingFace ViTModel key naming -> timm VisionTransformer key naming.

    HF keeps query/key/value as three separate projections; timm fuses them into
    one qkv matrix, so those are concatenated in q, k, v order at the end.
    """
    remapped = {}
    qkv_weights: dict = {}
    qkv_biases: dict = {}

    for k, v in checkpoint_model.items():
        if k == 'embeddings.cls_token':
            remapped['cls_token'] = v; continue
        if k == 'embeddings.position_embeddings':
            remapped['pos_embed'] = v; continue
        if k == 'embeddings.patch_embeddings.projection.weight':
            remapped['patch_embed.proj.weight'] = v; continue
        if k == 'embeddings.patch_embeddings.projection.bias':
            remapped['patch_embed.proj.bias'] = v; continue
        if k == 'layernorm.weight':
            remapped['fc_norm.weight'] = v; continue
        if k == 'layernorm.bias':
            remapped['fc_norm.bias'] = v; continue

        m = re.match(r'encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.(weight|bias)', k)
        if m:
            idx, qkv_name, param_type = m.group(1), m.group(2), m.group(3)
            store = qkv_weights if param_type == 'weight' else qkv_biases
            store.setdefault(idx, {})[qkv_name] = v
            continue

        m = re.match(r'encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)', k)
        if m:
            remapped[f'blocks.{m.group(1)}.attn.proj.{m.group(2)}'] = v; continue

        m = re.match(r'encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)', k)
        if m:
            remapped[f'blocks.{m.group(1)}.mlp.fc1.{m.group(2)}'] = v; continue

        m = re.match(r'encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)', k)
        if m:
            remapped[f'blocks.{m.group(1)}.mlp.fc2.{m.group(2)}'] = v; continue

        m = re.match(r'encoder\.layer\.(\d+)\.layernorm_before\.(weight|bias)', k)
        if m:
            remapped[f'blocks.{m.group(1)}.norm1.{m.group(2)}'] = v; continue

        m = re.match(r'encoder\.layer\.(\d+)\.layernorm_after\.(weight|bias)', k)
        if m:
            remapped[f'blocks.{m.group(1)}.norm2.{m.group(2)}'] = v; continue

        remapped[k] = v

    for idx in set(qkv_weights.keys()) | set(qkv_biases.keys()):
        if idx in qkv_weights and all(x in qkv_weights[idx] for x in ('query', 'key', 'value')):
            p = qkv_weights[idx]
            remapped[f'blocks.{idx}.attn.qkv.weight'] = torch.cat(
                [p['query'], p['key'], p['value']], dim=0)
        if idx in qkv_biases and all(x in qkv_biases[idx] for x in ('query', 'key', 'value')):
            p = qkv_biases[idx]
            remapped[f'blocks.{idx}.attn.qkv.bias'] = torch.cat(
                [p['query'], p['key'], p['value']], dim=0)

    return remapped


def _interpolate_pos_embed(model, checkpoint_model, quiet=False):
    """Resize the checkpoint pos_embed to the model's (possibly non-square) grid.

    Stock RETFound is 14x14. At 224x448 the target grid is 14x28, so this is a
    14x14 -> 14x28 bicubic resize: the height is untouched and the width is
    doubled.
    """
    if 'pos_embed' not in checkpoint_model:
        return

    ckpt_pe = checkpoint_model['pos_embed']
    model_pe = model.pos_embed

    if ckpt_pe.shape == model_pe.shape:
        if not quiet:
            print(f"  pos_embed already matches model {tuple(model_pe.shape)} — no interpolation")
        return

    embed_dim = ckpt_pe.shape[-1]
    num_prefix = getattr(model, 'num_prefix_tokens', 1)
    gh, gw = model.patch_embed.grid_size

    n_ckpt = ckpt_pe.shape[-2] - num_prefix
    orig = int(round(n_ckpt ** 0.5))
    if orig * orig != n_ckpt:
        print(f"  [WARN] checkpoint pos_embed has {n_ckpt} patch tokens — not a square "
              f"grid, and shape != model. Skipping (pos_embed will be randomly init'd).")
        checkpoint_model.pop('pos_embed')
        return

    if not quiet:
        print(f"  Interpolating pos_embed {orig}x{orig} -> {gh}x{gw}  "
              f"({n_ckpt} -> {gh*gw} tokens)")
    extra = ckpt_pe[:, :num_prefix]
    pos = ckpt_pe[:, num_prefix:]
    pos = pos.reshape(1, orig, orig, embed_dim).permute(0, 3, 1, 2)
    pos = F.interpolate(pos, size=(gh, gw), mode='bicubic', align_corners=False)
    pos = pos.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model['pos_embed'] = torch.cat((extra, pos), dim=1)


class RetFoundBinary(nn.Module):
    """RETFound ViT-L encoder + the same Dropout/Linear head the ResNet uses.

    state_dict order is `encoder.*` then `fc.1.*`. Nothing below may reorder or
    rename those keys — the federation matches them positionally.
    """

    def __init__(self, input_hw=RETFOUND_INPUT_HW, num_classes: int = 2):
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            RETFOUND_MODEL,
            pretrained=False,
            num_classes=0,
            global_pool='avg',
            img_size=tuple(input_hw),
        )
        self.input_hw = tuple(input_hw)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.encoder.num_features, num_classes),
        )

    def forward(self, x):
        return self.fc(self.encoder(x))


def _load_retfound_weights(model: RetFoundBinary, weights_path: str, quiet=False) -> None:
    """Load a RETFound checkpoint into model.encoder in place."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"RETFound weights not found: {weights_path}")

    enc = model.encoder
    if not quiet:
        print(f"  [retfound] loading weights: {weights_path}")

    if weights_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        raw_checkpoint = load_file(weights_path)
    else:
        ck = torch.load(weights_path, map_location='cpu', weights_only=False)
        raw_checkpoint = ck.get('model', ck)

    is_hf = any('embeddings.' in k or 'encoder.layer.' in k for k in raw_checkpoint.keys())
    checkpoint_model = _remap_hf_to_timm(raw_checkpoint) if is_hf else dict(raw_checkpoint)

    _interpolate_pos_embed(enc, checkpoint_model, quiet=quiet)

    enc_sd = enc.state_dict()
    enc_keys = set(enc_sd.keys())
    filtered = {k: v for k, v in checkpoint_model.items()
                if k in enc_keys and enc_sd[k].shape == v.shape}
    dropped = [k for k in checkpoint_model
               if k in enc_keys and enc_sd[k].shape != checkpoint_model[k].shape]
    msg = enc.load_state_dict(filtered, strict=False)

    if not quiet:
        print(f"  [retfound] loaded — matched {len(filtered)}/{len(enc_keys)}  "
              f"missing: {len(msg.missing_keys)}  unexpected: {len(msg.unexpected_keys)}")
    if dropped:
        print(f"  [retfound] WARN shape-mismatched keys dropped: {dropped[:8]}")
    if msg.missing_keys:
        print(f"  [retfound] WARN missing (random-init): {msg.missing_keys[:8]}")

    # Sanity gate: if the transformer blocks didn't load, the run is worthless.
    # Better to die here than to burn GPU-hours silently training from scratch.
    n_loaded_blocks = len({k.split('.')[1] for k in filtered if k.startswith('blocks.')})
    if n_loaded_blocks < len(enc.blocks):
        raise RuntimeError(
            f"[retfound] only {n_loaded_blocks}/{len(enc.blocks)} transformer blocks got "
            f"weights — the encoder did NOT load. Check the checkpoint key naming."
        )


def encoder_fingerprint(model: nn.Module) -> Optional[float]:
    """Mean |w| of the first attention projection, or None for a non-ViT.

    Cheap way to tell a loaded RETFound apart from a random init, and to confirm
    two sites are holding the same checkpoint. Random init lands near 0.026;
    the real checkpoint is distinctly different.
    """
    sd = model.state_dict()
    w = sd.get("encoder.blocks.0.attn.qkv.weight")
    if w is None:
        return None
    return float(w.detach().abs().mean())


def build_model(
    backbone: str = "resnet",
    weights_path: Optional[str] = None,
    finetune: bool = True,
    input_hw=RETFOUND_INPUT_HW,
    pretrained: bool = True,
    quiet: bool = False,
) -> Tuple[nn.Module, dict]:
    """
    Build a binary classifier and the spec describing it.

    backbone     "resnet" or "retfound"
    weights_path RETFound checkpoint; None builds the architecture only, which
                 is what the clients do — they are overwritten with the server's
                 global weights before the first gradient step either way
    finetune     retfound only. False freezes everything but fc and encoder.fc_norm
    input_hw     retfound model input (H, W). NOT the cache image size
    pretrained   resnet only: load ImageNet weights

    Returns (model, spec). The spec travels to the clients in the /join response
    so every site builds an identical state_dict.
    """
    if backbone not in BACKBONES:
        raise ValueError(
            f"unknown backbone {backbone!r}; expected one of {list(BACKBONES)}"
        )

    if backbone == "resnet":
        model = get_resnet50_binary(pretrained=pretrained)
        spec = {"backbone": "resnet", "finetune": True, "input_hw": None}
        return model, spec

    model = RetFoundBinary(input_hw=input_hw)
    gh, gw = model.encoder.patch_embed.grid_size
    if not quiet:
        print(f"  [retfound] ViT-L input {input_hw[0]}x{input_hw[1]} -> grid {gh}x{gw} "
              f"= {model.encoder.patch_embed.num_patches} tokens (+1 CLS)")

    if weights_path:
        _load_retfound_weights(model, weights_path, quiet=quiet)
    elif not quiet:
        print("  [retfound] no weights_path — architecture only "
              "(clients are seeded from the server's global weights)")

    if not finetune:
        trainable = ("fc_norm.weight", "fc_norm.bias")
        for name, p in model.encoder.named_parameters():
            p.requires_grad = name in trainable
        for p in model.fc.parameters():
            p.requires_grad = True
        if not quiet:
            print("  [retfound] frozen encoder (fc_norm + head trainable)")
    elif not quiet:
        print("  [retfound] fine-tuning end-to-end")

    spec = {
        "backbone": "retfound",
        "finetune": bool(finetune),
        "input_hw": [int(input_hw[0]), int(input_hw[1])],
    }
    return model, spec
