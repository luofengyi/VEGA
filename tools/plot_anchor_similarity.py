#!/usr/bin/env python
"""Visualize utterance-anchor similarity in 2D."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm

# Ensure repository root is importable when running as: python tools/plot_anchor_similarity.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.iemocap_config import IEMOCAP_CONFIG
from main import create_model, setup_data_and_loss
from vega_utils.anchor_utils import get_anchors
from vega_utils.common import emotion_labels, seed_everything


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    cfg = IEMOCAP_CONFIG.copy()

    args = argparse.Namespace()
    args.Dataset = cli_args.dataset
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.cuda = torch.cuda.is_available() and not cli_args.cpu

    args.audio_dim = 1582 if args.Dataset == "IEMOCAP" else 300
    args.visual_dim = 342
    args.text_dim = 1024
    args.n_speakers = 9 if args.Dataset == "MELD" else 2
    args.n_classes = 7 if args.Dataset == "MELD" else 6

    args.hidden_dim = cfg["hidden_dim"]
    args.n_head = cfg["n_head"]
    args.dropout = cfg["dropout"]
    args.outlayer_drop = cfg["outlayer_drop"]
    args.outlayer_num = cfg["outlayer_num"]
    args.outlayer_activation_fn = cfg["outlayer_activation_fn"]

    args.clip_loss = True
    args.clip_dim = cfg["clip_dim"]
    args.clip_proj_layer_num = cfg["clip_proj_layer_num"]
    args.clip_proj_activation_fn = cfg["clip_proj_activation_fn"]
    args.clip_proj_drop = cfg["clip_proj_drop"]

    args.expr_img_folder = cli_args.expr_img_folder
    args.rand = cfg["rand"]
    args.CLIP_Model = cfg["CLIP_Model"]
    return args


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    ckp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckp.get("state_dict", ckp) if isinstance(ckp, dict) else ckp
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[plot] warning: missing keys count={len(missing)}")
    if unexpected:
        print(f"[plot] warning: unexpected keys count={len(unexpected)}")


@torch.no_grad()
def extract_clip_space_features(args, model, test_loader, anchor_dict):
    model.eval()
    all_features = []
    all_labels = []

    anchor_center = F.normalize(anchor_dict["anchor_center"], dim=-1)
    anchor_center = anchor_center.cpu().numpy()

    for data in tqdm(test_loader, desc="Extract features"):
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        textf = textf.permute(1, 2, 0)
        acouf = acouf.permute(1, 2, 0)
        visuf = visuf.permute(1, 2, 0)

        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, all_transformer_out = model(
            anchor_dict, textf, visuf, acouf, umask, qmask, lengths, False
        )

        clip_proj = model.all_clip_proj(all_transformer_out)
        clip_proj = F.normalize(clip_proj, dim=-1)
        flat_feat = clip_proj.reshape(-1, clip_proj.size(-1))
        flat_label = label.view(-1)

        valid_mask = flat_label != -1
        all_features.append(flat_feat[valid_mask].cpu().numpy())
        all_labels.append(flat_label[valid_mask].cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, anchor_center


def plot_tsne(features, labels, anchor_center, dataset, output_png: Path) -> None:
    merged = np.concatenate([features, anchor_center], axis=0)
    n_samples = merged.shape[0]
    perplexity = min(30, max(5, n_samples // 10))
    tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(merged)

    feat_coords = coords[: features.shape[0]]
    anchor_coords = coords[features.shape[0] :]

    cls_names = emotion_labels[dataset]
    colors = plt.cm.tab10(np.linspace(0, 1, len(cls_names)))

    plt.figure(figsize=(10, 7))
    for i, cls_name in enumerate(cls_names):
        idx = labels == i
        if np.any(idx):
            plt.scatter(feat_coords[idx, 0], feat_coords[idx, 1], s=14, c=[colors[i]], alpha=0.8, label=cls_name)

    plt.scatter(
        anchor_coords[:, 0],
        anchor_coords[:, 1],
        s=220,
        marker="*",
        c=colors[: len(cls_names)],
        edgecolors="black",
        linewidths=0.8,
        label="anchor_center",
    )
    for i, cls_name in enumerate(cls_names):
        plt.text(anchor_coords[i, 0], anchor_coords[i, 1], f" {cls_name}", fontsize=9, weight="bold")

    plt.title(f"VEGA Anchor Similarity t-SNE ({dataset})")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VEGA anchor similarity visualization.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--dataset", default="IEMOCAP", choices=["IEMOCAP", "MELD"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--expr_img_folder", default="35")
    parser.add_argument("--output_png", required=True, help="Path to output PNG figure.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=64)
    cli_args = parser.parse_args()

    seed_everything(cli_args.seed)
    args = build_args(cli_args)
    _, test_loader, _, _ = setup_data_and_loss(args)
    model = create_model(args)
    ckpt_path = resolve_path(cli_args.checkpoint)
    out_png_path = resolve_path(cli_args.output_png)

    load_checkpoint(model, ckpt_path)
    anchor_dict = get_anchors(args)
    features, labels, anchor_center = extract_clip_space_features(args, model, test_loader, anchor_dict)
    plot_tsne(features, labels, anchor_center, cli_args.dataset, out_png_path)
    print(f"[plot] figure saved: {out_png_path}")


if __name__ == "__main__":
    main()
