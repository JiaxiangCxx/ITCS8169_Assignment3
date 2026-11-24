"""Encode images and class prompts with a frozen foundation model and cache the embeddings.

Only the one-shot training images listed in splits/oneshot_seed*.json are ever read from train/.

  python extract_features.py --model siglip_so400m_384 --data data --out features
"""
import argparse
import glob
import json
import os

import numpy as np
import open_clip
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.common import CLASS_NAMES, CLASSES, MODELS, TEMPLATES, list_split

N_AUG = 32  # random augmented views of each one-shot image


class Images(Dataset):
    def __init__(self, root, paths, tf, n_views=1):
        self.root, self.paths, self.tf, self.n_views = root, paths, tf, n_views

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.root, self.paths[i])).convert("RGB")
        return torch.stack([self.tf(im) for _ in range(self.n_views)])


def view_transforms(model, preprocess):
    """Deterministic test-time views, using the model's input size and normalization."""
    size = model.visual.image_size
    size = size if isinstance(size, int) else size[0]
    tail = [t for t in preprocess.transforms if isinstance(t, (T.ToTensor, T.Normalize))]
    bic = T.InterpolationMode.BICUBIC
    center = T.Compose([T.Resize(size, interpolation=bic), T.CenterCrop(size)] + tail)
    flip = T.Compose([T.Resize(size, interpolation=bic), T.CenterCrop(size),
                      T.RandomHorizontalFlip(p=1.0)] + tail)
    squash = T.Compose([T.Resize((size, size), interpolation=bic)] + tail)
    aug = T.Compose([
        T.RandomResizedCrop(size, scale=(0.4, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3),
    ] + tail)
    return {"center": center, "flip": flip, "squash": squash}, aug


@torch.no_grad()
def encode(model, loader, device):
    out = []
    for x in loader:
        b, v = x.shape[:2]
        with torch.autocast("cuda", dtype=torch.float16):
            f = model.encode_image(x.flatten(0, 1).to(device, non_blocking=True))
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        out.append(f.view(b, v, -1).cpu())
    return torch.cat(out).numpy().astype(np.float16)


@torch.no_grad()
def encode_text(model, tokenizer, device):
    per_class = []
    for c in CLASSES:
        prompts = [t.format(n) for n in CLASS_NAMES[c] for t in TEMPLATES]
        with torch.autocast("cuda", dtype=torch.float16):
            e = model.encode_text(tokenizer(prompts).to(device))
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        per_class.append(torch.nn.functional.normalize(e.mean(0), dim=-1))
    return torch.stack(per_class).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", default="features")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(0)
    arch, pretrained, bs = MODELS[args.model]
    device = "cuda"
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = model.half().to(device).eval()  # cast first: EVA02-E does not fit on 16 GB in fp32
    tokenizer = open_clip.get_tokenizer(arch)
    views, aug = view_transforms(model, preprocess)
    print(args.model, preprocess, flush=True)

    bias = getattr(model, "logit_bias", None)
    res = {"text": encode_text(model, tokenizer, device),
           "logit_scale": float(model.logit_scale.exp()),
           "logit_bias": float(bias) if bias is not None else 0.0}

    def run(paths, tf, n_views=1):
        dl = DataLoader(Images(args.data, paths, tf, n_views), batch_size=max(1, bs // n_views),
                        num_workers=args.workers, pin_memory=True)
        return encode(model, dl, device)

    for split in ["test", "test2"]:
        paths, labels = list_split(args.data, split)
        res[f"{split}_paths"], res[f"{split}_labels"] = np.array(paths), labels
        res[f"{split}_feats"] = np.concatenate([run(paths, tf) for tf in views.values()], axis=1)
        print(split, res[f"{split}_feats"].shape, flush=True)

    # one-shot support images only (union over the pre-generated seeds)
    shots = sorted({p for f in glob.glob(os.path.join(args.splits, "oneshot_seed*.json"))
                    for p in json.load(open(f))["train_images"]})
    res["shot_paths"] = np.array(shots)
    res["shot_labels"] = np.array([CLASSES.index(p.split(os.sep)[1]) for p in shots])
    res["shot_feats"] = np.concatenate([run(shots, tf) for tf in views.values()], axis=1)
    res["shot_aug_feats"] = run(shots, aug, n_views=N_AUG)
    print("shots", res["shot_feats"].shape, res["shot_aug_feats"].shape, flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, f"{args.model}.npz"), views=np.array(list(views)), **res)
    print("saved", args.model, flush=True)


if __name__ == "__main__":
    main()
