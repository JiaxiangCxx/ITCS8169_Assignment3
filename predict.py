"""Inference from raw images with the saved one-shot model.

  python predict.py --model checkpoints/oneshot_model.npz --images data/test --out predictions_test.csv

--images may be a folder with one sub-folder per class (accuracy is printed) or a flat folder.
Transductive methods adapt to the whole folder of unlabeled images at once.
"""
import argparse
import csv
import json
import os

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader

from extract_features import Images, encode, view_transforms
from src.common import CLASSES, list_split
from src.methods import DEFAULTS, METHODS, l2n


@torch.no_grad()
def embed_folder(arch, pretrained, root, split, workers=8, bs=32):
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = model.half().cuda().eval()
    views, _ = view_transforms(model, preprocess)
    paths, labels = list_split(root, split)
    feats = []
    for v in ("center", "flip"):
        dl = DataLoader(Images(root, paths, views[v]), batch_size=bs, num_workers=workers)
        feats.append(encode(model, dl, "cuda").astype(np.float32))
    del model
    torch.cuda.empty_cache()
    return l2n(np.concatenate(feats, 1).mean(1)), paths, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/oneshot_model.npz")
    ap.add_argument("--images", required=True, help="e.g. data/test or data/test2")
    ap.add_argument("--method", default=None, help="override the saved method")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()

    m = np.load(a.model)
    tags = list(m["tags"])
    method = a.method or str(m["method"])
    kw = {**DEFAULTS, **json.loads(str(m["params"]))}
    root, split = os.path.split(os.path.normpath(a.images))
    k = np.sqrt(len(tags))
    xs, Ts, Ss = [], [], []
    for t in tags:
        arch, pre = m[f"{t}/arch"]
        x, paths, labels = embed_folder(str(arch), str(pre), root, split)
        xs.append(x / k)
        Ts.append(m[f"{t}/text"] / k)
        Ss.append(m[f"{t}/support"].astype(np.float32) / k)
        print("encoded with", t, flush=True)
    x, T, S = (np.concatenate(v, -1) for v in (xs, Ts, Ss))
    P = l2n(S.mean(1))
    scores = METHODS[method](x, T, S, P, 100.0, kw)
    pred = scores.argmax(1)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "prediction"])
        for p, c in zip(paths, pred):
            w.writerow([p, CLASSES[c]])
    print(f"wrote {len(pred)} predictions to {a.out} (method={method})")
    if (labels >= 0).all():
        print(f"accuracy: {(pred == labels).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
