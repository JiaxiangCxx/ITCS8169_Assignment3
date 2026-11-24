"""Save the one-shot "trained model": per-backbone text classifier + one-shot support embeddings.

The backbones stay frozen, so the learned state is small: for each backbone, the class
text embeddings [C, D] and the embeddings of the 16 support images (2 views + 32 augmentations).

  python build_model.py --models pe_core_bigg_448+siglip2_gopt_384 --seed 0 --out checkpoints/oneshot_model.npz
"""
import argparse
import json
import os

import numpy as np

from src.common import CLASSES, MODELS
from src.methods import DEFAULTS, l2n, load_features, support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="backbones joined with '+'")
    ap.add_argument("--method", default="softkmeans+LP")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", default="features")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", default="checkpoints/oneshot_model.npz")
    a = ap.parse_args()

    split_file = os.path.join(a.splits, f"oneshot_seed{a.seed}.json")
    tags = a.models.split("+")
    out = {"tags": np.array(tags), "classes": np.array(CLASSES), "method": a.method,
           "params": json.dumps(DEFAULTS), "split": open(split_file).read()}
    for t in tags:
        d = load_features(a.features, t)
        S, _ = support(d, split_file)
        out[f"{t}/text"] = l2n(d["text"].astype(np.float32))
        out[f"{t}/support"] = l2n(S).astype(np.float16)
        out[f"{t}/arch"] = np.array(MODELS[t][:2])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **out)
    print("saved", a.out, {k: v.shape for k, v in out.items() if hasattr(v, "shape")})


if __name__ == "__main__":
    main()
