"""Evaluate every method on the labeled test set for each one-shot seed.

A model spec may join several backbones with '+', e.g. dfn5b_h14_378+siglip_so400m_384;
their normalised embeddings are concatenated (equivalent to averaging cosine similarities).

  python evaluate.py --models clip_l14_336 dfn5b_h14_378 dfn5b_h14_378+siglip_so400m_384
"""
import argparse
import glob
import json
import os

import numpy as np

from src.methods import DEFAULTS, METHODS, embed, l2n, load_features, support


def combined(feat_dir, spec):
    tags = spec.split("+")
    ds = [load_features(feat_dir, t) for t in tags]
    return ds


def prepare(ds, split_file, split="test", views=("center", "flip"), n_aug=32, pool=()):
    """pool: extra unlabeled splits appended after the evaluated split (for transductive methods)."""
    xs, Ts, Ss, Ps = [], [], [], []
    m = np.sqrt(len(ds))
    for d in ds:
        S, P = support(d, split_file, n_aug)
        xs.append(np.concatenate([embed(d, s, views) for s in (split, *pool)]) / m)
        Ts.append(l2n(d["text"].astype(np.float32)) / m)
        Ss.append(l2n(S) / m)
        Ps.append(P / m)
    cat = lambda a: np.concatenate(a, axis=-1)
    x, T, S, P = cat(xs), cat(Ts), cat(Ss), cat(Ps)
    return x, T, S, l2n(P), ds[0][f"{split}_labels"], float(np.mean([d["logit_scale"] for d in ds]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--features", default="features")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--views", nargs="+", default=["center", "flip"])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--set", nargs="*", default=[], help="override defaults, e.g. kappa=1 alpha=0.9")
    ap.add_argument("--pool", nargs="*", default=[],
                    help="extra unlabeled splits (e.g. test2) given to transductive methods")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    kw = dict(DEFAULTS)
    for s in args.set:
        k, v = s.split("=")
        kw[k] = type(DEFAULTS[k])(float(v))
    split_files = sorted(glob.glob(os.path.join(args.splits, "oneshot_seed*.json")),
                         key=lambda f: int(f.split("seed")[1].split(".")[0]))

    os.makedirs(args.out, exist_ok=True)
    rows = []
    print("cells: mean±std over seeds [seed 0]" + (f"; transductive pool += {args.pool}" if args.pool else ""))
    print(f"{'model':40s} " + " ".join(f"{m:>21s}" for m in args.methods))
    for spec in args.models:
        ds = combined(args.features, spec)
        accs = {m: [] for m in args.methods}
        for sf in split_files:
            x, T, S, P, y, ls = prepare(ds, sf, views=tuple(args.views), pool=tuple(args.pool))
            for m in args.methods:
                pred = METHODS[m](x, T, S, P, ls, kw).argmax(1)[: len(y)]
                accs[m].append(float((pred == y).mean() * 100))
        seed0 = {m: accs[m][0] for m in args.methods}
        name = spec.replace("clip_l14_336", "clipL").replace("siglip2_so400m_512", "sg2so").replace(
            "siglip2_gopt_384", "sg2g").replace("siglip_so400m_384", "sgso").replace("dfn5b_h14_378", "dfn").replace(
            "laion_bigg14", "bigG").replace("pe_core_bigg_448", "pe").replace("eva02_e14_plus", "evaE")
        print(f"{name:40s} " + " ".join(f"{np.mean(accs[m]):6.2f}±{np.std(accs[m]):4.2f} [{seed0[m]:5.2f}]".rjust(21)
                                         for m in args.methods), flush=True)
        rows.append({"model": spec, "views": args.views, "params": kw, "pool": args.pool,
                     "acc_per_seed": accs, "mean": {m: float(np.mean(v)) for m, v in accs.items()},
                     "std": {m: float(np.std(v)) for m, v in accs.items()}, "seed0": seed0})
    tag = "_".join(args.set) or "default"
    with open(os.path.join(args.out, f"eval_{tag}_{'-'.join(args.views)}.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
