"""One-shot classifiers on cached, L2-normalised embeddings.

All methods return class scores [N, C]. Text and image similarities live on different scales
(the CLIP "modality gap"), so before mixing them each score matrix is standardised per sample.
"""
import json
import os

import numpy as np

# Defaults fixed before evaluation; sensitivity is reported separately.
DEFAULTS = dict(
    w_img=1.0,       # weight of the image-prototype scores relative to the text scores
    kappa=2.0,       # sharpness of soft assignments in the transductive step
    iters=20,        # transductive refinement iterations
    knn=10,          # neighbours in the label-propagation graph
    alpha=0.8,       # label-propagation strength
    tip_alpha=1.0, tip_beta=5.5,   # Tip-Adapter defaults from the paper
)


def l2n(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def zscore(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8)


def softmax(s):
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------- data helpers
def load_features(feat_dir, tag):
    d = dict(np.load(os.path.join(feat_dir, f"{tag}.npz")))
    d["views"] = list(d["views"])
    return d


def embed(d, split, views=("center", "flip")):
    idx = [d["views"].index(v) for v in views]
    return l2n(d[f"{split}_feats"][:, idx].astype(np.float32).mean(1))


def support(d, split_file, n_aug=32):
    """Support set for one seed: [C, V, D] views of the single image per class, and prototypes [C, D]."""
    picks = json.load(open(split_file))["train_images"]
    paths = list(d["shot_paths"])
    idx = [paths.index(p) for p in picks]
    assert list(d["shot_labels"][idx]) == list(range(len(picks))), "one image per class, in class order"
    base = d["shot_feats"][idx][:, :2].astype(np.float32)          # center + flip
    aug = d["shot_aug_feats"][idx][:, :n_aug].astype(np.float32)   # random augmentations
    S = np.concatenate([base, aug], axis=1)
    return S, l2n(S.mean(1))


# ---------------------------------------------------------------- inductive
def zero_shot(x, T, **_):
    return x @ T.T


def prototype(x, P, **_):
    return x @ P.T


def text_plus_proto(x, T, P, w_img=1.0, **_):
    return zscore(x @ T.T) + w_img * zscore(x @ P.T)


def tip_adapter(x, T, S, tip_alpha=1.0, tip_beta=5.5, **_):
    """Training-free Tip-Adapter (Zhang et al. 2022): support views as cache keys, mean affinity per class."""
    C, V, D = S.shape
    keys = S.reshape(C * V, D)
    aff = np.exp(-tip_beta * (1 - x @ keys.T)).reshape(len(x), C, V).mean(2)
    return 100.0 * (x @ T.T) + tip_alpha * aff


# ---------------------------------------------------------------- transductive (uses unlabeled test images)
def soft_kmeans(x, T, P, w_img=1.0, kappa=2.0, iters=20, **_):
    """Refine image-space class centroids on the unlabeled test batch.

    The text scores act as a fixed prior; each centroid stays anchored to its one-shot prototype.
    """
    zt = zscore(x @ T.T)
    mu = P.copy()
    for _ in range(iters):
        s = zt + w_img * zscore(x @ mu.T)
        q = softmax(kappa * s)
        mu = l2n(l2n(q.T @ x) + P)
    return zt + w_img * zscore(x @ mu.T)


def label_propagation(x, scores, knn=10, alpha=0.8, kappa=2.0, **_):
    """Smooth initial class probabilities over a kNN graph of the test images (Zhou et al. 2004)."""
    n = len(x)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :knn]
    W = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), knn)
    W[rows, nn.ravel()] = np.clip(sim[rows, nn.ravel()], 0, None) ** 3
    W = W + W.T
    dinv = 1.0 / np.sqrt(W.sum(1) + 1e-8)
    Sn = W * dinv[:, None] * dinv[None, :]
    Y0 = softmax(kappa * zscore(scores))
    return np.linalg.solve(np.eye(n) - alpha * Sn, Y0)


def kmeans_then_lp(x, T, P, **kw):
    return label_propagation(x, soft_kmeans(x, T, P, **kw), **kw)


INDUCTIVE = {
    "zero_shot": lambda x, T, S, P, ls, kw: zero_shot(x, T),
    "prototype": lambda x, T, S, P, ls, kw: prototype(x, P),
    "text+proto": lambda x, T, S, P, ls, kw: text_plus_proto(x, T, P, **kw),
    "tip_adapter": lambda x, T, S, P, ls, kw: tip_adapter(x, T, S, **kw),
}
TRANSDUCTIVE = {
    "zs+LP": lambda x, T, S, P, ls, kw: label_propagation(x, zero_shot(x, T), **kw),
    "text+proto+LP": lambda x, T, S, P, ls, kw: label_propagation(x, text_plus_proto(x, T, P, **kw), **kw),
    "softkmeans": lambda x, T, S, P, ls, kw: soft_kmeans(x, T, P, **kw),
    "softkmeans+LP": lambda x, T, S, P, ls, kw: kmeans_then_lp(x, T, P, **kw),
}
METHODS = {**INDUCTIVE, **TRANSDUCTIVE}
