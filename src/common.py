"""Shared definitions: class list, prompts, model zoo, data listing and one-shot splits."""
import glob
import json
import os
import re

import numpy as np

CLASSES = [
    "Bedroom", "Coast", "Flower", "Forest", "Highway", "Industrial", "InsideCity", "Kitchen",
    "LivingRoom", "Mountain", "Office", "OpenCountry", "Store", "Street", "Suburb", "TallBuilding",
]

# Natural-language names for each folder name (written once, before looking at any results).
# Several synonyms per class; their text embeddings are averaged.
CLASS_NAMES = {
    "Bedroom": ["a bedroom", "a bedroom with a bed"],
    "Coast": ["a coast", "a beach", "a seashore"],
    "Flower": ["a flower", "flowers"],
    "Forest": ["a forest", "woods with trees"],
    "Highway": ["a highway", "a freeway"],
    "Industrial": ["an industrial area", "a factory", "an industrial plant"],
    "InsideCity": ["a city center", "downtown city buildings", "buildings inside a city"],
    "Kitchen": ["a kitchen"],
    "LivingRoom": ["a living room", "a living room with a sofa"],
    "Mountain": ["a mountain", "a mountain landscape"],
    "Office": ["an office", "an office with desks and computers"],
    "OpenCountry": ["open country", "an open countryside field", "a rural landscape"],
    "Store": ["a store", "the inside of a shop", "a supermarket aisle"],
    "Street": ["a street", "a city street"],
    "Suburb": ["a suburban house", "a suburb", "a residential house"],
    "TallBuilding": ["a tall building", "a skyscraper", "high-rise buildings"],
}

# Templates; the dataset is grayscale, so a few explicitly mention it.
TEMPLATES = [
    "a photo of {}.",
    "a black and white photo of {}.",
    "a grayscale photo of {}.",
    "a low resolution photo of {}.",
    "a picture of {}.",
    "an image of {}.",
    "a photo of {}, a type of scene.",
    "a black and white photo of {}, a type of scene.",
]

# tag -> (open_clip architecture, pretrained tag, batch size)
MODELS = {
    "clip_l14_336": ("ViT-L-14-336", "openai", 128),
    "dfn5b_h14_378": ("ViT-H-14-378-quickgelu", "dfn5b", 64),
    "siglip_so400m_384": ("ViT-SO400M-14-SigLIP-384", "webli", 64),
    "siglip2_so400m_512": ("ViT-SO400M-16-SigLIP2-512", "webli", 32),
    "laion_bigg14": ("ViT-bigG-14", "laion2b_s39b_b160k", 64),
    "eva02_e14_plus": ("EVA02-E-14-plus", "laion2b_s9b_b144k", 32),
    "pe_core_bigg_448": ("PE-Core-bigG-14-448", "meta", 16),
    "siglip2_gopt_384": ("ViT-gopt-16-SigLIP2-384", "webli", 32),
}


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_split(root, split):
    """Return (relative paths, labels). Unlabeled split (test2) gets label -1."""
    split_dir = os.path.join(root, split)
    if os.path.isdir(os.path.join(split_dir, CLASSES[0])):
        paths, labels = [], []
        for ci, c in enumerate(CLASSES):
            fs = sorted(glob.glob(os.path.join(split_dir, c, "*.jpg")), key=natural_key)
            paths += [os.path.relpath(f, root) for f in fs]
            labels += [ci] * len(fs)
        return paths, np.array(labels)
    fs = sorted(glob.glob(os.path.join(split_dir, "*.jpg")), key=natural_key)
    return [os.path.relpath(f, root) for f in fs], -np.ones(len(fs), dtype=int)


def oneshot_split(root, seed):
    """Pick exactly one training image per class with a fixed RNG seed (no looking at other data)."""
    paths, labels = list_split(root, "train")
    rng = np.random.default_rng(seed)
    picks = []
    for ci in range(len(CLASSES)):
        cand = [p for p, l in zip(paths, labels) if l == ci]
        picks.append(cand[int(rng.integers(len(cand)))])
    return picks


def save_splits(root, out_dir, seeds):
    os.makedirs(out_dir, exist_ok=True)
    for s in seeds:
        with open(os.path.join(out_dir, f"oneshot_seed{s}.json"), "w") as f:
            json.dump({"seed": s, "classes": CLASSES, "train_images": oneshot_split(root, s)}, f, indent=1)
