"""Optional: download all backbone checkpoints into the Hugging Face cache in parallel."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from open_clip.pretrained import download_pretrained, get_pretrained_cfg  # noqa: E402

from src.common import MODELS  # noqa: E402


def get(tag):
    arch, pre, _ = MODELS[tag]
    return tag, download_pretrained(get_pretrained_cfg(arch, pre))


if __name__ == "__main__":
    with ThreadPoolExecutor(8) as ex:
        for tag, path in ex.map(get, MODELS):
            print("OK", tag, path, flush=True)
