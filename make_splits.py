"""Write splits/oneshot_seed{0..9}.json: exactly one randomly chosen training image per class.

Seed 0 is the main split; seeds 1-9 measure how much the choice of the single image matters.
"""
import argparse

from src.common import save_splits

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    save_splits(a.data, a.out, range(a.seeds))
    print(open(f"{a.out}/oneshot_seed0.json").read())
