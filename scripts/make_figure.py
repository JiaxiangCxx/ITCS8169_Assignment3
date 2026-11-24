"""Report figure: test accuracy of each method over the 10 one-shot splits."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "results", "eval_default_center-flip.jsonl")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "report", "accuracy_seeds.pdf")

METHODS = [("zero_shot", "Zero-shot\n(text)"), ("prototype", "Prototype\n(1 image)"),
           ("text+proto", "Text +\nprototype"), ("tip_adapter", "Tip-\nAdapter"),
           ("softkmeans", "+ test-time\nk-means")]
SERIES = [("clip_l14_336", "CLIP ViT-L/14-336", "#2a78d6"),
          ("clip_l14_336+siglip2_so400m_512", "CLIP-L + SigLIP2-SO400M", "#eb6834")]

rows = {}
for line in open(SRC):
    r = json.loads(line)
    if not r.get("pool") and r["model"] not in rows and "softkmeans" in r["acc_per_seed"]:
        rows[r["model"]] = r

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(3.4, 2.25))
ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
ax.set_axisbelow(True)
for s, (tag, label, color) in enumerate(SERIES):
    for i, (m, _) in enumerate(METHODS):
        accs = rows[tag]["acc_per_seed"][m]
        x0 = i + (s - 0.5) * 0.3
        ax.scatter([x0] * len(accs), accs, s=9, color=color, alpha=0.55, linewidths=0,
                   label=label if i == 0 else None, zorder=3)
        mean = sum(accs) / len(accs)
        ax.plot([x0 - 0.11, x0 + 0.11], [mean, mean], color=color, linewidth=2, solid_capstyle="round", zorder=4)
best = rows[SERIES[1][0]]["mean"]["softkmeans"]
ax.annotate(f"{best:.1f}%", (4 + 0.15, best), xytext=(4 - 0.55, 97.2), fontsize=7, color="#0b0b0b",
            arrowprops=dict(arrowstyle="-", color="#8a8984", linewidth=0.6))
ax.set_xticks(range(len(METHODS)), [m[1] for m in METHODS], fontsize=6.5)
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(60, 99)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower right", fontsize=6.5, frameon=False, handletextpad=0.2, markerscale=1.5)
fig.tight_layout(pad=0.3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("saved", OUT)
