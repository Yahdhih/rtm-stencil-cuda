"""Stencil RTM d'ordre 8 : medianes, tableau, figure.

Usage : python analysis/analyse.py results/raw_<job>.csv
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "figures").mkdir(exist_ok=True)
rows = list(csv.DictReader(open(sys.argv[1])))
g = defaultdict(list)
for r in rows:
    g[(r["version"], int(r["nx"]))].append(float(r["mpts_per_s"]))
med = {k: st.median(v) for k, v in g.items()}
GPU = ["naive", "shared", "sharedp", "shared-32x4", "shared-32x16", "shared-64x4", "shared-64x8", "shared-zs",
       "shared-64x4-zs"]
NS = [256, 512, 768]
md = ["## GPU : debit median (Gpoints/s) et bande passante effective (16 octets/point)\n",
      "| version | " + " | ".join(f"{n}^3" for n in NS) + " |", "|---" * (1 + len(NS)) + "|"]
for v in GPU:
    md.append(f"| {v} | " + " | ".join(f"{med[(v, n)] / 1e3:.1f} ({16 * med[(v, n)] / 1e3:.0f} Go/s)" for n in NS) + " |")
cpu = sorted([k for k in med if k[0].startswith("cpu")], key=lambda k: (k[0].rstrip("0123456789"), int(k[0].lstrip("cpu-flatbocked"))))
md += ["", "## CPU Grace (512^3), OpenMP\n", "| version, threads | Gpoints/s | Go/s eff. |", "|---|---|---|"]
for k in cpu:
    md.append(f"| {k[0][4:]} | {med[k] / 1e3:.2f} | {16 * med[k] / 1e3:.0f} |")
best512 = max(med[(v, 512)] for v in GPU)
md.append(f"\nMeilleur GPU / CPU 72 coeurs a 512^3 : {best512 / med[("cpu-blocked72", 512)]:.0f}x\n")
(ROOT / "results" / "summary.md").write_text("\n".join(md))
print("\n".join(md))

fig, ax = plt.subplots(figsize=(10, 4.4))
show = ["naive", "shared", "shared-64x4", "shared-zs", "shared-64x4-zs"]
col = {"naive": "#b5452c", "shared": "#d9822b", "shared-64x4": "#c9a227", "shared-zs": "#59a68f", "shared-64x4-zs": "#1b6e5a"}
w = 0.16
for i, v in enumerate(show):
    ax.bar([j + (i - 2) * w for j in range(len(NS))], [med[(v, n)] / 1e3 for n in NS], w, color=col[v], label=v)
ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"{n}^3" for n in NS])
ax.set_ylabel("Gpoints / s (plus haut = mieux)"); ax.set_title("Stencil d'ordre 8 (25 points), GH200, simple precision")
ax.set_ylim(0, 170); ax.legend(fontsize=8, ncol=5, loc="upper center"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "rtm_gpu.png", dpi=140)
