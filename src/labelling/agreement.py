"""Inter-model agreement: Groq cloud labels vs local Ollama labels on the awards both labelled.

  python scripts/agreement.py [--cloud data/qwen_labels.jsonl] [--local data/qwen_labels_local.jsonl]

Ground sets per award use valid taxonomy ids only. Reports exact set match, mean Jaccard,
per-label agreement with Cohen's kappa, and the G1/G7 cross-tab.
"""

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
IDS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G99"]


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["uid"], {g["id"] for g in r["grounds"] if g["id"] in IDS})
    return out


def main(cloud_path, local_path):
    cloud, local = load(cloud_path), load(local_path)
    both = sorted(set(cloud) & set(local))
    print(f"cloud {len(cloud)} awards, local {len(local)}, overlap {len(both)}")
    if not both:
        return
    exact = sum(cloud[u] == local[u] for u in both)
    jac = [len(cloud[u] & local[u]) / len(cloud[u] | local[u]) if cloud[u] | local[u] else 1.0 for u in both]
    print(f"EXACT ground-set match: {exact}/{len(both)} = {exact / len(both):.1%}; mean Jaccard {sum(jac) / len(jac):.3f}")
    rows = []
    for g in IDS:
        a = [g in cloud[u] for u in both]
        b = [g in local[u] for u in both]
        if not any(a) and not any(b):
            continue
        k = cohen_kappa_score(a, b) if len(set(a)) > 1 or len(set(b)) > 1 else float("nan")
        rows.append({"ground": g, "cloud": sum(a), "local": sum(b), "both": sum(x and y for x, y in zip(a, b)),
                     "agree_%": round(100 * sum(x == y for x, y in zip(a, b)) / len(both), 1), "kappa": round(k, 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    ct = pd.crosstab(pd.Series([("G1" in cloud[u]) * 1 + ("G7" in cloud[u]) * 2 for u in both], name="cloud"),
                     pd.Series([("G1" in local[u]) * 1 + ("G7" in local[u]) * 2 for u in both], name="local"))
    lab = {0: "neither", 1: "G1 only", 2: "G7 only", 3: "G1+G7"}
    print("G1/G7 cross-tab (rows cloud, cols local):")
    print(ct.rename(index=lab, columns=lab).to_string())
    diff = [u for u in both if cloud[u] != local[u]]
    print(f"disagreements: {len(diff)}; first 10:")
    for u in diff[:10]:
        print(f"  {u.split('|')[0]}: cloud {sorted(cloud[u])} local {sorted(local[u])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default=os.path.join(DATA, "qwen_labels.jsonl"))
    ap.add_argument("--local", default=os.path.join(DATA, "qwen_labels_local.jsonl"))
    a = ap.parse_args()
    main(a.cloud, a.local)
