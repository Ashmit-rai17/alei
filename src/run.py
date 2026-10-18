"""Project Alei — pipeline orchestrator.

Each stage reads the previous stage's file from disk, so any stage can be run
alone. discover/download touch the network and only run when named.

    python src/run.py                          # inventory segment redact phrases
    python src/run.py --only sample            # regenerates labeling_sample.xlsx — only when meant to
    python src/run.py --only redact phrases
    python src/run.py --skip sample
    python src/run.py --only discover download # network; download writes to --dest
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # run.py sits in src/
sys.path[:0] = [os.path.join(SRC_DIR, p) for p in ("acquisition", "parsing", "privacy", "labelling", "modelling", "reporting")]


DATA = os.environ.get("ALEI_DATA", os.path.join(ROOT, "data"))  # ALEI_DATA: data kept outside the repo
# sample is not in the default run: the shipped labeling sheet (data/v1_shipped/)
# was drawn from v1 segments, and redrawing it changes the award ids
DEFAULT = ["inventory", "segment", "redact", "phrases"]
ALL = ["discover", "download", "inventory", "segment", "redact", "sample", "phrases"]


def mtime(name):
    p = os.path.join(DATA, name)
    return os.path.getmtime(p) if os.path.exists(p) else None


def require_fresh(product, source, stage):
    """Refuse to build `product`'s successor from a file older than its own input."""
    p, s = mtime(product), (mtime(source) if source else None)
    if p is None:
        sys.exit(f"[{stage}] {product} is missing — run the stage that makes it first")
    if s is not None and p < s:
        sys.exit(f"[{stage}] {product} is older than {source} — it is stale. "
                 f"Re-run the stage that makes {product} first.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=ALL)
    ap.add_argument("--skip", nargs="+", choices=ALL, default=[])
    ap.add_argument("--dest", default=os.path.join(DATA, "pdfs"),
                    help="download target (use a staging dir while the pipeline is running)")
    a = ap.parse_args()
    stages = [s for s in (a.only or DEFAULT) if s not in a.skip]
    stages.sort(key=ALL.index)

    for st in stages:
        print(f"\n{'#' * 70}\n# {st}\n{'#' * 70}")
        if st == "discover":
            from discover import discover
            discover()
        elif st == "download":
            from download import download
            download(a.dest)
        elif st == "inventory":
            from inventory import inventory
            inventory()
        elif st == "segment":
            require_fresh("inventory.parquet", None, st)
            from segment import segment
            segment()
        elif st == "redact":
            # never redact a stale awards.parquet (this happened once)
            require_fresh("awards.parquet", "inventory.parquet", st)
            from redact import redact
            redact()
            print("\nGATE 2: a human reads data/redaction_review.txt before ANY file is shared.")
        elif st == "phrases":
            require_fresh("awards_safe.parquet", "awards.parquet", st)
            from phrases import phrases
            phrases()
        elif st == "sample":
            require_fresh("awards_safe.parquet", "awards.parquet", st)
            from sample import sample
            sample()
            print("\nlabeling_sample.xlsx is not shareable until gate 2 is signed off.")


if __name__ == "__main__":
    main()
