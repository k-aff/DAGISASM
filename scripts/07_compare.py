#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

def _bootstrap() -> None:
    """Find the `sadf` package whether it sits in src/ or at the repo root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        for cand in (parent / "src", parent):
            if (cand / "sadf" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise ImportError("Could not locate the `sadf` package.")


_bootstrap()

from sadf.utils.common import load_json

HEADLINE = ["accuracy", "precision", "recall", "auc", "fpr"]
GAPS = ["fpr_gap", "auc_gap", "tpr_gap", "equalized_odds_gap", "worst_group_auc"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_dirs", nargs=2)
    ap.add_argument("--names", nargs=2, default=["run_a", "run_b"])
    ap.add_argument("--group", default="skin_tone")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [f"# Comparison: {args.names[0]} vs {args.names[1]}", ""]
    overall = [load_json(Path(d) / "overall.json") for d in args.eval_dirs]
    gaps = [load_json(Path(d) / "gaps.json").get(args.group, {}) for d in args.eval_dirs]

    rows += ["## Overall detection", "",
             f"| metric | {args.names[0]} | {args.names[1]} | delta |",
             "| --- | --- | --- | --- |"]
    for m in HEADLINE:
        a, b = overall[0].get(m), overall[1].get(m)
        rows.append(f"| {m} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")

    rows += ["", f"## Fairness gaps (grouped by `{args.group}`)", "",
             f"| gap | {args.names[0]} | {args.names[1]} | delta (neg = fairer) |",
             "| --- | --- | --- | --- |"]
    for m in GAPS:
        a, b = gaps[0].get(m), gaps[1].get(m)
        if a is None or b is None:
            continue
        rows.append(f"| {m} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")

    for name, g in zip(args.names, gaps):
        ci = g.get("fpr_gap_ci")
        if ci:
            rows.append(f"\n{name} FPR-gap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    text = "\n".join(rows)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()