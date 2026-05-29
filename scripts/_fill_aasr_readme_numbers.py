"""Patch the AASR README TBD placeholders with actual Stage 2 + Stage 4 numbers.

Reads ``tier_A_klados_aggregate.json`` and (if present)
``tier_C_iclabel_comparison.csv`` from
``reports/paper_validation/aasr/`` and rewrites the two TBD tables in the
README in-place. Idempotent: re-running it overwrites with the latest
numbers.
"""

# ruff: noqa: I001

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "paper_validation" / "aasr"
README = REPORT_DIR / "README.md"

VARIANT_ORDER = ("init", "mw", "psp", "psw")
CONTAM_FIRST = ("contaminated", "init", "mw", "psp", "psw")
ICLABEL_CLASSES_DISPLAY = (
    ("brain", "brain"),
    ("eye blink", "eye blink"),
    ("muscle artifact", "muscle artifact"),
    ("dipolar", "dipolar (brain >= 0.5)"),
)


def _fmt(v: float, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _build_tier_a_table(agg_path: Path) -> str | None:
    if not agg_path.exists():
        return None
    d = json.loads(agg_path.read_text(encoding="utf-8"))
    rows = [
        "| Variant | n | median Pearson | median SNR-imp (dB) | median RMSE reduction (%) | median wall time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in VARIANT_ORDER:
        stats = d["aggregate"].get(v, {})
        if not stats or stats.get("n", 0) == 0:
            rows.append(f"| {v} | 0 | n/a | n/a | n/a | n/a |")
            continue
        rows.append(
            f"| {v} | {stats['n']} | "
            f"{stats['mean_correlation']['median']:.4f} | "
            f"{stats['snr_improvement_db']['median']:+.3f} | "
            f"{stats['rmse_reduction_pct']['median']:+.2f} | "
            f"{stats['wall_time_s']['median']:.2f} |"
        )
    return "\n".join(rows)


def _build_tier_c_table(csv_path: Path) -> str | None:
    if not csv_path.exists():
        return None
    rows: dict[str, dict[str, int]] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows[r["variant"]] = r
    if not rows:
        return None
    lines = ["| Class | " + " | ".join(v for v in CONTAM_FIRST) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in CONTAM_FIRST) + "|")
    for label_display, csv_key in ICLABEL_CLASSES_DISPLAY:
        cells = [label_display]
        for v in CONTAM_FIRST:
            r = rows.get(v, {})
            cells.append(r.get(csv_key, "—"))
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def _replace_block(
    text: str, marker_start: str, marker_end: str, replacement: str
) -> str:
    start = text.find(marker_start)
    end = text.find(marker_end, start)
    if start < 0 or end < 0:
        return text
    return text[: start + len(marker_start)] + "\n" + replacement + "\n\n" + text[end:]


def main() -> int:
    if not README.exists():
        raise SystemExit(f"README not found: {README}")
    text = README.read_text(encoding="utf-8")

    tier_a = _build_tier_a_table(REPORT_DIR / "tier_A_klados_aggregate.json")
    if tier_a:
        # Replace the TBD Tier A table -- match its surrounding fences
        start_token = "| Variant | median Pearson | median SNR-imp (dB) | median RMSE reduction (%) | median wall time (s) |"
        if start_token in text:
            # Find the end (blank line after the table)
            start_idx = text.find(start_token)
            after_table_idx = text.find("\n\n", start_idx)
            text = text[:start_idx] + tier_a + text[after_table_idx:]
            print("Tier A table patched.")
        else:
            print("Tier A start token not found; skipping.")

    tier_c = _build_tier_c_table(REPORT_DIR / "tier_C_iclabel_comparison.csv")
    if tier_c:
        start_token = "| Class | contaminated | init | mw | psp | psw |"
        if start_token in text:
            start_idx = text.find(start_token)
            after_table_idx = text.find("\n\n", start_idx)
            text = text[:start_idx] + tier_c + text[after_table_idx:]
            print("Tier C table patched.")
        else:
            print("Tier C start token not found; skipping.")

    README.write_text(text, encoding="utf-8")
    print(f"Updated {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
