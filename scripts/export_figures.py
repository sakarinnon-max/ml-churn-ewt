"""Export chart images embedded in executed notebooks to PNG files.

Usage:
    ./venv/bin/python scripts/export_figures.py            # all charts -> figures/_preview/
    ./venv/bin/python scripts/export_figures.py --final    # only the curated list -> figures/

The curated list maps (notebook, image index within that notebook) to a
readable filename used in README.md.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (notebook stem, 0-based image index in execution order) -> output filename
CURATED: dict[tuple[str, int], str] = {
    ("ch03_eda", 1): "survival_curve.png",
    ("ch06_validation", 1): "model_comparison_bootstrap_ci.png",
    ("ch06_validation", 2): "threshold_vs_mentor_capacity.png",
    ("ch07_interpret", 0): "model_coefficients.png",
}


def iter_images(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    idx = 0
    for ci, cell in enumerate(nb.get("cells", [])):
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                yield idx, ci, base64.b64decode(data["image/png"])
                idx += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    if args.final:
        outdir = ROOT / "figures"
        outdir.mkdir(exist_ok=True)
        for (stem, want), name in CURATED.items():
            for idx, ci, png in iter_images(ROOT / "notebooks" / f"{stem}.ipynb"):
                if idx == want:
                    (outdir / name).write_bytes(png)
                    print(f"figures/{name}  <- {stem} image#{idx} (cell {ci})")
    else:
        outdir = ROOT / "figures" / "_preview"
        outdir.mkdir(parents=True, exist_ok=True)
        for nb_path in sorted((ROOT / "notebooks").glob("ch*.ipynb")):
            for idx, ci, png in iter_images(nb_path):
                name = f"{nb_path.stem}_img{idx:02d}_cell{ci:02d}.png"
                (outdir / name).write_bytes(png)
                print(name)


if __name__ == "__main__":
    main()
