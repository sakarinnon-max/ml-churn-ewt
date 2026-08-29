"""Build .ipynb files from a plain-Python cell list (authoring helper).

Chapter sources live in notebooks_src/chXX_cells.py as:

    TITLE = "01 — นิยามปัญหา + สร้าง label"
    CELLS = [
        ("md", "## หัวข้อ ..."),
        ("code", "import pandas as pd\n..."),
    ]

Build one chapter:   python -m src.nb_build notebooks_src/ch01_cells.py
Build all chapters:  python -m src.nb_build --all
Output: notebooks/<name>.ipynb (name = source filename minus "_cells").
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build(src_path: Path) -> Path:
    spec = importlib.util.spec_from_file_location(src_path.stem, src_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3", "display_name": "Python 3", "language": "python",
    }
    for kind, source in mod.CELLS:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        elif kind == "code":
            nb.cells.append(nbf.v4.new_code_cell(source))
        else:
            raise ValueError(f"unknown cell kind {kind!r} in {src_path}")

    out = PROJECT_ROOT / "notebooks" / (src_path.stem.replace("_cells", "") + ".ipynb")
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    return out


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--all":
        srcs = sorted((PROJECT_ROOT / "notebooks_src").glob("ch*_cells.py"))
    else:
        srcs = [Path(a).resolve() for a in argv]
    if not srcs:
        print("usage: python -m src.nb_build <notebooks_src/chXX_cells.py> | --all")
        sys.exit(1)
    for s in srcs:
        print("built", build(s))


if __name__ == "__main__":
    main(sys.argv[1:])
