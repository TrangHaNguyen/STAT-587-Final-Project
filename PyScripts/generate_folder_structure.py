#!/usr/bin/env python3
"""Generate a folder-only project structure as text, PNG, and LaTeX include."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    ".DS_Store",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".mplconfig",
}


def build_folder_tree_lines(root: Path, *, exclude_names: set[str]) -> list[str]:
    """Return an ASCII tree containing directories only, at all nesting levels."""

    def _dirs(path: Path) -> list[Path]:
        return sorted(
            [
                child
                for child in path.iterdir()
                if child.is_dir() and child.name not in exclude_names
            ],
            key=lambda p: p.name.lower(),
        )

    lines = [f"{root.name}/"]

    def _walk(path: Path, prefix: str) -> None:
        children = _dirs(path)
        for idx, child in enumerate(children):
            is_last = idx == len(children) - 1
            branch = "`-- " if is_last else "|-- "
            lines.append(f"{prefix}{branch}{child.name}/")
            extension = "    " if is_last else "|   "
            _walk(child, prefix + extension)

    _walk(root, "")
    return lines


def render_tree_png(lines: list[str], output_path: Path) -> None:
    """Render the tree lines to a PNG using Pillow only."""
    font = ImageFont.load_default()
    padding_x = 24
    padding_y = 20
    line_gap = 8

    max_width = 0
    line_heights = []
    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        max_width = max(max_width, width)
        line_heights.append(height)

    avg_height = max(line_heights) if line_heights else 12
    width = max_width + 2 * padding_x
    height = len(lines) * (avg_height + line_gap) + 2 * padding_y

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    y = padding_y
    for line in lines:
        draw.text((padding_x, y), line, fill="black", font=font)
        y += avg_height + line_gap

    image.save(output_path)


def write_latex_include(output_path: Path, png_name: str, caption: str, label: str) -> None:
    snippet = f"""\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth,height=0.82\\textheight,keepaspectratio]{{../output/{png_name}}}
  \\caption{{{caption}}}
  \\label{{{label}}}
\\end{{figure}}
"""
    output_path.write_text(snippet, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a folder-only structure snapshot for the project.")
    parser.add_argument(
        "--output-prefix",
        default="project",
        help="Prefix for generated files in output/.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = build_folder_tree_lines(project_root, exclude_names=DEFAULT_EXCLUDES)

    txt_path = output_dir / f"{args.output_prefix}_folder_structure.txt"
    png_path = output_dir / f"{args.output_prefix}_folder_structure.png"
    tex_path = output_dir / f"{args.output_prefix}_folder_structure_include.tex"

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_tree_png(lines, png_path)
    write_latex_include(
        tex_path,
        png_path.name,
        caption="Folder structure of the project repository used for this report.",
        label="fig:project_folder_structure",
    )

    print(f"Wrote tree text to: {txt_path}")
    print(f"Wrote PNG figure to: {png_path}")
    print(f"Wrote LaTeX include snippet to: {tex_path}")


if __name__ == "__main__":
    main()
