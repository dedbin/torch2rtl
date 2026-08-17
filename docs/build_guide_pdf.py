#!/usr/bin/env python3
"""Build docs/torch2rtl_guide.pdf from the editable Markdown guide.

The repository environment used for this document has XeLaTeX but no pandoc.
This small converter supports only the Markdown constructs used by the guide:
headings, paragraphs, bullet lists, fenced code blocks, inline code, and simple
pipe tables. It deliberately has no third-party Python dependencies.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN = ROOT / "docs" / "torch2rtl_guide.md"
DEFAULT_PDF = ROOT / "docs" / "torch2rtl_guide.pdf"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--out", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--keep-tex", action="store_true")
    args = parser.parse_args()

    markdown_path = args.markdown.resolve()
    out_path = args.out.resolve()
    if not markdown_path.exists():
        raise FileNotFoundError(markdown_path)
    if shutil.which("xelatex") is None:
        raise RuntimeError("xelatex not found; install TeX Live XeTeX to build the PDF")

    markdown = markdown_path.read_text(encoding="utf-8")
    title = extract_title(markdown)
    latex = render_latex(title, markdown)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="torch2rtl-guide-") as tmp:
        tmp_path = Path(tmp)
        tex_path = tmp_path / "torch2rtl_guide.tex"
        tex_path.write_text(latex, encoding="utf-8")
        for _ in range(2):
            result = subprocess.run(
                [
                    "xelatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(tmp_path),
                    str(tex_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                log_path = tmp_path / "torch2rtl_guide.log"
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(
                    "xelatex failed\n"
                    f"stdout:\n{result.stdout[-2000:]}\n"
                    f"stderr:\n{result.stderr[-2000:]}\n"
                    f"log tail:\n{detail}"
                )
        shutil.copy2(tmp_path / "torch2rtl_guide.pdf", out_path)
        if args.keep_tex:
            shutil.copy2(tex_path, out_path.with_suffix(".tex"))

    try:
        display_path = out_path.relative_to(ROOT)
    except ValueError:
        display_path = out_path
    print(f"wrote {display_path}")
    return 0


def extract_title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "torch2rtl: руководство по проекту"


def render_latex(title: str, markdown: str) -> str:
    body = markdown_to_latex(markdown)
    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[a4paper,margin=22mm]{{geometry}}
\usepackage{{fontspec}}
\defaultfontfeatures{{}}
\setmainfont{{DejaVu Serif}}
\setsansfont{{DejaVu Sans}}
\setmonofont{{DejaVu Sans Mono}}[Scale=MatchLowercase]
\usepackage{{xcolor}}
\usepackage{{xurl}}
\usepackage{{hyperref}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{enumitem}}
\usepackage{{fvextra}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage{{parskip}}
\hypersetup{{unicode=true,colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black}}
\urlstyle{{tt}}
\definecolor{{codeframe}}{{HTML}}{{D8DED6}}
\definecolor{{codebg}}{{HTML}}{{F7F8F5}}
\DefineVerbatimEnvironment{{CodeBlock}}{{Verbatim}}{{fontsize=\small,breaklines=true,breakanywhere=true,frame=single,framesep=2mm,rulecolor=\color{{codeframe}}}}
\renewcommand{{\contentsname}}{{Содержание}}
\renewcommand{{\tablename}}{{Таблица}}
\renewcommand{{\figurename}}{{Схема}}
\setlist[itemize]{{leftmargin=*,itemsep=2pt,topsep=3pt}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{torch2rtl}}
\fancyhead[R]{{\thepage}}
\titlespacing*{{\section}}{{0pt}}{{1.2em}}{{0.5em}}
\titlespacing*{{\subsection}}{{0pt}}{{0.9em}}{{0.35em}}
\title{{{escape_latex(title)}}}
\author{{Подготовлено по текущему состоянию репозитория}}
\date{{17 августа 2026}}
\begin{{document}}
\begin{{titlepage}}
\centering
\vspace*{{24mm}}
{{\Huge\bfseries {escape_latex(title)}\par}}
\vspace{{12mm}}
{{\Large От PyTorch-модели к SystemVerilog-описанию\par}}
\vspace{{18mm}}
{{\large Редактируемый источник: \texttt{{docs/torch2rtl\_guide.md}}\par}}
{{\large Итоговый файл: \texttt{{docs/torch2rtl\_guide.pdf}}\par}}
\vfill
{{\large Подготовлено по текущему состоянию репозитория\par}}
{{\large 17 августа 2026\par}}
\end{{titlepage}}
\tableofcontents
\clearpage
{body}
\end{{document}}
"""


def markdown_to_latex(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    i = 0
    first_h1 = True

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            out.append(render_code_block(code_lines))
            continue

        if is_table_start(lines, i):
            table_lines: list[str] = [lines[i]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(render_table(table_lines))
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1 and first_h1:
                first_h1 = False
                i += 1
                continue
            first_h1 = False
            command = {1: "section", 2: "subsection", 3: "subsubsection"}.get(
                min(level, 3),
                "subsubsection",
            )
            out.append(rf"\{command}{{{format_inline(text)}}}")
            i += 1
            continue

        if stripped.startswith("- "):
            items: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            out.append(render_list(items))
            continue

        paragraph: list[str] = [stripped]
        i += 1
        while i < len(lines) and is_paragraph_continuation(lines, i):
            paragraph.append(lines[i].strip())
            i += 1
        out.append(format_inline(" ".join(paragraph)) + "\n")

    return "\n\n".join(out)


def is_paragraph_continuation(lines: list[str], index: int) -> bool:
    stripped = lines[index].strip()
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.startswith("- ") or stripped.startswith("```"):
        return False
    if is_table_start(lines, index):
        return False
    return True


def is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return lines[index].strip().startswith("|") and is_table_separator(lines[index + 1])


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    cells = split_table_row(stripped)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def split_table_row(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def render_table(lines: list[str]) -> str:
    rows = [split_table_row(line) for line in lines]
    if not rows:
        return ""
    column_count = len(rows[0])
    header = rows[0]
    width = 0.92 / max(column_count, 1)
    spec = "@{}" + "".join(
        rf">{{\raggedright\arraybackslash}}p{{{width:.3f}\textwidth}}"
        for _ in range(column_count)
    ) + "@{}"
    header_line = " & ".join(format_inline(cell) for cell in header) + r" \\"
    rendered_rows = []
    for row in rows[1:]:
        padded = row + [""] * (column_count - len(row))
        rendered_rows.append(
            " & ".join(format_inline(cell) for cell in padded[:column_count]) + r" \\"
        )
    body = "\n".join(rendered_rows)
    return (
        r"{\small" "\n"
        rf"\begin{{longtable}}{{{spec}}}" "\n"
        r"\toprule" "\n"
        f"{header_line}\n"
        r"\midrule" "\n"
        r"\endfirsthead" "\n"
        r"\toprule" "\n"
        f"{header_line}\n"
        r"\midrule" "\n"
        r"\endhead" "\n"
        f"{body}\n"
        r"\bottomrule" "\n"
        r"\end{longtable}" "\n"
        r"}"
    )


def render_code_block(lines: list[str]) -> str:
    text = "\n".join(lines).rstrip()
    return "\\begin{CodeBlock}\n" + text + "\n\\end{CodeBlock}"


def render_list(items: list[str]) -> str:
    rendered = [r"\begin{itemize}"]
    rendered.extend(rf"\item {format_inline(item)}" for item in items)
    rendered.append(r"\end{itemize}")
    return "\n".join(rendered)


def format_inline(text: str) -> str:
    parts = re.split(r"(`[^`]*`)", text)
    rendered: list[str] = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            rendered.append(format_code_span(part[1:-1]))
        else:
            rendered.append(format_bold(part))
    return "".join(rendered)


def format_code_span(text: str) -> str:
    if "{" in text or "}" in text or "\\" in text:
        return r"\texttt{\detokenize{" + text + "}}"
    return r"\nolinkurl{" + text + "}"


def format_bold(text: str) -> str:
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    rendered: list[str] = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            rendered.append(r"\textbf{" + escape_latex(part[2:-2]) + "}")
        else:
            rendered.append(escape_latex(part))
    return "".join(rendered)


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


if __name__ == "__main__":
    raise SystemExit(main())
