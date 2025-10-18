#!/usr/bin/env python3
"""
Usage:
  python table_convert.py study3_zeroshot_per_construct_metrics.json --title "Construct Metrics"

Dependencies:
  pip install pandas matplotlib tabulate
"""
import argparse
from pathlib import Path
from textwrap import fill
import sys
import pandas as pd
import matplotlib.pyplot as plt


def load_to_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() == ".json":
        return pd.read_json(p)
    else:
        raise ValueError("Input must be .csv or .json")


def pretty_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "freq_in_data" in df.columns:
        df["freq_in_data"] = (
            (df["freq_in_data"].astype(float) * 100).round(0).astype(int).astype(str) + "%"
        )

    for c in ["hum_gpt_kappa", "hum_gpt_precision", "hum_gpt_recall"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(2)

    if "n_eval" in df.columns:
        df = df.drop(columns=["n_eval"])

    col_order = [c for c in [
        "construct", "freq_in_data", "hum_gpt_kappa",
        "hum_gpt_precision", "hum_gpt_recall"
    ] if c in df.columns]
    return df[col_order] if col_order else df


def dataframe_to_png_table(
    df: pd.DataFrame,
    out_path: str,
    title: str = None,
    construct_wrap_width: int = 16,
    base_fontsize: int = 9
):
    df = df.copy()

    if "construct" in df.columns:
        df["construct"] = df["construct"].astype(str).map(
            lambda s: fill(s, width=construct_wrap_width, break_long_words=False)
        )

    n_rows, n_cols = df.shape

    fig_w = max(9, 1.5 * n_cols + 2)
    fig_h = max(3.5, 0.6 * (n_rows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, pad=16)

    if "construct" in df.columns:
        colWidths = [0.55] + [0.18] * (n_cols - 1)
    else:
        colWidths = [0.2] * n_cols

    tbl = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        cellLoc="left",
        colLoc="left",
        colWidths=colWidths,
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(base_fontsize)
    tbl.scale(1.15, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")

    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_markdown(df: pd.DataFrame, out_path: str):
    try:
        md = df.to_markdown(index=False)
        Path(out_path).write_text(md, encoding="utf-8")
    except Exception as e:
        headers = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(map(str, row.values)) + " |")
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def write_latex(df: pd.DataFrame, out_path: str):
    tex = df.to_latex(index=False, escape=True, longtable=True)
    Path(out_path).write_text(tex, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Convert CSV/JSON to table figure + Markdown/LaTeX.")
    ap.add_argument("input_path", help="Path to .csv or .json")
    ap.add_argument("--png", default=None, help="Output PNG path (default: input stem + .png)")
    ap.add_argument("--md",  default=None, help="Output Markdown path (default: input stem + .md)")
    ap.add_argument("--tex", default=None, help="Output LaTeX path (default: input stem + .tex)")
    ap.add_argument("--title", default=None, help="Optional title for the figure")
    ap.add_argument("--wrap", type=int, default=16, help="Wrap width for 'construct' column")
    ap.add_argument("--fontsize", type=int, default=9, help="Base font size for table")
    args = ap.parse_args()

    df = load_to_df(args.input_path)
    df = pretty_format(df)

    stem = str(Path(args.input_path).with_suffix(""))
    png_path = args.png or f"{stem}.png"
    md_path  = args.md  or f"{stem}.md"
    tex_path = args.tex or f"{stem}.tex"

    dataframe_to_png_table(
        df, png_path, title=args.title,
        construct_wrap_width=args.wrap,
        base_fontsize=args.fontsize
    )
    write_markdown(df, md_path)
    write_latex(df, tex_path)

    print("Saved:")
    print("  PNG:", png_path)
    print("  MD: ", md_path)
    print("  TeX:", tex_path)


if __name__ == "__main__":
    main()
