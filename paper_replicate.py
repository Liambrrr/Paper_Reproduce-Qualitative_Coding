#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import warnings
import math
import difflib

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "openai package not installed."
    ) from e

def build_codebook():
    return {
        "If Header": "Modifications to the if condition/header.",
        "If Body": "Modifications to the lines enabled by the if condition/header.",
        "Function Return": "Modifications inside the return statement.",
        "Function Body": "Modifications inside the body of the function. These modifications include adding more conditional statements, auxiliary variables, and others.",
        "Comment": "A new commented line or a deletion or modification of an already existing comment.",
        "Testing": "Modifications inside the Main function (section of the code used for testing), such as adding a line to print results in the console and testing the correct functioning of their code.",
        "Added Lines": "Contains at least one completely new code line.",
        "Removed Lines": "Student removed code lines in the submission.",
        "Variable Usage & Assignment": "Student submission adds a new variable or deletes or modifies the value assignment of an already existing variable.",
        "Variable-type Change": "A modification of the type of variable on its initial declaration.",
        "Variable-type Conversion Change": "Modification in the conversion of the type of variable after its initial declaration or a conversion in the type of variable obtained after using an already existing method.",
        "Value Change": "Modification of any value. It can be in the if header/condition, in the coefficient in an equation, or in the assignment of a variable.",
        "Operator": "Modification of an operator, such as changing the “greater than” operator to “equal to” in a conditional statement.",
        "Syntax Change": "A modification in the syntax of a code line to correct a compiler error.",
    }

ZERO_SHOT_TEMPLATE = (
    "Please review the provided text and code it based on the construct: {construct}. "
    "The definition of this construct is: {definition} "
    "After reviewing the text, assign a code of '1' if you believe the text exemplifies {construct}, "
    "or a '0' if it does not. Your response should only be '1' or '0'.\n\n"
    "TEXT:\n{snippet}\n"
    "Answer:"
)

# helpers
def canonical(s: str) -> str:
    """
    Canonicalize construct/column names so small variants match:
    '&' vs 'and', hyphens, case, punctuation, extra spaces, etc.
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()

def map_construct_to_coded_column(construct_value: str, coded_construct_cols: list[str]) -> str | None:
    """
    Returns the column name in coded data that best corresponds to `construct_value`.
    Uses canonical exact match, then fuzzy match.
    """
    can_target = canonical(construct_value)
    if not can_target:
        return None

    canon_map = {canonical(c): c for c in coded_construct_cols}
    if can_target in canon_map:
        return canon_map[can_target]

    choices = list(canon_map.keys())
    matches = difflib.get_close_matches(can_target, choices, n=1, cutoff=0.72)
    if matches:
        return canon_map[matches[0]]
    return None

ID_INT_TOL = 1e-9
def _normalize_id_value(v):
    """
    Return a canonical string ID for merge:
    - numeric near-int -> int string (e.g., 12345.0 -> "12345")
    - numeric non-int -> keep as string without losing decimals unless it ends with .0+
    - otherwise -> stripped string (also collapses '12345.0' -> '12345')
    """
    if pd.isna(v):
        return np.nan
    try:
        f = float(v)
        if math.isfinite(f):
            r = round(f)
            if abs(f - r) < ID_INT_TOL:
                return str(int(r))
            s = str(v).strip()
            if re.fullmatch(r"\d+(\.0+)?", s):
                return s.split(".")[0]
            return s
    except Exception:
        pass
    s = str(v).strip()
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m:
        return m.group(1)
    return s

def make_id_key(series: pd.Series) -> pd.Series:
    """Vectorized normalization to canonical string IDs."""
    return series.map(_normalize_id_value)

def coerce_raw_columns(df: pd.DataFrame, id_col: str | None, text_col: str, construct_col: str) -> pd.DataFrame:
    """
    Make RAW Excel tolerant to missing/unexpected headers and align to expected names.
    """
    cols = list(df.columns)

    # If exactly 2 columns and id provided -> assume [id, text]
    if len(cols) == 2 and (id_col is not None) and (text_col not in cols):
        df = df.copy()
        df.columns = [id_col, text_col]
        return df

    # Ensure TEXT present
    if text_col not in df.columns:
        obj_cols = [c for c in cols if df[c].dtype == "object"]
        if obj_cols:
            def avg_len(s):
                try:
                    return s.dropna().astype(str).str.len().mean()
                except Exception:
                    return -1
            best = max(obj_cols, key=lambda c: avg_len(df[c]))
            df = df.rename(columns={best: text_col})
        else:
            df = df.rename(columns={cols[-1]: text_col})

    # If construct column is missing but a close name exists, adopt it
    if construct_col not in df.columns:
        match = difflib.get_close_matches(construct_col, cols, n=1, cutoff=0.8)
        if match:
            df = df.rename(columns={match[0]: construct_col})

    # Ensure ID if requested
    if id_col and (id_col not in df.columns):
        num_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
        if num_cols:
            df = df.rename(columns={num_cols[0]: id_col})
        else:
            df = df.rename(columns={cols[0]: id_col})

    return df

def melt_coded_wide_to_long(coded_df: pd.DataFrame, id_col: str | None, construct_col_name: str) -> pd.DataFrame:
    """
    Convert wide coded data (one column per construct) to long form:
    columns: [id_col?], construct_col_name, 'human_label'
    """
    df = coded_df.copy()

    # Choose id column(s) to carry through
    if id_col and id_col in df.columns:
        id_vars = [id_col]
    elif "task_submit_id" in df.columns:
        id_vars = ["task_submit_id"]
    else:
        id_vars = [df.columns[0]]  # fallback best-effort

    # All other columns are likely constructs
    value_vars = [c for c in df.columns if c not in id_vars]

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=construct_col_name,
        value_name="human_label",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        long_df["human_label"] = pd.to_numeric(long_df["human_label"], errors="coerce")

    # Canonical construct for merge
    long_df["_construct_canon"] = long_df[construct_col_name].astype(str).map(canonical)
    return long_df

def get_construct_list_from_coded(coded_df: pd.DataFrame, id_col: str | None) -> list[str]:
    """
    From a 'wide' coded CSV, return the list of construct column names
    (all columns except the ID column if present).
    """
    cols = list(coded_df.columns)
    if id_col and id_col in coded_df.columns:
        return [c for c in cols if c != id_col]
    if "task_submit_id" in coded_df.columns:
        return [c for c in cols if c != "task_submit_id"]
    return cols

def expand_raw_by_constructs(raw_df: pd.DataFrame, constructs: list[str],
                             id_col: str, text_col: str, construct_col: str) -> pd.DataFrame:
    """
    Cartesian-expand RAW so each row is duplicated once per construct.
    RAW must contain id_col and text_col. Returns a new DataFrame with construct_col added.
    """
    base = raw_df[[id_col, text_col]].copy()
    cdf = pd.DataFrame({construct_col: constructs})
    base["_key"] = 1
    cdf["_key"] = 1
    expanded = base.merge(cdf, on="_key").drop(columns="_key")
    return expanded

def load_client(model_name: str, timeout: float = 60.0):
    client = OpenAI(timeout=timeout)
    return client, model_name

def llm_predict_binary(client, model: str, prompt: str, max_new_tokens: int = 4):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_new_tokens,
    )
    txt = resp.choices[0].message.content or ""
    m = re.search(r"\b([01])\b", txt)
    if m:
        return int(m.group(1))
    m2 = re.search(r"^[\s\n\r]*([01])", txt)
    if m2:
        return int(m2.group(1))
    warnings.warn(f"Could not parse binary label from model output: {txt!r}. Defaulting to 0.")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Zero-shot replication with OpenAI GPT-4.x (Responses API).")
    parser.add_argument("--raw_path", required=True, help="dataset/Raw_Data.xlsx")
    parser.add_argument("--coded_path", required=True, help="dataset/Coded_Data.csv")
    parser.add_argument("--sheet_name", default=0, help="Excel sheet name or index for raw data")
    parser.add_argument("--id_col", default=None, help="ID column present in BOTH files for merging (e.g., task_submit_id)")
    parser.add_argument("--text_col", default="code_change_text", help="Column with the snippet/text to label")
    parser.add_argument("--construct_col", default="construct_name", help="Column with construct name (if RAW lacks it, we expand)")
    parser.add_argument("--label_col", default=None, help="(Rare) if coded CSV already has a single label column")
    parser.add_argument("--no_merge", action="store_true", help="(kept for compatibility; not used)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model id (e.g., gpt-4o, gpt-4.1, gpt-4.1-mini)")
    parser.add_argument("--out_prefix", default="study3_zeroshot", help="Prefix for outputs")
    args = parser.parse_args()

    raw_df = pd.read_excel(args.raw_path, sheet_name=args.sheet_name, header=0)
    coded_df = pd.read_csv(args.coded_path)

    # 2) Make RAW robust
    raw_df = coerce_raw_columns(raw_df, args.id_col, args.text_col, args.construct_col)
    if args.text_col not in raw_df.columns:
        raise KeyError(f"Could not find a text column '{args.text_col}' in RAW after coercion. Got: {list(raw_df.columns)}")

    # 3) If RAW lacks construct names, expand by constructs found in CODED
    if args.construct_col not in raw_df.columns:
        if not args.id_col or args.id_col not in raw_df.columns:
            raise KeyError(
                f"RAW is missing construct column '{args.construct_col}' and also lacks a usable ID column "
                f"('{args.id_col}'). To expand RAW by constructs, RAW must contain an ID (e.g., task_submit_id). "
                f"RAW columns: {list(raw_df.columns)}"
            )
        constructs_from_coded = get_construct_list_from_coded(coded_df, args.id_col)
        raw_df = expand_raw_by_constructs(raw_df, constructs_from_coded, args.id_col, args.text_col, args.construct_col)

    # Canonical construct for merge
    raw_df["_construct_canon"] = raw_df[args.construct_col].map(canonical)

    # 4) Prepare CODED labels (wide -> long is the normal case)
    if args.label_col and args.label_col in coded_df.columns:
        coded_long = coded_df.copy()
        if "human_label" not in coded_long.columns:
            coded_long = coded_long.rename(columns={args.label_col: "human_label"})
        if args.construct_col in coded_long.columns:
            coded_long["_construct_canon"] = coded_long[args.construct_col].map(canonical)
        else:
            warnings.warn(f"'{args.construct_col}' not found in CODED; labels will be merged by ID only.")
    else:
        coded_long = melt_coded_wide_to_long(coded_df, args.id_col, args.construct_col)

    # 5) Build canonical ID keys (normalize float/int/string)
    has_raw_id = args.id_col and (args.id_col in raw_df.columns)
    has_coded_id = args.id_col and (args.id_col in coded_long.columns)
    if has_raw_id:
        raw_df["_id_key"] = make_id_key(raw_df[args.id_col])
    if has_coded_id:
        coded_long["_id_key"] = make_id_key(coded_long[args.id_col])

    # 6) Merge preference: (ID + construct), else construct-only
    if has_raw_id and has_coded_id:
        merge_keys = ["_id_key", "_construct_canon"]
    else:
        warnings.warn(
            f"ID column '{args.id_col}' missing on one side; falling back to construct-only merge. "
            "Assumes labels are per-construct globally."
        )
        merge_keys = ["_construct_canon"]

    if "_construct_canon" not in coded_long.columns and args.construct_col in coded_long.columns:
        coded_long["_construct_canon"] = coded_long[args.construct_col].map(canonical)

    merged = pd.merge(
        raw_df,
        coded_long[merge_keys + ["human_label"]],
        on=merge_keys,
        how="left",
    )

    # 7) Load OpenAI model
    codebook = build_codebook()
    print(f"Loading OpenAI model: {args.model}", file=sys.stderr)
    client, model_name = load_client(args.model)

    # 8) Iterate and predict
    preds = []
    for _, row in merged.iterrows():
        construct_raw = str(row[args.construct_col])
        snippet = str(row[args.text_col])

        cb_keys = list(codebook.keys())
        best_cb_key = map_construct_to_coded_column(construct_raw, cb_keys) or construct_raw
        definition = codebook.get(
            best_cb_key,
            "No definition available for this construct. Decide as best as you can."
        )

        prompt = ZERO_SHOT_TEMPLATE.format(
            construct=construct_raw,
            definition=definition,
            snippet=snippet
        )
        yhat = llm_predict_binary(client, model_name, prompt)
        preds.append(int(yhat))

    preds = np.array(preds, dtype=np.int32)

    # 9) Attach predictions to the merged frame and save a full copy
    merged["gpt_pred"] = preds
    merged_path = f"{args.out_prefix}_merged_with_preds.csv"
    merged.to_csv(merged_path, index=False)

    # 10) Evaluate metrics
    # Overall (same as before)
    valid_mask = ~pd.isna(merged["human_label"])
    if valid_mask.sum() == 0:
        print("No human labels found for evaluation after merge (check ID/construct alignment).", file=sys.stderr)
        overall_kappa = None
    else:
        y_true_all = merged.loc[valid_mask, "human_label"].astype(int).values
        y_pred_all = merged.loc[valid_mask, "gpt_pred"].astype(int).values
        overall_kappa = float(cohen_kappa_score(y_true_all, y_pred_all))

    # ---- Per-construct table ----
    # For each construct, compute:
    # - frequency in data: proportion of human_label == 1 among evaluated rows
    # - Hum–GPT κ
    # - Hum–GPT precision
    # - Hum–GPT recall
    rows = []
    for construct, g in merged.groupby(args.construct_col):
        mask = ~pd.isna(g["human_label"])
        n_eval = int(mask.sum())
        if n_eval == 0:
            rows.append({
                "construct": construct,
                "freq_in_data": None,
                "hum_gpt_kappa": None,
                "hum_gpt_precision": None,
                "hum_gpt_recall": None,
                "n_eval": 0
            })
            continue

        y_true = g.loc[mask, "human_label"].astype(int).values
        y_pred = g.loc[mask, "gpt_pred"].astype(int).values

        # frequency = share of human positives (how often the construct occurs)
        freq = float((y_true == 1).mean())

        # kappa can be undefined in degenerate cases; guard it
        try:
            kappa_c = float(cohen_kappa_score(y_true, y_pred))
        except Exception:
            kappa_c = None

        # precision / recall (binary, 1 is positive)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, zero_division=0))

        rows.append({
            "construct": construct,
            "freq_in_data": freq,
            "hum_gpt_kappa": kappa_c,
            "hum_gpt_precision": prec,
            "hum_gpt_recall": rec,
            "n_eval": n_eval
        })

    per_construct_df = pd.DataFrame(rows).sort_values("construct").reset_index(drop=True)

    # 11) Save artifacts
    npy_path = f"{args.out_prefix}_preds.npy"
    csv_path = f"{args.out_prefix}_preds.csv"
    metrics_path = f"{args.out_prefix}_metrics.json"
    per_construct_csv = f"{args.out_prefix}_per_construct_metrics.csv"
    per_construct_json = f"{args.out_prefix}_per_construct_metrics.json"

    # raw predictions (unchanged)
    np.save(npy_path, preds)
    pd.DataFrame({
        "row_index": np.arange(len(preds)),
        "pred": preds
    }).to_csv(csv_path, index=False)

    # per-construct table
    per_construct_df.to_csv(per_construct_csv, index=False)
    with open(per_construct_json, "w") as f:
        json.dump(per_construct_df.to_dict(orient="records"), f, indent=2)

    # overall metrics JSON (kept; now also includes paths to new files)
    metrics = {
        "model": model_name,
        "overall_kappa": overall_kappa,
        "n_rows": int(len(merged)),
        "n_eval_overall": int(valid_mask.sum()),
        "merge_keys": merge_keys,
        "columns": {
            "text_col": args.text_col,
            "construct_col": args.construct_col,
            "id_col": args.id_col,
        },
        "artifacts": {
            "merged_with_preds_csv": merged_path,
            "preds_npy": npy_path,
            "preds_csv": csv_path,
            "per_construct_csv": per_construct_csv,
            "per_construct_json": per_construct_json
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 12) Console summary
    print(f"Saved predictions to {npy_path} and {csv_path}")
    print(f"Saved merged data with predictions to {merged_path}")
    print(f"Saved per-construct metrics to {per_construct_csv} and {per_construct_json}")
    if overall_kappa is not None:
        print(f"Cohen's kappa (overall, hum–model): {overall_kappa:.4f} on {valid_mask.sum()} examples")
    else:
        print("Overall kappa not computed (no human labels matched).")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.warn("OPENAI_API_KEY not set; set it in your environment before running.")
    main()