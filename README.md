# Paper_Reproduce-Qualitative_Coding

## Overview

### What was implemented

This project partially reproduces **Study 3** of the paper *Qualitative Coding with GPT-4: Where it Works Better* by re-running the **zero-shot qualitative coding** pipeline. Using the authors’ released, de-identified raw code-change data and human-coded labels, the script prompts gpt-4-turbo-2024-04-09 to decide whether each code edit exhibits a specific qualitative construct (e.g., *If Header*, *Syntax Change*). The model’s binary predictions are then compared against the original human annotations using **Cohen’s kappa**.

### Understand the findings

Each execution evaluates **one construct at a time**, matching the experimental setup of Study 3. The primary output is the **human–model Cohen’s kappa** score. Higher kappa values indicate stronger agreement with human coders, while lower values highlight constructs that remain difficult for zero-shot LLMs. By comparing kappa scores across constructs or across different open-source models, we can quickly see which qualitative coding tasks transfer well to LLMs and which still pose challenges without additional examples or context.

## Quick Start
### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Pipeline
```bash
# run paper_replicate_system_prompt.py
export OPENAI_API_KEY="API_KEY"
python paper_replicate_system_prompt.py \
  --raw_path "Raw_Data.xlsx"\
  --coded_path "Coded_Data.csv"\
  --id_col task_submit_id \
  --construct_col construct_name \
  --text_col code_change_text \
  --model gpt-4-turbo-2024-04-09

# convert results to table
python table_convert.py run4/study3_zeroshot_per_construct_metrics_run4.json \ 
  --title "Construct Metrics Run1"
```


## Results
### Frequency 

| construct | freq paper | freq run1 | freq run2 | mean freq | std freq | Δ freq |
|---------|------------|-----------|-----------|-----------|----------|--------|
| added lines | 11 | 11 | 11 | 11 | 0.00 | 0 |
| comment | 2 | 2 | 2 | 2 | 0.00 | 0 |
| function body | 20 | 18 | 18 | 18 | 0.00 | -2 |
| function return | 22 | 25 | 25 | 25 | 0.00 | +3 |
| if body | 6 | 9 | 9 | 9 | 0.00 | +3 |
| if header | 36 | 36 | 36 | 36 | 0.00 | 0 |
| operator | 30 | 30 | 30 | 30 | 0.00 | 0 |
| removed lines | 9 | 9 | 9 | 9 | 0.00 | 0 |
| syntax change | 29 | 35 | 35 | 35 | 0.00 | +6 |
| testing | 28 | 27 | 27 | 27 | 0.00 | -1 |
| value change | 10 | 10 | 10 | 10 | 0.00 | 0 |
| variable usage and assignment | 16 | 22 | 22 | 22 | 0.00 | +6 |
| variable-type change | 3 | 1 | 1 | 1 | 0.00 | -2 |
| variable-type conversion | 9 | 12 | 12 | 12 | 0.00 | +3 |

---

### Cohen’s kappa

| construct | kappa paper | kappa run1 | kappa run2 | mean kappa | std kappa | Δ kappa |
|---------|-------------|------------|------------|------------|-----------|---------|
| added lines | 0.93 | 0.91 | 0.91 | 0.91 | 0.00 | -0.02 |
| comment | 0.80 | 0.34 | 0.38 | 0.36 | 0.03 | -0.44 |
| function body | 0.10 | 0.15 | 0.13 | 0.14 | 0.01 | +0.04 |
| function return | 0.54 | 0.65 | 0.69 | 0.67 | 0.03 | +0.13 |
| if body | 0.12 | 0.05 | 0.01 | 0.03 | 0.03 | -0.09 |
| if header | 0.78 | 0.72 | 0.74 | 0.73 | 0.01 | -0.05 |
| operator | 0.73 | 0.70 | 0.72 | 0.71 | 0.01 | -0.02 |
| removed lines | 0.71 | 0.78 | 0.78 | 0.78 | 0.00 | +0.07 |
| syntax change | 0.49 | 0.29 | 0.21 | 0.25 | 0.06 | -0.24 |
| testing | 0.54 | 0.29 | 0.34 | 0.32 | 0.04 | -0.22 |
| value change | 0.40 | 0.13 | 0.12 | 0.12 | 0.01 | -0.28 |
| variable usage and assignment | 0.30 | 0.28 | 0.37 | 0.33 | 0.06 | +0.03 |
| variable-type change | 0.45 | 0.32 | 0.39 | 0.35 | 0.05 | -0.10 |
| variable-type conversion | 0.57 | 0.35 | 0.29 | 0.32 | 0.04 | -0.25 |

---

### Precision


| construct | prec paper | prec run1 | prec run2 | mean prec | std prec | Δ prec |
|---------|------------|-----------|-----------|-----------|----------|--------|
| added lines | 0.88 | 0.85 | 0.85 | 0.85 | 0.00 | -0.03 |
| comment | 0.67 | 0.22 | 0.25 | 0.24 | 0.02 | -0.43 |
| function body | 0.25 | 0.26 | 0.24 | 0.25 | 0.01 | 0.00 |
| function return | 0.90 | 0.84 | 0.85 | 0.85 | 0.01 | -0.05 |
| if body | 0.14 | 0.12 | 0.09 | 0.11 | 0.02 | -0.03 |
| if header | 0.93 | 0.73 | 0.73 | 0.73 | 0.00 | -0.20 |
| operator | 0.75 | 0.76 | 0.78 | 0.77 | 0.01 | +0.02 |
| removed lines | 0.67 | 0.73 | 0.73 | 0.73 | 0.00 | +0.06 |
| syntax change | 0.58 | 0.48 | 0.45 | 0.47 | 0.02 | -0.11 |
| testing | 0.90 | 0.78 | 0.80 | 0.79 | 0.01 | -0.11 |
| value change | 0.38 | 0.16 | 0.16 | 0.16 | 0.00 | -0.22 |
| variable usage and assignment | 0.33 | 0.38 | 0.42 | 0.40 | 0.03 | +0.07 |
| variable-type change | 0.40 | 0.20 | 0.25 | 0.23 | 0.04 | -0.17 |
| variable-type conversion | 0.47 | 0.31 | 0.48 | 0.40 | 0.12 | -0.07 |

---

### Recall

| construct | recall paper | recall run1 | recall run2 | mean recall | std recall | Δ recall |
|---------|---------------|------------|------------|-------------|------------|----------|
| added lines | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| comment | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| function body | 0.66 | 0.61 | 0.61 | 0.61 | 0.00 | -0.05 |
| function return | 0.48 | 0.64 | 0.68 | 0.66 | 0.03 | +0.18 |
| if body | 0.39 | 0.44 | 0.33 | 0.39 | 0.08 | 0.00 |
| if header | 0.78 | 0.97 | 1.00 | 0.99 | 0.02 | +0.21 |
| operator | 0.90 | 0.83 | 0.83 | 0.83 | 0.00 | -0.07 |
| removed lines | 0.84 | 0.89 | 0.89 | 0.89 | 0.00 | +0.05 |
| syntax change | 0.79 | 0.77 | 0.71 | 0.74 | 0.04 | -0.05 |
| testing | 0.48 | 0.26 | 0.30 | 0.28 | 0.03 | -0.20 |
| value change | 0.62 | 0.90 | 1.00 | 0.95 | 0.07 | +0.33 |
| variable usage and assignment | 0.85 | 0.64 | 0.77 | 0.71 | 0.09 | -0.14 |
| variable-type change | 0.80 | 1.00 | 1.00 | 1.00 | 0.00 | +0.20 |
| variable-type conversion | 0.96 | 1.00 | 0.92 | 0.96 | 0.06 | 0.00 |

### Significance Tests (p-values)

*(Two-tailed t-tests comparing paper results vs reproduced runs)*

| metric        |   p_value |
|---------------|-----------|
| freq p run1   |    0.1276 |
| freq p run2   |    0.1276 |
| kappa p run1  |    0.0242 |
| kappa p run2  |    0.0468 |
| prec p run1   |    0.0157 |
| prec p run2   |    0.0369 |
| recall p run1 |    0.4758 |
| recall p run2 |    0.4161 |

## Reference
Liu, X., Zambrano, A. F., Baker, R. S., Barany, A., Ocumpaugh, J., Zhang, J., Pankiewicz, M., Nasiar, N., & Wei, Z. (2025). Qualitative Coding with GPT-4: Where it Works Better. Journal of Learning Analytics, 12(1), 169-185. https://doi.org/10.18608/jla.2025.8575