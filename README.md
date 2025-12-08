# Paper_Reproduce-Qualitative_Coding (Raw Data)

## Overview
Replication results of paper *Qualitative Coding with GPT-4: Where it Works Better*.

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
### Step Instruction
Step 1 - Data requirements

Download the data from https://osf.io/pjk5t/files/osfstorage?view_only=17c6538775c94f1ba93596424af379e8. We’ll use the “Study 3 Raw Data.xlsx” file with the model and the “Study 3 Coded Data.csv” file for calculating the Kappa values.

Step 2 - Run the experiment

Use GPT-4 (if possible gpt-4-turbo-2024-04-09) model with default hyperparameters and temperature = 0. We’ll just replicate zero-shot results, because they didn’t exactly give the few-shot prompts, and in study 3 zero-shot approach outperformed the few-shot approach. So, use the prompt above to reproduce the zero-shot approach results.
Send the prompt as a system message to the Chat Completions endpoint, followed by the specific line of data that GPT should code, send as a user message.

Step 3 - Prepare the prompt payload

System prompt: Include the construct name and its definition from the table above (see Table 8, codebook with definitions).
User content: Pass only the line-level diff between two consecutive submissions (the raw data file is already in this format)

Step 4 - Inference loop

For each data row, iterate over the 14 constructs and call the model once per construct (System = zero-shot prompt with that construct’s definition; User = that row’s diff text). Collect a 0/1 for each construct. (Binary per-construct setup per paper.)

Step 5 - Evaluation (IRR & metrics)

Join predictions to Study 3 Coded Data.csv by the shared key (e.g., task_submit_id) and compute Cohen’s κ, precision, and recall (report per-construct and macro summaries). The paper computes and reports κ, precision, recall for Study 3 (see their Table 9/results). Because outputs can vary, the paper ran the coding three times and averaged κ/precision/recall across runs. But it is not necessary for us.
## Results

### Main Results
*(Metrics averaged across runs 4 and 5 with standard deviations)*

| construct                     |   freq paper |   freq run4 |   freq run5 |   mean freq |   std freq |   kappa paper |   kappa run4 |   kappa run5 |   mean kappa |   std kappa |   prec paper |   prec run4 |   prec run5 |   mean prec |   std prec |   recall paper |   recall run4 |   recall run5 |   mean recall |   std recall |
|-------------------------------|--------------|-------------|-------------|-------------|------------|---------------|--------------|--------------|--------------|-------------|--------------|-------------|-------------|-------------|------------|----------------|---------------|---------------|---------------|--------------|
| added lines                   |        11 |          11 |          11 |          11 |       0.00 |          0.93 |         0.91 |         0.91 |         0.91 |        0.00 |         0.88 |        0.85 |        0.85 |        0.85 |       0.00 |           1.00 |          1.00 |          1.00 |          1.00 |         0.00 |
| comment                       |         2 |           2 |           2 |           2 |       0.00 |          0.80 |         0.34 |         0.38 |         0.36 |        0.03 |         0.67 |        0.22 |        0.25 |        0.24 |       0.02 |           1.00 |          1.00 |          1.00 |          1.00 |         0.00 |
| function body                 |        20 |          18 |          18 |          18 |       0.00 |          0.10 |         0.15 |         0.13 |         0.14 |        0.01 |         0.25 |        0.26 |        0.24 |        0.25 |       0.01 |           0.66 |          0.61 |          0.61 |          0.61 |         0.00 |
| function return               |        22 |          25 |          25 |          25 |       0.00 |          0.54 |         0.65 |         0.69 |         0.67 |        0.03 |         0.90 |        0.84 |        0.85 |        0.85 |       0.01 |           0.48 |          0.64 |          0.68 |          0.66 |         0.03 |
| if body                       |         6 |           9 |           9 |           9 |       0.00 |          0.12 |         0.05 |         0.01 |         0.03 |        0.03 |         0.14 |        0.12 |        0.09 |        0.11 |       0.02 |           0.39 |          0.44 |          0.33 |          0.39 |         0.08 |
| if header                     |        36 |          36 |          36 |          36 |       0.00 |          0.78 |         0.72 |         0.74 |         0.73 |        0.01 |         0.93 |        0.73 |        0.73 |        0.73 |       0.00 |           0.78 |          0.97 |          1.00 |          0.99 |         0.02 |
| operator                      |        30 |          30 |          30 |          30 |       0.00 |          0.73 |         0.70 |         0.72 |         0.71 |        0.01 |         0.75 |        0.76 |        0.78 |        0.77 |       0.01 |           0.90 |          0.83 |          0.83 |          0.83 |         0.00 |
| removed lines                 |         9 |           9 |           9 |           9 |       0.00 |          0.71 |         0.78 |         0.78 |         0.78 |        0.00 |         0.67 |        0.73 |        0.73 |        0.73 |       0.00 |           0.84 |          0.89 |          0.89 |          0.89 |         0.00 |
| syntax change                 |        29 |          35 |          35 |          35 |       0.00 |          0.49 |         0.29 |         0.21 |         0.25 |        0.06 |         0.58 |        0.48 |        0.45 |        0.47 |       0.02 |           0.79 |          0.77 |          0.71 |          0.74 |         0.04 |
| testing                       |        28 |          27 |          27 |          27 |       0.00 |          0.54 |         0.29 |         0.34 |         0.32 |        0.04 |         0.90 |        0.78 |        0.80 |        0.79 |       0.01 |           0.48 |          0.26 |          0.30 |          0.28 |         0.03 |
| value change                  |        10 |          10 |          10 |          10 |       0.00 |          0.40 |         0.13 |         0.12 |         0.12 |        0.01 |         0.38 |        0.16 |        0.16 |        0.16 |       0.00 |           0.62 |          0.90 |          1.00 |          0.95 |         0.07 |
| variable usage and assignment |        16 |          22 |          22 |          22 |       0.00 |          0.30 |         0.28 |         0.37 |         0.33 |        0.06 |         0.33 |        0.38 |        0.42 |        0.40 |       0.03 |           0.85 |          0.64 |          0.77 |          0.71 |         0.09 |
| variable-type change          |         3 |           1 |           1 |           1 |       0.00 |          0.45 |         0.32 |         0.39 |         0.35 |        0.05 |         0.40 |        0.20 |        0.25 |        0.23 |       0.04 |           0.80 |          1.00 |          1.00 |          1.00 |         0.00 |
| variable-type conversion      |         9 |          12 |          12 |          12 |       0.00 |          0.57 |         0.35 |         0.29 |         0.32 |        0.04 |         0.47 |        0.31 |        0.48 |        0.40 |       0.12 |           0.96 |          1.00 |          0.92 |          0.96 |         0.06 |

### Significance Tests (p-values)

*(Two-tailed t-tests comparing paper results vs reproduced runs)*

| metric        |   p_value |
|---------------|-----------|
| freq p run4   |    0.1276 |
| freq p run5   |    0.1276 |
| kappa p run4  |    0.0242 |
| kappa p run5  |    0.0468 |
| prec p run4   |    0.0157 |
| prec p run5   |    0.0369 |
| recall p run4 |    0.4758 |
| recall p run5 |    0.4161 |

## Reference
Liu, X., Zambrano, A. F., Baker, R. S., Barany, A., Ocumpaugh, J., Zhang, J., Pankiewicz, M., Nasiar, N., & Wei, Z. (2025). Qualitative Coding with GPT-4: Where it Works Better. Journal of Learning Analytics, 12(1), 169-185. https://doi.org/10.18608/jla.2025.8575