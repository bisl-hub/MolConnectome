"""LLM benchmarking: accuracy, macro precision / recall / F1 and the confusion matrix of one
model's predictions against the gold labels.

Input:  llm_benchmarking/inputs/<data_file_name> (gold_label) and the predictions file written by
        run_experiment_oss.py.
Output: printed metrics; the model's row in scripts/llm_benchmarking/<eval_metrics_file_name>
        (tab-separated) is added or updated.
Usage:  python scripts/llm_benchmarking/check_metrics.py [predictions_csv] [gold_csv] [model_label]
        [update_table]  (see LLM_BENCHMARKING.md).
"""

import os
import sys
from pathlib import Path

import pandas as pd
import sklearn.metrics as metrics
import yaml

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'llm_benchmarking'))

from run_experiment_oss import DEFAULT_DATA_FILE, DEFAULT_RESULTS_FILE, KEY_CANDIDATES

LLM_SETTINGS = SETTINGS['llm_benchmarking']
EVAL_METRICS_FILE = os.path.join(PROJECT_ROOT, 'scripts', 'llm_benchmarking',
                                 LLM_SETTINGS['eval_metrics_file_name'])
EVAL_MODEL_LABEL = LLM_SETTINGS['eval_model_label']
LABELS = ['SUPPORT', 'REJECT', 'NEUTRAL']


def filter_and_retype_essential_columns(all_results_df: pd.DataFrame,
                                        key_column: str,
                                        model_name: str,
                                        filter_error: bool = True) -> pd.DataFrame:
    essential_df = all_results_df[[key_column, 'verdict']].copy()
    if filter_error:
        essential_df = essential_df[essential_df['verdict'] != 'ERROR']
    # a pair can have several rows after reruns; the last (most recent) one wins
    essential_df = essential_df.drop_duplicates(key_column, keep='last')
    essential_df.columns = [key_column, f'predicted_{model_name}']
    return essential_df


def report_missing(golden_standard_data: pd.DataFrame,
                   prediction: pd.DataFrame, key_column: str):
    gs_keys = set(golden_standard_data[key_column])
    pred_keys = set(prediction[key_column])
    missing = sorted(gs_keys - pred_keys)
    extra = sorted(pred_keys - gs_keys)
    if not missing and not extra:
        print("All keys are present.")
        return None
    print(f"{len(missing)} keys are present in GS but missing in prediction"
          + (f": {missing}" if 0 < len(missing) <= 10 else ""))
    print(f"{len(extra)} keys are present in prediction but missing in GS"
          + (f": {extra}" if 0 < len(extra) <= 10 else ""))
    return None


def calculate_metrics_using_sklearn(result_data: pd.DataFrame) -> pd.DataFrame:
    models = [x for x in result_data.columns if x != 'golden_standard']
    model_metrics_list = []
    for model in models:
        valid_data = result_data.dropna(subset=['golden_standard', model])
        true_labels = valid_data['golden_standard'].str.strip().str.upper().to_numpy()
        pred_labels = valid_data[model].str.strip().str.upper().to_numpy()
        model_metrics_list.append({
            'Model': model,
            'n_pairs': valid_data.shape[0],
            'Accuracy': metrics.accuracy_score(true_labels, pred_labels),
            'Precision': metrics.precision_score(true_labels, pred_labels, labels=LABELS,
                                                 average='macro', zero_division=0),
            'Recall': metrics.recall_score(true_labels, pred_labels, labels=LABELS,
                                           average='macro', zero_division=0),
            'F1_Score': metrics.f1_score(true_labels, pred_labels, labels=LABELS,
                                         average='macro', zero_division=0),
        })
    return pd.DataFrame(model_metrics_list)


def update_eval_metrics_table(model_metrics: pd.Series, model_label: str,
                              table_path: str = EVAL_METRICS_FILE) -> pd.DataFrame:
    """Add or overwrite `model_label`'s row (Accuracy / Precision / Recall, 4 decimals) in the
    tab-separated summary table. Other rows and the file's line endings are kept."""
    with open(table_path, 'rb') as f:
        lineterminator = '\r\n' if b'\r\n' in f.read() else '\n'
    table = pd.read_csv(table_path, sep='\t', dtype=str)
    values = {col: str(round(float(model_metrics[col]), 4)) for col in ['Accuracy', 'Precision', 'Recall']}
    if (table['Model ID'] == model_label).any():
        for col, value in values.items():
            table.loc[table['Model ID'] == model_label, col] = value
    else:
        table = pd.concat([table, pd.DataFrame([{'Model ID': model_label, **values}])], ignore_index=True)
    table.to_csv(table_path, sep='\t', index=False, lineterminator=lineterminator)
    print(f"Updated {model_label} in {table_path}: {values}")
    return table


def main(predictions_file: str = DEFAULT_RESULTS_FILE,
         gold_file: str = DEFAULT_DATA_FILE,
         model_label: str = EVAL_MODEL_LABEL,
         update_table: bool = True) -> pd.DataFrame:
    golden_standard_df = pd.read_csv(gold_file, dtype='str', header=0)
    key_column = next(c for c in KEY_CANDIDATES if c in golden_standard_df.columns)
    data_length = golden_standard_df.shape[0]
    golden_standard_df = golden_standard_df.drop_duplicates(key_column).copy()
    if golden_standard_df.shape[0] != data_length:
        print("Error: Not all keys are unique")
    GS_essential = golden_standard_df[[key_column, 'gold_label']].copy()
    GS_essential.columns = [key_column, 'golden_standard']

    predicted_results_df = pd.read_csv(predictions_file, dtype='str', header=0)
    print(f"{predictions_file}: {predicted_results_df.shape[0]} rows, "
          f"{(predicted_results_df['verdict'] == 'ERROR').sum()} ERROR rows.")
    essential_df = filter_and_retype_essential_columns(predicted_results_df, key_column, model_label)
    report_missing(GS_essential, essential_df, key_column)

    to_calculate = pd.merge(GS_essential, essential_df, on=key_column, how='inner')
    essential = to_calculate.set_index(key_column)
    sklearn_metrics_df = calculate_metrics_using_sklearn(essential)
    print("Metrics:")
    print(sklearn_metrics_df.to_string(index=False))

    data_col = f'predicted_{model_label}'
    contingency_table = pd.crosstab(
        to_calculate['golden_standard'].str.strip().str.upper(),
        to_calculate[data_col].str.strip().str.upper(),
        rownames=['Actual (Golden Standard)'],
        colnames=['Predicted']
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)
    print("Contingency table (confusion matrix) for", data_col, ":")
    print(contingency_table)
    if 'label_cat' in golden_standard_df.columns:
        by_category = pd.crosstab(
            golden_standard_df.set_index(key_column).loc[to_calculate[key_column], 'label_cat'].to_numpy(),
            to_calculate[data_col].str.strip().str.upper().to_numpy(),
            rownames=['label_cat'],
            colnames=['Predicted']
        ).reindex(columns=LABELS, fill_value=0)
        print("Predictions per label_cat:")
        print(by_category)

    if update_table:
        update_eval_metrics_table(sklearn_metrics_df.iloc[0], model_label)
    return sklearn_metrics_df


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_FILE,
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATA_FILE,
        sys.argv[3] if len(sys.argv) > 3 else EVAL_MODEL_LABEL,
        sys.argv[4] != 'False' if len(sys.argv) > 4 else True,
    )
