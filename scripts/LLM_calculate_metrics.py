import pandas as pd
import numpy as np
import os
import sklearn.metrics as metrics
from functools import reduce
from typing import Any

workspace_dir = os.path.dirname(os.path.dirname(__file__)) # your workspace
inputs_dir = os.path.join(workspace_dir, 'test', "inputs") # your inputs
results_dir = os.path.join(workspace_dir, 'test', "LLM_output") # your results

INPUT_PREFIX = "bionli_augmented"
OUTPUT_TAG = "all_predictions"


def load_golden_standard(input_type: str) -> pd.DataFrame:
    if input_type == 'augmented_400':
        golden_standard_path = os.path.join(inputs_dir, f"{INPUT_PREFIX}_400.csv")
    else:
        raise ValueError(f"Invalid input type: {input_type}")
    golden_standard_df = pd.read_csv(golden_standard_path, dtype='str', header=0)
    return golden_standard_df


def load_predicted_results(output_filename: str) -> pd.DataFrame:
    predicted_results_path = os.path.join(results_dir, f"{output_filename}.csv")
    predicted_results_df = pd.read_csv(predicted_results_path, dtype='str', header=0)
    return predicted_results_df

    
def filter_and_retype_essential_columns(all_results_df: pd.DataFrame, 
                                        output_filename: str, 
                                        filter_error:bool = True) -> pd.DataFrame:
    model_name = output_filename.split('_')[0] + '_' + output_filename.split('_')[-1]
    if 'data_index' in all_results_df.columns:
        essential_df = all_results_df[['data_index', 'verdict']].copy();
        essential_df.drop_duplicates('data_index').copy()
        essential_df.columns = ['data_index', f'predicted_{model_name}']
    else:
        if 'pair_id' in all_results_df.columns:
            essential_df = all_results_df[['pair_id', 'verdict']].copy();
            essential_df.drop_duplicates('pair_id').copy()
            essential_df.columns = ['pair_id', f'predicted_{model_name}']
        else:
            essential_df = all_results_df[['pmid', 'verdict']].copy();
            essential_df.drop_duplicates('pmid').copy()
            essential_df.columns = ['pmid', f'predicted_{model_name}']
    if filter_error:
        essential_df = essential_df[essential_df['predicted_'+ model_name] != 'ERROR']
    return essential_df


def report_missing(golden_standard_data: pd.DataFrame, 
                   to_checks: list[Any], key_column: str):    
    for each_prediction in to_checks:
        if each_prediction.shape[0] != golden_standard_data.shape[0]:
            print("Error: some keys are missing, and the results are:")
            gs_keys = set(golden_standard_data[key_column])
            pred_keys = set(each_prediction[key_column])
            print(len(gs_keys - pred_keys), 'keys are present in GS but missing in prediction, and')
            print(len(pred_keys - gs_keys), 'keys are present in predicted but missing in GS.')
        else:
            print("All keys are present.")
    return None
    

def calculate_metrics_using_sklearn(result_data: pd.DataFrame) -> pd.DataFrame:
    labels = ['SUPPORT', 'REJECT', 'NEUTRAL']
    models = [x for x in result_data.columns if x not in ['pair_id', 'pmid', 'golden_standard']]
    model_metrics_list = []
    for model in models:
        valid_data = result_data.dropna(subset=['golden_standard', model])
        true_labels = valid_data['golden_standard'].str.strip().str.upper().to_numpy()
        pred_labels = valid_data[model].str.strip().str.upper().to_numpy()
        model_acc = metrics.accuracy_score(true_labels, pred_labels)
        model_prec = metrics.precision_score(true_labels, pred_labels, average='macro')
        model_rec = metrics.recall_score(true_labels, pred_labels, average='macro')
        model_f1 = metrics.f1_score(true_labels, pred_labels, average='macro')
        model_metrics_list.append({
            'Model': model,
            'Accuracy': model_acc,
            'Precision': model_prec,
            'Recall': model_rec,
            'F1_Score': model_f1
        })
    return pd.DataFrame(model_metrics_list)

   
def report_all_data_metrics(input_data_type: str, output_filename: str, 
                            selected_indices: list = None,
                            report_statistics: bool = False, 
                            filter_errors: bool = True,
                            drop_duplicates: bool = False):
    golden_standard_df = load_golden_standard(input_data_type)
    KEY_COLUMN = 'pmid'

    if drop_duplicates:
        golden_standard_df = golden_standard_df.drop_duplicates(KEY_COLUMN).copy()
    if selected_indices is not None:
        golden_standard_df = golden_standard_df[golden_standard_df[KEY_COLUMN].isin(np.array(selected_indices))].copy()

    GS_essential = golden_standard_df[[KEY_COLUMN, 'gold_label']].copy(); 
    GS_essential.columns = [KEY_COLUMN, 'golden_standard']
    predicted_results_df = load_predicted_results(output_filename)
    print("Total cost spent:", np.sum(predicted_results_df['cost'].to_numpy().astype(float)))
    essential_df = filter_and_retype_essential_columns(
        predicted_results_df, output_filename, filter_error = filter_errors)

    if report_statistics:
        data_length = golden_standard_df.shape[0]
        uniques = np.unique(golden_standard_df[KEY_COLUMN].to_numpy()).shape[0]
        if data_length != uniques:
            print("Error: Not all keys are unique")      
        report_missing(GS_essential, [predicted_results_df,], KEY_COLUMN)
        print('ERROR rows are total', 
              predicted_results_df[predicted_results_df['verdict'] == 'ERROR'].shape[0], 
              'rows.' )

    to_merge = [GS_essential, essential_df]
    to_calculate = reduce(
        lambda left, right: pd.merge(left, right, on=KEY_COLUMN, how='inner'),
        to_merge)
    essential = to_calculate.set_index(KEY_COLUMN)
    if 'pmid_x' in essential.columns:
        essential = essential[[x for x in to_calculate.columns if 'id' not in x]]
    sklearn_metrics_df = calculate_metrics_using_sklearn(essential)
    print("Metrics:")
    print(sklearn_metrics_df)
    data_column = [x for x in essential.columns if x != 'golden_standard']
    for data_col in data_column:
        labels = ['SUPPORT', 'REJECT', 'NEUTRAL']
        contingency_table = pd.crosstab(
            to_calculate['golden_standard'].str.strip().str.upper(),
            to_calculate[data_col].str.strip().str.upper(),
            rownames=['Actual (Golden Standard)'],
            colnames=['Predicted']
        ).reindex(index=labels, columns=labels, fill_value=0)
        print("Contingency table (confusion matrix) for data", data_col, ":")
        print(contingency_table)
    

def compare_results_and_match_ids(input_data_type: str, output_data_names: list[str]):
    filter_errors = True;
    golden_standard_df = load_golden_standard(input_data_type)
    if 'data_index' in golden_standard_df.columns: # 8400 case
        KEY_COLUMN = 'data_index'
    else:  # 400 case
        if 'pair_id' in golden_standard_df.columns: # 8200 and 8700 case
            KEY_COLUMN = 'pair_id'
        else: # 400 case
            KEY_COLUMN = 'pmid'
    golden_standard_df = golden_standard_df.drop_duplicates(KEY_COLUMN).copy()
    outputs_collector = []
    for each_output in output_data_names:
        predicted_results_df = load_predicted_results(each_output)
        essential_df = filter_and_retype_essential_columns(
            predicted_results_df, each_output, filter_error = filter_errors)
        outputs_collector.append(essential_df)
    to_merge = [golden_standard_df] + outputs_collector
    to_calculate = reduce(lambda left, right: pd.merge(left, right, on=KEY_COLUMN, how='inner'),
                          to_merge)
    remained_index = golden_standard_df[golden_standard_df[KEY_COLUMN].isin(to_calculate[KEY_COLUMN])][KEY_COLUMN]
    return remained_index


def main():
    print("For gpt-oss output for 400 selected rows:")
    report_all_data_metrics('augmented_400', 'oss-high-thinking_all_predictions_400', 
                            report_statistics = True, drop_duplicates=True)
    return None


if __name__ == "__main__":
    main()
