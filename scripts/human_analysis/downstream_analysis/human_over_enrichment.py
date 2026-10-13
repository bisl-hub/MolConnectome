"""Human stage 6: GO over-representation of the driver genes of each direction-consistent connection.

Input:  stage 5's driver_gene_prioritization.csv and P2's GO_baseline/; needs internet (PANTHER).
Output: human_analysis_results/go_enrichment/<JOB>/<NT>_<sender>_<receiver>.xlsx
Usage:  see HUMAN_PIPELINE.md, section 4.
"""

import ast
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys

import numpy as np
import pandas as pd

from pathlib import Path
import yaml
with open(Path(__file__).resolve().parent.parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

import src.go_enrichment_analysis as gomodule
from src.project_variables import GO_TERMS, NT_TO_GENE_SET

GO_ENRICHMENT_DIR = os.path.join(PROJECT_ROOT,
    SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_over_enrichment']['report_save_dir'])
BASELINE_SAVE = os.path.join(GO_ENRICHMENT_DIR, SETTINGS['human_over_enrichment']['baseline_save_dir'])
# argv[1] defaults to settings.yaml's label-shuffle job; a folder or csv path also works.
DEFAULT_JOB_NAME = SETTINGS['human_label_shuffle_wrapper']['initial_job_name']
# human_prioritize_drivers.py writes its csv into label_shuffle_summary/<JOB_NAME>/
LABEL_SHUFFLE_SUMMARY_DIR = os.path.join(PROJECT_ROOT,
    SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_delta_quantile_report']['report_save_dir'])
DRIVER_GENE_CSV_FILENAME = SETTINGS['human_over_enrichment']['driver_gene_csv']
# cat1 = receiver, cat2 = sender (CURATED_GENE_SET.pickle convention; see src/quantile.py)
PRIORITIZED_COLUMNS = {'cat1': 'receiver_prioritized', 'cat2': 'sender_prioritized'}


def load_driver_genes(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['connection_id'] = df['neurotransmitter'] + '_' + df['sender_region'] + '_' + df['receiver_region']
    for col in PRIORITIZED_COLUMNS.values():
        df[col] = df[col].apply(ast.literal_eval)

    unmapped = sorted(set(df['neurotransmitter']) - set(NT_TO_GENE_SET))
    if unmapped:
        raise ValueError(f'No CURATED_GENE_SET gene set for neurotransmitter(s): {unmapped}')
    return df


def load_baseline_enrichment_dataframe(gene_set_id: str, category_name: str, go_term: str) -> pd.DataFrame:
    """baseline_GO_enrichment.py saves each real result with go_term_id as
    the row index (not a named column) -- so on disk it round-trips as an
    unnamed first column; renamed back here rather than at save time, since
    all 54 gene set x cat1/cat2 x GO_TERMS baseline files are written in
    this format."""
    baseline_path = os.path.join(BASELINE_SAVE, gene_set_id, f'{category_name}_baseline_{go_term}.csv')
    baseline_df = pd.read_csv(baseline_path)
    if 'fold_enrichment' not in baseline_df.columns:  # placeholder csv (empty gene list / empty enrichment), not a real result
        raise ValueError(f'Baseline enrichment result empty: {baseline_path}')
    if 'go_term_id' not in baseline_df.columns:
        baseline_df = baseline_df.rename(columns={baseline_df.columns[0]: 'go_term_id'})
    return baseline_df


def get_overenriched_terms(custom_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """GO terms common to both enrichment results where the prioritized
    (custom) subset scores MORE enriched than the gene set's own baseline --
    lower FDR-corrected p-value and/or higher fold enrichment. Duplicate
    go_term_ids are dropped up front."""
    custom = (custom_df[['go_term_id', 'fold_enrichment', 'fdr_corrected_pval']]
              .drop_duplicates('go_term_id').set_index('go_term_id'))
    baseline = (baseline_df[['go_term_id', 'fold_enrichment', 'fdr_corrected_pval']]
                .drop_duplicates('go_term_id').set_index('go_term_id'))
    common_terms = custom.index.intersection(baseline.index)
    custom, baseline = custom.loc[common_terms], baseline.loc[common_terms]

    lower_fdr = custom['fdr_corrected_pval'] < baseline['fdr_corrected_pval']
    higher_fold = custom['fold_enrichment'] > baseline['fold_enrichment']
    print(f'{higher_fold.sum()} terms were overenriched in fold enrichment and '
          f'{lower_fdr.sum()} terms were overenriched in fdr corrected p-value')

    overenriched_ids = common_terms[lower_fdr | higher_fold]
    return pd.concat([
        custom_df.set_index('go_term_id').loc[overenriched_ids],
        baseline.loc[overenriched_ids, ['fold_enrichment', 'fdr_corrected_pval']].add_suffix('_baseline'),
    ], axis=1)


def enrich_and_compare_connection(row: pd.Series) -> 'dict[str, pd.DataFrame]':
    """One sheet-name ('{go_term}#{cat1_or_cat2}') -> result-dataframe per
    category x GO domain for this connection, restricted to overenriched GO
    terms; a sheet holds a one-row placeholder instead if this category's
    gene list, or its own enrichment result, comes up empty."""
    gene_set_id = NT_TO_GENE_SET[row['neurotransmitter']]
    sheets = {}
    for category_name, gene_col in PRIORITIZED_COLUMNS.items():
        gene_symbols = np.asarray(row[gene_col])
        for go_term in GO_TERMS:
            sheet_name = f'{go_term}#{category_name}'

            if gene_symbols.size == 0:
                sheets[sheet_name] = pd.DataFrame.from_dict({
                    0: 'no prioritized genes',
                    1: f'conditions: {gene_set_id};{category_name}',
                }, orient='index')
                continue

            custom_df = gomodule.construct_enrichment_dataframe(gene_symbols, go_term)
            if custom_df.empty:
                sheets[sheet_name] = pd.DataFrame.from_dict({
                    0: 'custom enrichment result empty',
                    1: f'conditions: {gene_set_id};{category_name}',
                }, orient='index')
                continue

            baseline_df = load_baseline_enrichment_dataframe(gene_set_id, category_name, go_term)
            sheets[sheet_name] = get_overenriched_terms(custom_df, baseline_df)
    return sheets


def resolve_driver_gene_csv(target: str) -> str:
    """Accept a driver_gene_prioritization.csv path, a folder that contains
    one, or a bare job name (looked up under label_shuffle_summary/<job>/) --
    the same idea as human_corr_contrib.resolve_flagged_path."""
    if os.path.isfile(target):
        return target
    in_folder = os.path.join(target, DRIVER_GENE_CSV_FILENAME)
    if os.path.isfile(in_folder):
        return in_folder
    return os.path.join(LABEL_SHUFFLE_SUMMARY_DIR, target, DRIVER_GENE_CSV_FILENAME)


def main(target: str = DEFAULT_JOB_NAME, output_dir: str = None) -> None:
    csv_path = resolve_driver_gene_csv(target)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'{csv_path} not found -- run human_prioritize_drivers.py for {target!r} first.')
    # default: go_enrichment/<job>/, the job being the csv's own folder name
    save_dir = output_dir if output_dir is not None else os.path.join(GO_ENRICHMENT_DIR, Path(csv_path).parent.name)
    os.makedirs(save_dir, exist_ok=True)

    driver_genes = load_driver_genes(csv_path)
    if 'is_direction_consistent' not in driver_genes.columns:
        raise ValueError(f'{csv_path} has no is_direction_consistent column -- re-run human_prioritize_drivers.py.')
    driver_genes = driver_genes[driver_genes['is_direction_consistent']]

    for _, row in driver_genes.iterrows():
        print(row['connection_id'])
        sheets = enrich_and_compare_connection(row)
        out_path = os.path.join(save_dir, f"{row['connection_id']}.xlsx")
        with pd.ExcelWriter(out_path, mode='w') as xwriter:
            for sheet_name, sheet_df in sheets.items():
                sheet_df.to_excel(xwriter, sheet_name=sheet_name, merge_cells=False)
    print(f'\nWritten to {save_dir}')


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JOB_NAME,  # job name, folder, or csv path
        sys.argv[2] if len(sys.argv) > 2 else None,  # output_dir
    )
