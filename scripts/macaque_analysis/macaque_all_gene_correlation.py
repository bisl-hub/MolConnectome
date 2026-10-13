"""Macaque, optional: cross-region correlation sums over all genes shared by the 15 regions.

Input:  macaque expression and sample-matching table.
Output: macaque_analysis_results/whole_transcriptome/regional_correlation_matrix_*.csv
Usage:  see MACAQUE_PIPELINE.md, section 8.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys
import itertools
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
# sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

import src.correlation as cor
import src.quantile as qu
from src.macaque_expression_load import load_macaque_log2_cpm, load_macaque_sample_matching
from src.project_variables import MACAQUE_REGIONS


DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 
    SETTINGS['analysis_results']['mq_analysis_save_dir'])
ALL_GENE_CORR_DIR = os.path.join(DEFAULT_OUTPUT_DIR,
    SETTINGS['macaque_all_gene_correlation']['all_gene_result_save_dir'])

def common_genes(_exprs: dict[str, pd.DataFrame]) -> set[str]:
    """Genes present in every macaque region's expression table."""
    gene_sets = (set(df.index) for df in _exprs.values())
    return set.intersection(*gene_sets)


def all_count_and_sum(r_block: np.ndarray, p_block: np.ndarray):
    mask = (r_block)
    count = mask.sum(axis=(-2, -1))
    value = np.where(mask, r_block, 0.0).sum(axis=(-2, -1))
    return count, value


def negative_count_and_sum(r_block: np.ndarray, p_block: np.ndarray):
    mask = (r_block < 0.0)
    count = mask.sum(axis=(-2, -1))
    value = np.where(mask, r_block, 0.0).sum(axis=(-2, -1))
    return count, value


def only_significants_count_and_sum(r_block: np.ndarray, p_block: np.ndarray):
    mask = (p_block < 0.05)
    count = mask.sum(axis=(-2, -1))
    value = np.where(mask, r_block, 0.0).sum(axis=(-2, -1))
    return count, value


def negative_significants_count_and_sum(r_block: np.ndarray, p_block: np.ndarray):
    mask = (p_block < 0.05) & (r_block < 0.0)
    count = mask.sum(axis=(-2, -1))
    value = np.where(mask, r_block, 0.0).sum(axis=(-2, -1))
    return count, value


_worker_state = {}


def _init_worker() -> None:
    monkey_exprs = load_macaque_log2_cpm(filtered=True)
    sample_matching = load_macaque_sample_matching()
    macaque_common_genes = common_genes(monkey_exprs)
    _worker_state.update(
        monkey_exprs=monkey_exprs,
        sample_matching=sample_matching,
        macaque_common_genes=macaque_common_genes,
    )


def _process_region_pair(pair_index: int, region_a: str, region_b: str) -> list[dict]:
    monkey_exprs = _worker_state['monkey_exprs']
    sample_matching = _worker_state['sample_matching']
    macaque_common_genes = _worker_state['macaque_common_genes']
    count_and_sum_fn = qu._non_filtered_positive_count_and_sum
    sample_ids_a, sample_ids_b = cor.matched_sample_ids(monkey_exprs, sample_matching, region_a, region_b)
    
    def correlate(gene_list):
        X = cor.panel_matrix(monkey_exprs, region_a, gene_list, sample_ids_a)
        Y = cor.panel_matrix(monkey_exprs, region_b, gene_list, sample_ids_b)
        r, p, _ = cor.cross_correlation_and_pvalues(X, Y)
        return r, p

    r_ab, p_ab = correlate(macaque_common_genes)
    r_ba, p_ba = r_ab.T, p_ab.T  # Pearson r (and its p-value) is symmetric -- free, exact
    
    # positive sum
    ab_count, ab_value = count_and_sum_fn(r_ab, p_ab)
    ba_count, ba_value = count_and_sum_fn(r_ba, p_ba)
    ab_row = (region_a, region_b, 'positive_sum', ab_value, ab_count)
    ba_row = (region_b, region_a, 'positive_sum', ba_value, ba_count)

    # negative sum
    ab_neg_count, ab_neg_value = negative_count_and_sum(r_ab, p_ab)
    ba_neg_count, ba_neg_value = negative_count_and_sum(r_ba, p_ba)
    ab_neg_row = (region_a, region_b, 'negative_sum', ab_neg_value, ab_neg_count)
    ba_neg_row = (region_b, region_a, 'negative_sum', ba_neg_value, ba_neg_count)
    
    # all sum   
    ab_all_count, ab_all_value = all_count_and_sum(r_ab, p_ab)
    ba_all_count, ba_all_value = all_count_and_sum(r_ba, p_ba)
    ab_all_row = (region_a, region_b, 'all_sum', ab_all_value, ab_all_count)
    ba_all_row = (region_b, region_a, 'all_sum', ba_all_value, ba_all_count)

    # positive significant p-values
    ab_positive_significant_count, ab_positive_significant_value = qu._qualifying_count_and_sum(r_ab, p_ab)
    ba_positive_significant_count, ba_positive_significant_value = qu._qualifying_count_and_sum(r_ba, p_ba)
    ab_positive_significant_row = (region_a, region_b, 'positive_significant', ab_positive_significant_value, ab_positive_significant_count)
    ba_positive_significant_row = (region_b, region_a, 'positive_significant', ba_positive_significant_value, ba_positive_significant_count)

    # negative significant p-values
    ab_negative_significant_count, ab_negative_significant_value = negative_significants_count_and_sum(r_ab, p_ab)
    ba_negative_significant_count, ba_negative_significant_value = negative_significants_count_and_sum(r_ba, p_ba)
    ab_negative_significant_row = (region_a, region_b, 'negative_significant', ab_negative_significant_value, ab_negative_significant_count)
    ba_negative_significant_row = (region_b, region_a, 'negative_significant', ba_negative_significant_value, ba_negative_significant_count)

    # all significant p-values
    ab_significant_count, ab_significant_value = only_significants_count_and_sum(r_ab, p_ab)
    ba_significant_count, ba_significant_value = only_significants_count_and_sum(r_ba, p_ba)
    ab_significant_row = (region_a, region_b, 'significant', ab_significant_value, ab_significant_count)
    ba_significant_row = (region_b, region_a, 'significant', ba_significant_value, ba_significant_count)


    return [ab_row, ba_row,
            ab_neg_row, ba_neg_row,
            ab_all_row, ba_all_row,
            ab_positive_significant_row, ba_positive_significant_row,
            ab_negative_significant_row, ba_negative_significant_row,
            ab_significant_row, ba_significant_row]


def metric_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """region_a x region_b matrix of one metric, in MACAQUE_REGIONS order.
    pivot() sorts both axes alphabetically, so reorder by label (reindex) --
    assigning MACAQUE_REGIONS to .index/.columns positionally mislabels them."""
    matrix = df[df['metric'] == metric].pivot(index='region_a', columns='region_b', values='value')
    return matrix.reindex(index=MACAQUE_REGIONS, columns=MACAQUE_REGIONS).rename_axis(index=None, columns=None)


def main(n_workers: int = 2,):

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 2)

    region_pairs = list(enumerate(itertools.combinations(MACAQUE_REGIONS, 2)))
    
    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker) as executor:
        futures = {
            executor.submit(_process_region_pair, pair_index, region_a, region_b): (region_a, region_b)
            for pair_index, (region_a, region_b) in region_pairs
        }
        for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            region_a, region_b = futures[future]
            rows = future.result()
            all_rows.extend(rows)
            print(f'[{done_count}/{len(region_pairs)}] {region_a}-{region_b}: {len(rows)} rows')


    df = pd.DataFrame(all_rows, columns=['region_a', 'region_b', 'metric', 'value', 'count'])

    df_pos_r_pos_p = metric_matrix(df, 'positive_sum')
    df_neg_r_pos_p = metric_matrix(df, 'negative_sum')
    df_all_r_all_p = metric_matrix(df, 'all_sum')
    df_pos_r_sig_p = metric_matrix(df, 'positive_significant')
    df_neg_r_sig_p = metric_matrix(df, 'negative_significant')
    df_all_r_sig_p = metric_matrix(df, 'significant')

    os.makedirs(ALL_GENE_CORR_DIR, exist_ok=True)

    df_pos_r_pos_p.to_csv(os.path.join(ALL_GENE_CORR_DIR, 'regional_correlation_matrix_positive_sum.csv'), index=True)
    df_neg_r_pos_p.to_csv(os.path.join(ALL_GENE_CORR_DIR, 'regional_correlation_matrix_negative_sum.csv'), index=True)
    df_all_r_all_p.to_csv(os.path.join(ALL_GENE_CORR_DIR, 'regional_correlation_matrix_all_sum.csv'), index=True)
    df_pos_r_sig_p.to_csv(os.path.join(ALL_GENE_CORR_DIR, 'regional_correlation_matrix_positive_significant.csv'), index=True)
    df_neg_r_sig_p.to_csv(os.path.join(ALL_GENE_CORR_DIR, 'regional_correlation_matrix_negative_significant.csv'), index=True)
    df_all_r_sig_p.to_csv(os.path.join(ALL_GENE_CORR_DIR, 'regional_correlation_matrix_significant.csv'), index=True)
    
    return df_pos_r_pos_p, df_neg_r_pos_p, df_all_r_all_p, df_pos_r_sig_p, df_neg_r_sig_p, df_all_r_sig_p


if __name__ == '__main__':
    main(n_workers = 2)
    # on 48-GB machine, running via more than 2 explodes your RAM.