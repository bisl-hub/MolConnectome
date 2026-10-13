"""Macaque step 3: step 1's z matrices weighted by sender-region abundance and summed.

Input:  step 1's directed_signaling_strength.csv; abundance is recomputed in memory.
Output: <output_dir>/<background stem>_<p method>/weighted_connectivity.xlsx
Usage:  see MACAQUE_PIPELINE.md, section 4.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys

import pandas as pd
import scipy.stats


from pathlib import Path

import yaml

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

from macaque_abundance import compute_macaque_abundance
from macaque_oneshot import (
    construct_output_path,
    DEFAULT_BACKGROUND_SOURCE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_P_VALUE_METHOD,
)
from src.calculate_abundance import (
    multiply_weight_with_adjacency,
    weight_sum_adjacencies,
)
from src.project_variables import MACAQUE_REGIONS, GENE_SET_TO_USE


STRENGTH_COLUMN = 'mean_z'
N_NULL_SAMPLES = SETTINGS['macaque_oneshot']['quantile_bg_n_null_samples']

# With N_NULL_SAMPLES permutations, rank in [0, N_NULL_SAMPLES], so the
# quantile in macaque_oneshot.py is clamped to [0.5/(N+1), (N+0.5)/(N+1)]
# and z = norm.ppf(quantile) is clamped to [-Z_BOUND, Z_BOUND] (mean_z, being
# an average of such z's, stays within the same bound). This is the exact
# min/max of the range, used below for min-max normalization instead of the
# batch's own empirical min/max.
Z_BOUND = abs(scipy.stats.norm.ppf(0.5 / (N_NULL_SAMPLES + 1))) # lower bound


def normalize_connectivity_strength(z_matrix: 'pd.DataFrame') -> 'pd.DataFrame':
    """Min-max normalize a z-scored adjacency matrix into [0, 1] using the
    theoretical bound Z_BOUND, not the observed min/max of this matrix."""
    return (z_matrix + Z_BOUND) / (Z_BOUND * 2)


def build_gene_set_adjacency(signaling_data: 'pd.DataFrame', gene_set_names: list,
                             regions: list, value_column: str = STRENGTH_COLUMN,
                             normalize: bool = True) -> 'dict[str, pd.DataFrame]':
    """One sender x receiver matrix per gene set, reindexed to `regions` order
    so multiply_weight_with_adjacency can relabel its raw numpy output correctly.
    `normalize` min-max scales into [0, 1] via Z_BOUND -- only valid for a
    theoretically-bounded z column (e.g. mean_z), not for unbounded columns
    like real_value."""
    adjacency_dict = {}
    for gene_set_name in gene_set_names:
        subset = signaling_data.loc[signaling_data['gene_set'] == gene_set_name]
        if subset.empty:
            # A silently all-NaN matrix here poisons every downstream sum it
            # ever gets added into (NaN + anything = NaN). Fail here instead.
            raise ValueError(
                f"gene_set {gene_set_name!r} has no rows in signaling_data -- "
                "it was never computed (name typo, or references a gene set "
                "that was never generated).")
        pivoted = subset.pivot(index='sender_region', columns='receiver_region',
                               values=value_column)
        pivoted = pivoted.reindex(index=regions, columns=regions)
        if normalize:
            pivoted = normalize_connectivity_strength(pivoted)
        adjacency_dict[gene_set_name] = pivoted
    return adjacency_dict


def main(background_source: str = DEFAULT_BACKGROUND_SOURCE,
         output_dir: str = DEFAULT_OUTPUT_DIR,
         p_value_method: str = DEFAULT_P_VALUE_METHOD,
         run_seed: int = None, 
         normalize_weights: bool = True):
    signaling_path = construct_output_path(output_dir, background_source, p_value_method, run_seed)
    results_dir = os.path.dirname(signaling_path)
    output_path = os.path.join(results_dir, 'weighted_connectivity.xlsx')

    mean_abundance, _, _ = compute_macaque_abundance()

    signaling_data = pd.read_csv(signaling_path)
    each_gs_adj_dict = build_gene_set_adjacency(signaling_data, GENE_SET_TO_USE, MACAQUE_REGIONS)

    weighted_by_neurotransmitter = multiply_weight_with_adjacency(
        each_gs_adj_dict, GENE_SET_TO_USE, mean_abundance, 
                                   normalize_weights = normalize_weights)
    weighted_total = weight_sum_adjacencies(
        each_gs_adj_dict, GENE_SET_TO_USE, mean_abundance, organism='macaque', 
                           normalize_weights = normalize_weights)

    with pd.ExcelWriter(output_path) as writer:
        weighted_total.to_excel(writer, sheet_name='total')
        for neurotransmitter_id, adjacency in weighted_by_neurotransmitter.items():
            adjacency.to_excel(writer, sheet_name=neurotransmitter_id[:31])
    
    return None


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BACKGROUND_SOURCE,
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR,
        sys.argv[3] if len(sys.argv) > 3 else DEFAULT_P_VALUE_METHOD,
        int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != 'None' else None,
        (sys.argv[5] == 'True') if len(sys.argv) > 5 else True, # ! normalize weights by maximum
    )

