"""Human stage 1c: stage 1b's z matrices weighted by each diagnosis's sender-region abundance.

Input:  the CTL and SCZ signaling files from stage 1b and abundance files from stage 1a.
Output: <output_dir>/<background stem>_<p method>/weighted_connectivity#{CTL,SCZ}.xlsx and
        meanz_weighted#{CTL,SCZ}.csv
Usage:  see HUMAN_PIPELINE.md, section 4 (normally run by human_label_shuffle_wrapper.py).
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

import human_oneshot
import human_abundance
from src.calculate_abundance import (
    multiply_weight_with_adjacency,
    weight_sum_adjacencies,
)
from src.project_variables import HUMAN_REGIONS, GENE_SET_TO_USE, DISEASES
# per-panel z relative to its own null: comparable magnitude across gene sets
# of very different sizes.
STRENGTH_COLUMN = 'mean_z'

# With N_NULL_SAMPLES permutations, rank in [0, N_NULL_SAMPLES], so the
# quantile in src/quantile.py is clamped to [0.5/(N+1), (N+0.5)/(N+1)] and
# z = norm.ppf(quantile) is clamped to [-Z_BOUND, Z_BOUND] (mean_z, being an
# average of such z's, stays within the same bound). This is the exact
# min/max of the range, used below for min-max normalization instead of the
# batch's own empirical min/max.
Z_BOUND = abs(scipy.stats.norm.ppf(0.5 / (human_oneshot.N_NULL_SAMPLES + 1)))

DEFAULT_SIGNALING_STRENGTH_PATHS = {
    disease: human_oneshot.output_path(
        human_oneshot.DEFAULT_OUTPUT_DIR, human_oneshot.DEFAULT_BACKGROUND_SOURCE,
        human_oneshot.DEFAULT_P_VALUE_METHOD, None, disease,
    )
    for disease in DISEASES
}
DEFAULT_ABUNDANCE_PATHS = dict(zip(
    DISEASES,
    human_abundance._abundance_output_paths(str(human_abundance.DEFAULT_ABUNDANCE_SAVE_PATH)),
))
DEFAULT_OUTPUT_DIR = human_oneshot.DEFAULT_OUTPUT_DIR


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
        pivoted = subset.pivot(index='sender_region', columns='receiver_region',
                               values=value_column)
        pivoted = pivoted.reindex(index=regions, columns=regions)
        if normalize:
            pivoted = normalize_connectivity_strength(pivoted)
        adjacency_dict[gene_set_name] = pivoted
    return adjacency_dict


def weighted_connectivity_output_paths(output_dir: str, background_source: str, p_value_method: str,
                                       disease: str) -> 'dict[str, str]':
    """Lands in the SAME `{background_slug}_{p_value_method}/` folder
    human_oneshot.py's output_path() uses under this `output_dir` --
    weighted connectivity is derived from oneshot's signaling-strength
    output, so it depends on background-gene selection exactly like oneshot
    does, unlike human_abundance.py. No separate label_shuffle_seed-based
    prefix/subfolder here -- the caller already gives each run (unshuffled or
    a given replicate) its own `output_dir`, so plain filenames suffice."""
    base = os.path.join(output_dir, f'{human_oneshot._background_slug(background_source)}_{p_value_method}')
    return {
        'weighted_connectivity': os.path.join(base, f'weighted_connectivity#{disease}.xlsx'),
        'meanz_weighted': os.path.join(base, f'meanz_weighted#{disease}.csv'),
    }


def main(signaling_strength_ctl_path: str = DEFAULT_SIGNALING_STRENGTH_PATHS['CTL'],
         signaling_strength_scz_path: str = DEFAULT_SIGNALING_STRENGTH_PATHS['SCZ'],
         abundance_ctl_path: str = DEFAULT_ABUNDANCE_PATHS['CTL'],
         abundance_scz_path: str = DEFAULT_ABUNDANCE_PATHS['SCZ'],
         output_dir: str = DEFAULT_OUTPUT_DIR,
         background_source: str = human_oneshot.DEFAULT_BACKGROUND_SOURCE,
         p_value_method: str = human_oneshot.DEFAULT_P_VALUE_METHOD,
         label_shuffle_seed: int = None, 
         normalize_weights: bool = False):
    signaling_paths = {'CTL': signaling_strength_ctl_path, 'SCZ': signaling_strength_scz_path}
    abundance_paths = {'CTL': abundance_ctl_path, 'SCZ': abundance_scz_path}

    results = {}
    for disease in DISEASES:
        signaling_data = pd.read_csv(signaling_paths[disease])
        mean_abundance = pd.read_excel(abundance_paths[disease], sheet_name='mean', index_col=0).loc[:, HUMAN_REGIONS]

        each_gs_adj_dict = build_gene_set_adjacency(signaling_data, GENE_SET_TO_USE, HUMAN_REGIONS)
        weighted_by_neurotransmitter = multiply_weight_with_adjacency(
            each_gs_adj_dict, GENE_SET_TO_USE, mean_abundance, normalize_weights=normalize_weights)
        weighted_total = weight_sum_adjacencies(
            each_gs_adj_dict, GENE_SET_TO_USE, mean_abundance, organism='human')

        paths = weighted_connectivity_output_paths(output_dir, background_source, p_value_method, disease)
        os.makedirs(os.path.dirname(paths['weighted_connectivity']), exist_ok=True)
        with pd.ExcelWriter(paths['weighted_connectivity']) as writer:
            weighted_total.to_excel(writer, sheet_name='total')
            for neurotransmitter_id, adjacency in weighted_by_neurotransmitter.items():
                adjacency.to_excel(writer, sheet_name=neurotransmitter_id[:31])

        weighted_total.to_csv(paths['meanz_weighted'])

        print(f'disease={disease} label_shuffle_seed={label_shuffle_seed}: wrote {paths["weighted_connectivity"]}')

        results[disease] = (weighted_by_neurotransmitter, weighted_total)

    return results


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SIGNALING_STRENGTH_PATHS['CTL'],
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SIGNALING_STRENGTH_PATHS['SCZ'],
        sys.argv[3] if len(sys.argv) > 3 else DEFAULT_ABUNDANCE_PATHS['CTL'],
        sys.argv[4] if len(sys.argv) > 4 else DEFAULT_ABUNDANCE_PATHS['SCZ'],
        sys.argv[5] if len(sys.argv) > 5 else DEFAULT_OUTPUT_DIR,
        sys.argv[6] if len(sys.argv) > 6 else human_oneshot.DEFAULT_BACKGROUND_SOURCE,
        sys.argv[7] if len(sys.argv) > 7 else human_oneshot.DEFAULT_P_VALUE_METHOD,
        int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != 'None' else None,
        (sys.argv[9] == 'True') if len(sys.argv) > 9 else False, # ! normalize weights by maximum
    )
