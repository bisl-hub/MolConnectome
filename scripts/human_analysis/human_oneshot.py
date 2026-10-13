"""Human stage 1b: directed signaling strength per diagnosis, gene set and directed region pair.

Input:  human expression and sample-matching table, the curated gene list and gene sets, one
        background panel file; an optional label-shuffle seed.
Output: <output_dir>/<background stem>_<p method>/unshuffled_directed_signaling_strength#{CTL,SCZ}.csv.gz
        (label_shuffled/run_NNNNNN#{CTL,SCZ}.csv.gz for a shuffle)
Usage:  see HUMAN_PIPELINE.md, section 4 (normally run by human_label_shuffle_wrapper.py).
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys
import itertools
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import yaml

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

import src.correlation as cor
import src.quantile as qu
from src.human_expression_load import (
    load_human_expression_by_region_and_disease,
    resolve_sample_matching,
)
from src.project_variables import (
    HUMAN_REGIONS,
    DISEASES,
    N_CONTROL_PANELS,
)


RUN_SEED_STRIDE = SETTINGS['human_oneshot']['run_seed_stride']  # 10 pairs * 2 diseases = 20 flat indices, far below this stride


DEFAULT_BACKGROUND_SOURCE = os.path.join(PROJECT_ROOT, 
    SETTINGS['project_structure_settings']['metadata_settings']['metadata_directory'], 
    SETTINGS['gene_set_settings']['gene_set_data_dir'], 
    SETTINGS['gene_set_settings']['default_bg_gene_selection'], 
    SETTINGS['gene_set_settings']['default_bg_source'])
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT,
    SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_label_shuffle_wrapper']['unshuffled_dir_name'])

DEFAULT_P_VALUE_METHOD = SETTINGS['human_oneshot']['correlation_p_value_method']
PERMUTATION_RNG_SEED = SETTINGS['human_oneshot']['correlation_permutation_seed']
N_PERMUTATIONS = SETTINGS['human_oneshot']['correlation_n_permutations']
QUANTILE_RNG_SEED = SETTINGS['human_oneshot']['quantile_bg_rng_seed']
N_NULL_SAMPLES = SETTINGS['human_oneshot']['quantile_bg_n_null_samples']


# 'restricted': only resample genes with actual data for this region pair,
# both the pool AND the sample size (qu._null_distribution_restricted).
RESTRICTED_SEED_BASE = QUANTILE_RNG_SEED
TIE_RANK_HANDLERS = ('strict', 'inclusive', 'mid-rank', 'discard')


def _matched_donor_ids(human_exprs: dict, sample_matching: pd.DataFrame,
                        region_a: str, region_b: str) -> tuple[list[str], list[str]]:
    """Donor ids with a non-null raw sample in both regions (per
    sample_matching) AND present in both regions' disease-restricted,
    donor_id-keyed expression columns (human_exprs[region].columns, from
    load_human_expression_by_region_and_disease). Unlike
    cor.matched_sample_ids, sender/receiver key on the SAME donor_id here --
    human_expression_load already renamed every cohort's raw sample id to
    donor_id during the region merge -- so both returned lists are identical.
    """
    matched = sample_matching[[region_a, region_b]].dropna()
    donor_ids = matched.index[
        matched.index.isin(human_exprs[region_a].columns) & matched.index.isin(human_exprs[region_b].columns)
    ].tolist()
    return donor_ids, donor_ids


def _permutation_seed(label_shuffle_seed: int, pair_index: int, disease_idx: int) -> int:
    """label_shuffle_seed=None is treated as 0. disease_idx keeps CTL/SCZ
    from ever drawing the same permutation at the same region pair."""
    effective_seed = 0 if label_shuffle_seed is None else label_shuffle_seed
    flat_index = pair_index * len(DISEASES) + disease_idx
    return PERMUTATION_RNG_SEED + effective_seed * RUN_SEED_STRIDE + flat_index


_worker_state = {}


def _init_worker(background_source: str, p_value_method: str, label_shuffle_seed: int,
                 tie_rank_handler: str = 'mid-rank',
                 ) -> None:
    final_genes = cor.load_final_genes()
    background_genes = cor.load_background_genes(background_source)  # {0..9: list of genes}
    gene_set_specs = qu._load_gene_set_specs(final_genes)

    sample_matching = resolve_sample_matching(label_shuffle_seed)
    expr_by_region_disease = load_human_expression_by_region_and_disease(sample_matching=sample_matching)

    _worker_state.update(
        expr_by_region_disease=expr_by_region_disease,
        sample_matching=sample_matching,
        final_genes=final_genes,
        background_genes=background_genes,
        gene_set_specs=gene_set_specs,
        p_value_method=p_value_method,
        label_shuffle_seed=label_shuffle_seed,
        tie_rank_handler=tie_rank_handler,
    )


def _process_region_pair(pair_index: int, region_a: str, region_b: str, disease: str) -> list[dict]:
    """All work for one (region pair, disease), fully in memory: correlate
    all panels (no CSV read or write), then run the quantile/resampling logic
    and return already-aggregated rows -- no per-control files, no separate
    merge step."""
    expr_by_region_disease = _worker_state['expr_by_region_disease']
    sample_matching = _worker_state['sample_matching']
    final_genes = _worker_state['final_genes']
    background_genes = _worker_state['background_genes']
    gene_set_specs = _worker_state['gene_set_specs']
    p_value_method = _worker_state['p_value_method']
    label_shuffle_seed = _worker_state['label_shuffle_seed']
    tie_rank_handler = _worker_state['tie_rank_handler']

    count_and_sum_fn = qu._non_filtered_positive_count_and_sum # # no p-filter and sum

    human_exprs = {
        region_a: expr_by_region_disease[region_a][disease],
        region_b: expr_by_region_disease[region_b][disease],
    }
    sample_ids_a, sample_ids_b = _matched_donor_ids(human_exprs, sample_matching, region_a, region_b)
    n_donors = len(sample_ids_a)

    disease_idx = DISEASES.index(disease)

    perm_indices_a = perm_indices_b = None
    if p_value_method == 'permutation': # depreciated, but for compatibility.
        # None (the real/unshuffled baseline run) must resolve to a fixed
        # sentinel, not OS entropy -- matches _permutation_seed's own
        # None-handling one function above.
        effective_label_shuffle_seed = 0 if label_shuffle_seed is None else label_shuffle_seed
        correlation_permutation_seed = np.random.default_rng(
            seed=effective_label_shuffle_seed).integers(0, 20260701 * 2)  # different from label shuffle seed
        # Generated ONCE per (pair, disease), reused across all panels below.
        perm_indices_a, perm_indices_b = cor.generate_permutation_indices(
            n_donors, N_PERMUTATIONS, _permutation_seed(correlation_permutation_seed, pair_index, disease_idx)
        )

    def correlate(gene_list):
        X = cor.panel_matrix(human_exprs, region_a, gene_list, sample_ids_a)
        Y = cor.panel_matrix(human_exprs, region_b, gene_list, sample_ids_b)
        if p_value_method == 'permutation':
            r, p, _ = cor.cross_correlation_and_permutation_pvalues(X, Y, perm_indices_a, perm_indices_b)
        else:
            r, p, _ = cor.cross_correlation_and_pvalues(X, Y)
        return r, p

    r_ab, p_ab = correlate(final_genes)
    r_ba, p_ba = r_ab.T, p_ab.T  # Pearson r (and its p-value) is symmetric -- free, exact

    control_data = {}
    for control_idx in range(N_CONTROL_PANELS):
        r_c_ab, p_c_ab = correlate(background_genes[control_idx])
        control_data[control_idx] = (r_c_ab, p_c_ab, r_c_ab.T, p_c_ab.T)

    rows = []
    for spec_index, spec in enumerate(gene_set_specs):
        for sender_region, receiver_region, r_mat, p_mat, is_forward in (
            (region_a, region_b, r_ab, p_ab, True),
            (region_b, region_a, r_ba, p_ba, False),
        ):
            real_r = r_mat[np.ix_(spec['cat2_idx'], spec['cat1_idx'])]
            real_p = p_mat[np.ix_(spec['cat2_idx'], spec['cat1_idx'])]
            n_qualifying, real_value = count_and_sum_fn(real_r, real_p)
            n_qualifying = int(n_qualifying)
            real_value = float(real_value)

            # null sample size = this gene set's own cat2/cat1 members with
            # valid real data for this pair (see qu._null_distribution_restricted)
            valid_real_row = ~np.isnan(r_mat).all(axis=1)
            valid_real_col = ~np.isnan(r_mat).all(axis=0)
            n_cat2_avail = int(valid_real_row[spec['cat2_idx']].sum())
            n_cat1_avail = int(valid_real_col[spec['cat1_idx']].sum())

            z_scores = np.empty(N_CONTROL_PANELS)
            for control_idx in range(N_CONTROL_PANELS):
                r_c_ab, p_c_ab, r_c_ba, p_c_ba = control_data[control_idx]
                r_c_mat, p_c_mat = (r_c_ab, p_c_ab) if is_forward else (r_c_ba, p_c_ba)
                # Deterministic per (pair, spec, direction, control) -- no
                # sharing across gene sets needed, resampling is cheap.
                direction_offset = 0 if is_forward else N_CONTROL_PANELS
                seed = (RESTRICTED_SEED_BASE + pair_index * 10_000_000
                        + spec_index * (2 * N_CONTROL_PANELS) + direction_offset + control_idx)
                restricted_rng = np.random.default_rng(seed)
                _, null_values = qu._null_distribution_restricted(
                    r_c_mat, p_c_mat, n_cat2_avail, n_cat1_avail, N_NULL_SAMPLES, restricted_rng,
                    qualifying_sum_fn=count_and_sum_fn,
                )
                z_scores[control_idx] = qu.compute_z(tie_rank_handler, null_values, real_value)['z']
            rows.append({
                'gene_set': spec['key'],
                'source': spec['source'],
                'neurotransmitter': spec['neurotransmitter'],
                'sender_region': sender_region,
                'receiver_region': receiver_region,
                'n_cat1': spec['n_cat1'],
                'n_cat2': spec['n_cat2'],
                'n_donors': n_donors,
                'n_qualifying_real_pairs': n_qualifying,
                'real_value': real_value,
                'mean_z': float(z_scores.mean()),
                'z_std': float(z_scores.std()),
                'z_min': float(z_scores.min()),
                'z_max': float(z_scores.max()),
            })
    return rows


def _background_slug(background_source: str) -> str:
    return Path(background_source).stem.lower()


def output_path(output_dir: str, background_source: str, p_value_method: str,
                label_shuffle_seed: int, disease: str) -> str:
    base = os.path.join(output_dir, f'{_background_slug(background_source)}_{p_value_method}')
    if label_shuffle_seed is None:
        return os.path.join(base, f'unshuffled_directed_signaling_strength#{disease}.csv.gz')
    return os.path.join(base, 'label_shuffled', f'run_{label_shuffle_seed:06d}#{disease}.csv.gz')


def main(background_source: str = DEFAULT_BACKGROUND_SOURCE,
         output_dir: str = DEFAULT_OUTPUT_DIR,
         p_value_method: str = DEFAULT_P_VALUE_METHOD,
         label_shuffle_seed: int = None,
         n_workers: int = None,
         tie_rank_handler: str = 'mid-rank',
         ):
    if tie_rank_handler not in TIE_RANK_HANDLERS:
        raise ValueError(f'tie_rank_handler must be one of {TIE_RANK_HANDLERS}, got {tie_rank_handler!r}')
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 2)

    region_pairs = list(enumerate(itertools.combinations(HUMAN_REGIONS, 2)))
    work_items = [
        (pair_index, region_a, region_b, disease)
        for pair_index, (region_a, region_b) in region_pairs
        for disease in DISEASES
    ]
    rows_by_disease = {disease: [] for disease in DISEASES}

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker,
        initargs=(background_source, p_value_method, label_shuffle_seed, tie_rank_handler)
    ) as executor:
        futures = {
            executor.submit(_process_region_pair, pair_index, region_a, region_b, disease):
                (region_a, region_b, disease)
            for pair_index, region_a, region_b, disease in work_items
        }
        for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            region_a, region_b, disease = futures[future]
            rows = future.result()
            rows_by_disease[disease].extend(rows)
            print(f'[{done_count}/{len(work_items)}] {region_a}-{region_b} ({disease}): {len(rows)} rows')

    dfs = {}
    for disease, rows in rows_by_disease.items():
        df = pd.DataFrame(rows)
        df = df.sort_values(['sender_region', 'receiver_region', 'gene_set']).reset_index(drop=True)
        path = output_path(output_dir, background_source, p_value_method, label_shuffle_seed, disease)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, compression='gzip')
        print(f'disease={disease} label_shuffle_seed={label_shuffle_seed} '
              f'{len(df)} rows written to {path}')
        dfs[disease] = df
    return dfs


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BACKGROUND_SOURCE,
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR,
        sys.argv[3] if len(sys.argv) > 3 else DEFAULT_P_VALUE_METHOD,
        int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != 'None' else None,
        int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != 'None' else None,
        sys.argv[6] if len(sys.argv) > 6 else 'mid-rank',
    )
