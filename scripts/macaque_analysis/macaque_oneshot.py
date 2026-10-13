"""Macaque step 1: directed signaling strength for every gene set and directed region pair.

Input:  macaque expression and sample-matching table, the curated gene list and gene sets, one
        background panel file.
Output: <output_dir>/<background stem>_<p method>/directed_signaling_strength.csv
Usage:  see MACAQUE_PIPELINE.md, section 4.
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
from src.project_variables import MACAQUE_REGIONS, N_CONTROL_PANELS


RUN_SEED_STRIDE = SETTINGS['macaque_oneshot']['run_seed_stride']  # 105 region pairs, far below this stride

DEFAULT_BACKGROUND_SOURCE = os.path.join(PROJECT_ROOT, 
    SETTINGS['project_structure_settings']['metadata_settings']['metadata_directory'], 
    SETTINGS['gene_set_settings']['gene_set_data_dir'], 
    SETTINGS['gene_set_settings']['default_bg_gene_selection'], 
    SETTINGS['gene_set_settings']['default_bg_source'])

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 
    SETTINGS['analysis_results']['mq_analysis_save_dir'])

DEFAULT_P_VALUE_METHOD = SETTINGS['macaque_oneshot']['correlation_p_value_method']
PERMUTATION_RNG_SEED = SETTINGS['macaque_oneshot']['correlation_permutation_seed']
N_PERMUTATIONS = SETTINGS['macaque_oneshot']['correlation_n_permutations']
QUANTILE_RNG_SEED = SETTINGS['macaque_oneshot']['quantile_bg_rng_seed']
N_NULL_SAMPLES = SETTINGS['macaque_oneshot']['quantile_bg_n_null_samples']


RESTRICTED_SEED_BASE = QUANTILE_RNG_SEED
# tie_rank_handler: how ties between a real value and its null distribution
# are credited when converting rank -> quantile -> z (src/quantile.py's
# compute_z). 'discard' collapses the tied mass (null == real, e.g. the common
# zero-qualifying-pairs degeneracy) down to a single mid-rank-credited
# representative before ranking, shrinking the quantile denominator to match
# (qu._tie_adjusted_rank_discard).
TIE_RANK_HANDLERS = ('strict', 'inclusive', 'mid-rank', 'discard')


def _permutation_seed(run_seed: int, pair_index: int) -> int:
    """run_seed=None (the real/unrepeated baseline run) resolves to the same
    effective seed as run_seed=0, matching human_oneshot.py's
    _permutation_seed's own None-handling."""
    effective_seed = 0 if run_seed is None else run_seed
    return PERMUTATION_RNG_SEED + effective_seed * RUN_SEED_STRIDE + pair_index


_worker_state = {}



def _init_worker(background_source: str, p_value_method: str, run_seed: int,
                 tie_rank_handler: str = 'mid-rank',) -> None:
    if tie_rank_handler not in TIE_RANK_HANDLERS:
        raise ValueError(f'tie_rank_handler must be one of {TIE_RANK_HANDLERS}, got {tie_rank_handler!r}')

    final_genes = cor.load_final_genes()
    background_genes = cor.load_background_genes(background_source)  # {0..9: list of genes}
    gene_set_specs = qu._load_gene_set_specs(final_genes)
    monkey_exprs = load_macaque_log2_cpm(filtered=True)
    sample_matching = load_macaque_sample_matching()

    _worker_state.update(
        monkey_exprs=monkey_exprs,
        sample_matching=sample_matching,
        final_genes=final_genes,
        background_genes=background_genes,
        gene_set_specs=gene_set_specs,
        p_value_method=p_value_method,
        run_seed=run_seed,
        tie_rank_handler=tie_rank_handler,
    )


def _z_stats(z_arr: np.ndarray) -> dict:
    """mean/std/min/max over the per-control-panel z array."""
    return {
        'mean_z': float(z_arr.mean()),
        'z_std': float(z_arr.std()),
        'z_min': float(z_arr.min()),
        'z_max': float(z_arr.max()),
    }


def _process_region_pair(pair_index: int, region_a: str, region_b: str) -> list[dict]:
    """All work for one region pair, fully in memory: correlate all panels
    (no CSV read or write), then run the quantile/resampling logic and return
    already-aggregated rows -- no per-control files, no separate merge step."""
    monkey_exprs = _worker_state['monkey_exprs']
    sample_matching = _worker_state['sample_matching']
    final_genes = _worker_state['final_genes']
    background_genes = _worker_state['background_genes']
    gene_set_specs = _worker_state['gene_set_specs']
    p_value_method = _worker_state['p_value_method']
    run_seed = _worker_state['run_seed']
    tie_rank_handler = _worker_state['tie_rank_handler']
    count_and_sum_fn = qu._non_filtered_positive_count_and_sum

    sample_ids_a, sample_ids_b = cor.matched_sample_ids(monkey_exprs, sample_matching, region_a, region_b)
    n_monkeys = len(sample_ids_a)

    perm_indices_a = perm_indices_b = None
    if p_value_method == 'permutation':  # deprecated, but for compatibility.
        # Generated ONCE per pair, reused across all panels below -- seed
        # depends on run_seed, so every repetition draws an independent
        # permutation.
        perm_indices_a, perm_indices_b = cor.generate_permutation_indices(
            n_monkeys, N_PERMUTATIONS, _permutation_seed(run_seed, pair_index)
        )

    def correlate(gene_list):
        X = cor.panel_matrix(monkey_exprs, region_a, gene_list, sample_ids_a)
        Y = cor.panel_matrix(monkey_exprs, region_b, gene_list, sample_ids_b)
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

            valid_real_row = ~np.isnan(r_mat).all(axis=1)
            valid_real_col = ~np.isnan(r_mat).all(axis=0)
            n_cat2_avail = int(valid_real_row[spec['cat2_idx']].sum())
            n_cat1_avail = int(valid_real_col[spec['cat1_idx']].sum())

            z_scores = np.empty(N_CONTROL_PANELS)
            for control_idx in range(N_CONTROL_PANELS):
                r_c_ab, p_c_ab, r_c_ba, p_c_ba = control_data[control_idx]
                r_c_mat, p_c_mat = (r_c_ab, p_c_ab) if is_forward else (r_c_ba, p_c_ba)
                # Deterministic per (pair, spec, direction, control) --
                # independent of run_seed (resampling basis is a separate
                # randomization axis from the permutation sample-shuffle
                # being scaled up across runs).
                direction_offset = 0 if is_forward else N_CONTROL_PANELS
                seed = (RESTRICTED_SEED_BASE + pair_index * 10_000_000
                        + spec_index * (2 * N_CONTROL_PANELS) + direction_offset + control_idx)
                restricted_rng = np.random.default_rng(seed)
                _, null_values = qu._null_distribution_restricted(
                    r_c_mat, p_c_mat, n_cat2_avail, n_cat1_avail, N_NULL_SAMPLES, restricted_rng,
                    qualifying_sum_fn=count_and_sum_fn,
                )
                z_scores[control_idx] = qu.compute_z(tie_rank_handler, null_values, real_value)['z']

            row = {
                'gene_set': spec['key'],
                'source': spec['source'],
                'neurotransmitter': spec['neurotransmitter'],
                'sender_region': sender_region,
                'receiver_region': receiver_region,
                'n_cat1': spec['n_cat1'],
                'n_cat2': spec['n_cat2'],
                'n_monkeys': n_monkeys,
                'n_qualifying_real_pairs': n_qualifying,
                'real_value': real_value,
                **_z_stats(z_scores),
            }
            rows.append(row)
    return rows


def _background_slug(background_source: str) -> str:
    return Path(background_source).stem.lower()


def construct_output_path(output_dir: str, background_source: str, p_value_method: str, run_seed: int = None) -> str:
    """run_seed=None -- the real, unrepeated baseline run (matching
    human_oneshot.py's label_shuffle_seed=None sentinel) -- gets a plain,
    uncompressed, unseeded 'directed_signaling_strength.csv'. Any other
    run_seed is a repetition-sweep replicate, named/compressed as before."""
    base = os.path.join(output_dir, f'{_background_slug(background_source)}_{p_value_method}')
    if run_seed is None:
        return os.path.join(base, 'directed_signaling_strength.csv')
    return os.path.join(base, f'run_{run_seed:06d}.csv.gz')


def main(background_source: str = DEFAULT_BACKGROUND_SOURCE,
         output_dir: str = DEFAULT_OUTPUT_DIR,
         p_value_method: str = DEFAULT_P_VALUE_METHOD,
         run_seed: int = None,
         n_workers: int = None,
         tie_rank_handler: str = 'mid-rank'):

    print(f'background_source={background_source} \n output_dir={output_dir} \n '
          f'p_value_method={p_value_method}, '
          f'run_seed={run_seed}, tie_rank_handler={tie_rank_handler}'
          f' #### ---- starting ....')
    
    if tie_rank_handler not in TIE_RANK_HANDLERS:
        raise ValueError(f'tie_rank_handler must be one of {TIE_RANK_HANDLERS}, got {tie_rank_handler!r}')
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 2)

    region_pairs = list(enumerate(itertools.combinations(MACAQUE_REGIONS, 2)))
    all_rows = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker,
        initargs=(background_source, p_value_method, run_seed, tie_rank_handler, )
    ) as executor:
        futures = {
            executor.submit(_process_region_pair, pair_index, region_a, region_b): (region_a, region_b)
            for pair_index, (region_a, region_b) in region_pairs
        }
        for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            region_a, region_b = futures[future]
            rows = future.result()
            all_rows.extend(rows)
            print(f'[{done_count}/{len(region_pairs)}] {region_a}-{region_b}: {len(rows)} rows')

    df = pd.DataFrame(all_rows)
    df = df.sort_values(['sender_region', 'receiver_region', 'gene_set']).reset_index(drop=True)
    path = construct_output_path(output_dir, background_source, p_value_method, run_seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    compression = 'gzip' if path.endswith('.gz') else None
    df.to_csv(path, index=False, compression=compression)
    print(f'run_seed={run_seed} tie_rank_handler={tie_rank_handler} '
          f': {len(df)} rows written to {path}')
    return df


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BACKGROUND_SOURCE,
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR,
        sys.argv[3] if len(sys.argv) > 3 else DEFAULT_P_VALUE_METHOD,
        int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != 'None' else None,
        int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != 'None' else None,
        sys.argv[6] if len(sys.argv) > 6 else 'mid-rank',
    )
