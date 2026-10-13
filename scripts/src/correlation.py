"""Curated-gene and background-panel loading, sample matching, and cross-region Pearson
correlation with transformation or permutation p-values.

Used by: macaque_oneshot.py, macaque_all_gene_correlation.py, human_oneshot.py,
         human_corr_contrib.py.
"""

import pandas as pd
import numpy as np
import os
import sys
import itertools
import pickle
import scipy.stats
import concurrent.futures
import datetime


from pathlib import Path
import yaml
with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)
PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))


FINAL_GENES_PATH = os.path.join(PROJECT_ROOT, 
                                SETTINGS['project_structure_settings']['metadata_settings']['metadata_directory'],
                                SETTINGS['gene_set_settings']['gene_set_data_dir'],
                                SETTINGS['gene_set_settings']['neurotransmission_gene_list'])
PERMUTATION_CHUNK_SIZE = SETTINGS['analysis_global_settings']['correlation_permutation_chunk_size']


def load_final_genes() -> list[str]:
    return np.load(FINAL_GENES_PATH, allow_pickle=True).tolist()


def load_background_genes(path: str) -> dict[int, list[str]]:
    """Load a 10-panel background/control gene pickle and normalize its keys
    to a stable 0..9 int range, regardless of the source file's own key
    scheme."""
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    ordered_keys = sorted(raw.keys(), key=lambda k: int(k))  # numeric sort, not lexicographic
    return {i: list(raw[k]) for i, k in enumerate(ordered_keys)}


def matched_sample_ids(_exprs: dict[str, pd.DataFrame], sample_matching: pd.DataFrame,
                        region_a: str, region_b: str) -> tuple[list[str], list[str]]:
    """Sample IDs for the monkeys that have a valid sample in BOTH regions,
    in matching order (same monkey at the same position in each list)."""
    matched = sample_matching[[region_a, region_b]].dropna()
    matched = matched[
        matched[region_a].isin(_exprs[region_a].columns)
        & matched[region_b].isin(_exprs[region_b].columns)
    ]
    return matched[region_a].tolist(), matched[region_b].tolist()


def panel_matrix(_exprs: dict[str, pd.DataFrame], region: str,
                  gene_list: list[str], sample_ids: list[str]) -> np.ndarray:
    """(n_samples, n_genes) matrix for `gene_list` genes / `sample_ids` samples
    in `region`. Genes absent from this region become all-NaN columns."""
    sub = _exprs[region].reindex(gene_list)[sample_ids]
    return sub.to_numpy(dtype=float).T


def cross_correlation_and_pvalues(X: np.ndarray, Y: np.ndarray):
    """Pearson correlation + two-sided p-value between every column of X and
    every column of Y across matched rows (samples). Vectorized: one matmul
    for the whole correlation matrix, one vectorized scipy call for all
    p-values -- no per-gene-pair loop.
    """
    n = X.shape[0]
    Zx = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Zy = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
    r = (Zx.T @ Zy) / (n - 1)

    r_safe = np.clip(r, -0.9999999, 0.9999999)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = r_safe * np.sqrt((n - 2) / (1 - r_safe ** 2))
        p = 2 * scipy.stats.t.sf(np.abs(t), df=n - 2) # key implementation
    return r, p, n


def generate_permutation_indices(n_samples: int, n_permutations: int, seed: int):
    """Two independent (n_permutations, n_samples) row-permutation index
    arrays, one per region. Meant to be generated ONCE per region pair and
    reused across every gene panel for that pair -- reusing one shared
    random basis across many downstream uses is statistically harmless here
    since nothing ever compares null draws across panels."""
    rng = np.random.default_rng(seed)
    perm_a = np.argsort(rng.random((n_permutations, n_samples)), axis=1)
    perm_b = np.argsort(rng.random((n_permutations, n_samples)), axis=1)
    return perm_a, perm_b


def cross_correlation_and_permutation_pvalues(X: np.ndarray, Y: np.ndarray,
                                               perm_indices_a: np.ndarray, perm_indices_b: np.ndarray,
                                               chunk_size: int = PERMUTATION_CHUNK_SIZE):
    """Same (r, p, n) contract as cross_correlation_and_pvalues, but p is a
    one-sided empirical p-value: independently shuffle each region's sample
    order (breaking the true same-monkey pairing), recompute the
    cross-correlation under many such mismatched pairings, and take the
    fraction of those null draws that meet or exceed the real correlation.
    No continuity correction -- p can legitimately be exactly 0.0 if the
    real value exceeds every permutation draw, matching the exact intended
    formula (not a bug to fix).
    """
    n = X.shape[0]
    Zx = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Zy = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
    real_r = (Zx.T @ Zy) / (n - 1)

    n_permutations = perm_indices_a.shape[0]
    count_less = np.zeros_like(real_r)
    done = 0
    while done < n_permutations:
        this_chunk = min(chunk_size, n_permutations - done)
        idx_a = perm_indices_a[done:done + this_chunk]
        idx_b = perm_indices_b[done:done + this_chunk]
        Zx_perm = Zx[idx_a]   # (chunk, n, n_genes_A) -- gather rows in shuffled order
        Zy_perm = Zy[idx_b]   # (chunk, n, n_genes_B)
        r_perm = np.matmul(Zx_perm.transpose(0, 2, 1), Zy_perm) / (n - 1)  # (chunk, n_genes_A, n_genes_B)
        count_less += (r_perm < real_r).sum(axis=0)
        done += this_chunk

    # NaN * False == NaN, but NaN < x == False (not NaN) -- so a structurally
    # absent gene's real_r of NaN would otherwise silently resolve to p=1.0
    # instead of NaN. np.where restores the correct NaN propagation.
    p = np.where(np.isnan(real_r), np.nan, 1.0 - count_less / n_permutations)
    return real_r, p, n


def save_pair_csv(path: str, gene_list: list[str], r: np.ndarray, p: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    genes = np.asarray(gene_list)
    df = pd.DataFrame({
        'gene_a': np.repeat(genes, len(genes)),
        'gene_b': np.tile(genes, len(genes)),
        'r': r.ravel(),
        'p_value': p.ravel(),
    })
    df.to_csv(path, index=False, float_format='%.6g', compression='gzip')


def load_saved_correlation_matrix(coexpression_dir: str, panel_name: str, region_a: str, region_b: str,
                                   gene_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read an already-saved coexpression CSV (written by save_pair_csv)
    back into (r, p) matrices, reindexed to `gene_list` order on both axes
    so the result aligns exactly with a fresh cross_correlation_and_*
    call's output. Lets a caller reuse an expensive already-computed
    correlation (e.g. permutation-based) instead of recomputing it."""
    path = os.path.join(coexpression_dir, panel_name, f'{region_a}__{region_b}.csv.gz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found -- calculate correlation for this '
            f'(background_source, p_value_method) combination first.'
        )
    df = pd.read_csv(path)
    r = df.pivot(index='gene_a', columns='gene_b', values='r').reindex(index=gene_list, columns=gene_list).to_numpy()
    p = df.pivot(index='gene_a', columns='gene_b', values='p_value').reindex(index=gene_list, columns=gene_list).to_numpy()
    return r, p


def main():
    return None


if __name__ == '__main__':
    main()
