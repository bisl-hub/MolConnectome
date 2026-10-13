"""Gene-set loading, and the background null and rank-based z-score of each gene set's summed
positive correlation.

Used by: macaque_oneshot.py, macaque_all_gene_correlation.py, human_oneshot.py,
         human_delta_quantile_report.py, human_corr_contrib.py, baseline_GO_enrichment.py.
"""

import os
import sys
import pickle
import warnings

import numpy as np
import scipy.stats

from pathlib import Path
import yaml
with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)
PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))


from project_variables import GENE_SET_TO_USE
NT_GENE_DIR = os.path.join(PROJECT_ROOT, 
                                SETTINGS['project_structure_settings']['metadata_settings']['metadata_directory'],
                                SETTINGS['gene_set_settings']['gene_set_data_dir'],
                                SETTINGS['gene_set_settings']['neurotransmission_gene_set_source'])


def load_gene_sets() -> dict:
    """CURATED_GENE_SET.pickle: {gene_set: {'cat1': genes, 'cat2': genes}},
    exactly the GENE_SET_TO_USE gene sets."""
    with open(NT_GENE_DIR, 'rb') as f:
        gene_sets = pickle.load(f)
    if set(gene_sets) != set(GENE_SET_TO_USE):
        raise ValueError(f'{NT_GENE_DIR} gene sets {sorted(gene_sets)} != GENE_SET_TO_USE {sorted(GENE_SET_TO_USE)}')
    return gene_sets


def _load_gene_set_specs(final_genes: list[str]) -> list[dict]:
    """Parse CURATED_GENE_SET.pickle into a sorted (deterministic), index-resolved
    list of gene-set specs. cat1='receiver' genes, cat2='sender' genes."""
    gene_sets = load_gene_sets()

    gene_to_pos = {g: i for i, g in enumerate(final_genes)}  # exact order, not re-sorted

    specs = []
    for key in sorted(gene_sets.keys()):
        v = gene_sets[key]
        source, neurotransmitter = key.split('_', 1)
        cat1_idx = np.array([gene_to_pos[g] for g in v['cat1']], dtype=int)
        cat2_idx = np.array([gene_to_pos[g] for g in v['cat2']], dtype=int)
        specs.append({
            'key': key,
            'source': source,
            'neurotransmitter': neurotransmitter,
            'cat1_idx': cat1_idx,
            'cat2_idx': cat2_idx,
            'n_cat1': len(cat1_idx),
            'n_cat2': len(cat2_idx),
        })
    return specs


def _qualifying_count_and_sum(r_block: np.ndarray, p_block: np.ndarray):
    """(count, sum of r) over pairs with r > 0 and p < 0.05."""
    mask = (r_block > 0) & (p_block < 0.05)
    count = mask.sum(axis=(-2, -1))
    value = np.where(mask, r_block, 0.0).sum(axis=(-2, -1))
    return count, value


def _non_filtered_positive_count_and_sum(r_block: np.ndarray, p_block: np.ndarray):
    """(count, sum of r) over pairs with r > 0, ignoring p."""
    mask = (r_block > 0)
    count = mask.sum(axis=(-2, -1))
    value = np.where(mask, r_block, 0.0).sum(axis=(-2, -1))
    return count, value


def _null_distribution(r_matrix: np.ndarray, p_matrix: np.ndarray,
                        sender_basis: np.ndarray, receiver_basis: np.ndarray,
                        n_cat2: int, n_cat1: int,
                        qualifying_sum_fn=_non_filtered_positive_count_and_sum):
    sender_idx = sender_basis[:, :n_cat2]
    receiver_idx = receiver_basis[:, :n_cat1]
    r_null = r_matrix[sender_idx[:, :, None], receiver_idx[:, None, :]]
    p_null = p_matrix[sender_idx[:, :, None], receiver_idx[:, None, :]]
    return qualifying_sum_fn(r_null, p_null)


def _null_distribution_restricted(r_matrix: np.ndarray, p_matrix: np.ndarray,
                                  n_cat2: int, n_cat1: int, n_null: int,
                                  rng: np.random.Generator,
                                  qualifying_sum_fn=_non_filtered_positive_count_and_sum):
    """Like _null_distribution, but only resamples gene positions that have
    at least one valid (non-NaN) correlation in r_matrix, so a
    structurally-missing gene (absent from this region pair's expression
    data entirely) never enters the null. Drawing from every panel position
    instead lets such genes contribute guaranteed zeros (e.g. 32-37% of a
    panel's genes were missing for PFC-CC), diluting the null toward zero
    and inflating apparent significance.

    n_cat2/n_cat1 here MUST be the count of THIS gene set's own cat2/cat1
    members that actually have valid data in the REAL correlation matrix for
    this region pair -- NOT the gene set's full theoretical size. Passing
    the unrestricted size draws a far richer null for gene sets with heavy
    real-data dropout. The caller computes n_cat2/n_cat1 from the REAL
    matrix's own validity mask, restricted to this gene set's
    cat2_idx/cat1_idx.

    Builds its own basis fresh from `rng` each call (no shared basis across
    gene sets).
    """
    valid_sender = np.where(~np.isnan(r_matrix).all(axis=1))[0]
    valid_receiver = np.where(~np.isnan(r_matrix).all(axis=0))[0]
    sender_order = np.argsort(rng.random((n_null, len(valid_sender))), axis=1)
    receiver_order = np.argsort(rng.random((n_null, len(valid_receiver))), axis=1)
    sender_basis = valid_sender[sender_order]
    receiver_basis = valid_receiver[receiver_order]
    return _null_distribution(r_matrix, p_matrix, sender_basis, receiver_basis, n_cat2, n_cat1,
                              qualifying_sum_fn=qualifying_sum_fn)


def _tie_adjusted_rank(null_values: np.ndarray, real_value, tie_rule: str) -> float:
    """Rank of real_value among null_values under the given tie rule:
      'strict'    -- count(null < real)                            (ties: no credit)
      'inclusive' -- count(null <= real)                           (ties: full credit)
      'mid-rank'  -- count(null < real) + 0.5*count(null == real)  (ties: half credit)
    'strict' forces a deterministic floor whenever real_value==0 (the
    minimum a qualifying sum can ever produce), regardless of what fraction
    of the null is ALSO tied at 0 -- 'inclusive'/'mid-rank' let that tied
    fraction inform the result instead. Float-typed uniformly (mid-rank can
    be fractional)."""
    less = int((null_values < real_value).sum())
    if tie_rule == 'strict':
        return float(less)
    equal = int((null_values == real_value).sum())
    if tie_rule == 'inclusive':
        return float(less + equal)
    if tie_rule == 'mid-rank':
        return less + 0.5 * equal
    raise ValueError(f"tie_rule must be one of ('strict', 'inclusive', 'mid-rank'), got {tie_rule!r}")


def _rank_to_z(rank: float, n_null: int) -> float:
    """(rank+0.5)/(n_null+1) continuity-corrected quantile -> z. rank in
    [0, n_null] always keeps the quantile strictly inside (0, 1), so
    norm.ppf never over/underflows."""
    quantile = (rank + 0.5) / (n_null + 1)
    return float(scipy.stats.norm.ppf(quantile))


def _tie_adjusted_rank_discard(null_values: np.ndarray, real_value) -> tuple[float, int]:
    """Like _tie_adjusted_rank, but instead of crediting the FULL tied mass
    (inclusive/mid-rank), collapses it down to a single representative tie
    before ranking -- discarding the rest -- and shrinks the denominator to
    match. Targets the case where real_value==0 and a large fraction of the
    null draws are ALSO trivially 0 (zero qualifying pairs, not genuine null
    variation): that duplicated mass shouldn't count as len(null_values)
    independent comparisons, so the tied group is reduced to one
    representative (mid-rank credit, 0.5).

    Returns (rank, n_reduced): n_reduced is the denominator to use in place
    of len(null_values), since discarding duplicate ties shrinks the null.
    With 0 ties, n_reduced == len(null_values) and this is identical to
    'strict'/'mid-rank' (nothing to discard)."""
    less = int((null_values < real_value).sum())
    equal = int((null_values == real_value).sum())
    n_null = len(null_values)
    if equal == 0:
        return float(less), n_null
    n_reduced = n_null - equal + 1  # discard all but 1 representative tie
    rank = less + 0.5  # mid-rank credit for that 1 remaining tie
    return rank, n_reduced


def compute_z(tie_rank_handler: str, null_values: np.ndarray, real_value: float) -> dict:
    """Per-control-panel z-score for one (gene_set, direction, control) cell.

    'strict'/'inclusive'/'mid-rank' -> {'z': ...}: real_value's rank among
    null_values under that tie rule.

    'discard' -> {'z': ...}: like 'mid-rank', but the tied mass (null ==
    real_value) is collapsed to a single representative before ranking
    (_tie_adjusted_rank_discard) -- both the rank's tie credit AND the
    quantile's denominator shrink accordingly, instead of counting every
    duplicate as independent null variation.
    """
    n_null = len(null_values)

    if tie_rank_handler in ('strict', 'inclusive', 'mid-rank'):
        rank = _tie_adjusted_rank(null_values, real_value, tie_rank_handler)
        return {'z': _rank_to_z(rank, n_null)}

    if tie_rank_handler == 'discard':
        rank, n_reduced = _tie_adjusted_rank_discard(null_values, real_value)
        return {'z': _rank_to_z(rank, n_reduced)}

    raise ValueError(
        f"tie_rank_handler must be one of ('strict', 'inclusive', 'mid-rank', 'discard'), "
        f"got {tie_rank_handler!r}"
    )


def main():
    return None


if __name__ == '__main__':
    main()