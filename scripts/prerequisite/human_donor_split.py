"""Human, optional: stratified splits of the donors into two groups.

Input:  human sample-matching table.
Output: human_analysis_results/donor_splits/ (one CSV per ratio, resampling mode and seed)
Usage:  python scripts/prerequisite/human_donor_split.py (runs a fixed set of 10 seeds; not part
        of the pipeline in HUMAN_PIPELINE.md).
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import pandas as pd
import numpy as np
import sys
import yaml
from pathlib import Path

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

from src.human_expression_load import (
    load_sample_matching_table,
    _dataset_abbr_of_donor,
    _coverage_pattern,
)

DEFAULT_SPLIT_SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_donor_split']['report_save_dir'],
)


def stratum_key(sample_matching: pd.DataFrame) -> pd.Series:
    """(dataset abbreviation, diagnosis, non-missing region-coverage pattern)
    signature per donor -- the stratification unit for split_donors(). Same
    dataset-abbreviation/coverage-pattern helpers as
    human_expression_load.shuffle_disease_labels(), plus diagnosis, since
    here (unlike the label-shuffle use case) diagnosis composition itself is
    exactly what must be preserved by the split.

    Args:
        sample_matching: table from load_sample_matching_table().

    Returns:
        pd.Series: donor_id -> (dataset_abbr, disease, coverage_pattern) tuple.
    """
    return pd.Series(
        [(_dataset_abbr_of_donor(donor_id), row['disease'], _coverage_pattern(row))
         for donor_id, row in sample_matching.iterrows()],
        index=sample_matching.index,
    )


def split_donors(sample_matching: pd.DataFrame, ratio: float, resample: bool,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Stratified two-way split of donors, sized by `ratio`.

    Stratifies by stratum_key() -- (dataset, diagnosis, region-coverage
    pattern) -- and splits *within* each stratum independently, so the
    per-dataset, per-diagnosis, and per-region-coverage composition of group
    A and group B each track the full table's composition instead of
    drifting toward whichever strata happen to dominate a global shuffle.

    Each stratum's split point is rounded stochastically (floor(ratio * n),
    plus one more with probability equal to the leftover fraction) rather
    than always rounding the same way, so singleton/small strata (10 of the
    49 real strata have exactly 1 donor) don't get deterministically shunted
    to the same side every time -- that would happen with plain
    round()/np.rint() on ratio=0.5, since round-half-to-even always maps 0.5
    to 0. Group sizes stay unbiased in expectation and converge to `ratio`
    as strata get larger.

    Args:
        sample_matching: table from load_sample_matching_table() (or an
            already-restricted subset of it).
        ratio: fraction of each stratum assigned to group A; group B gets the
            remainder. E.g. 0.5 for a half/half split, 0.8 for an 80/20 split.
        resample: if False, a clean partition -- every donor in a stratum
            appears in exactly one of the two groups, no duplicates. If True,
            group A and group B are instead each drawn independently *with
            replacement* from the stratum (bootstrap resampling), at the same
            sizes the partition would use -- so a donor may appear more than
            once in one group, or in both groups.
        rng: caller-owned generator, e.g. np.random.default_rng(seed) -- pass
            an independently-seeded rng per replicate.

    Returns:
        tuple[np.ndarray, np.ndarray]: (group_a_donor_ids, group_b_donor_ids).
            May contain repeated donor ids when resample=True; never
            otherwise.
    """
    if not 0 < ratio < 1:
        raise ValueError(f'ratio must be in (0, 1), got {ratio}')

    block_keys = stratum_key(sample_matching)
    group_a_parts = []
    group_b_parts = []
    for _, donor_ids in sample_matching.groupby(block_keys).groups.items():
        donor_ids = np.asarray(donor_ids)
        n = len(donor_ids)
        target = ratio * n
        n_a = int(np.floor(target))
        if rng.random() < (target - n_a):
            n_a += 1

        if resample:
            group_a_parts.append(rng.choice(donor_ids, size=n_a, replace=True))
            group_b_parts.append(rng.choice(donor_ids, size=n - n_a, replace=True))
        else:
            shuffled = rng.permutation(donor_ids)
            group_a_parts.append(shuffled[:n_a])
            group_b_parts.append(shuffled[n_a:])
    return np.concatenate(group_a_parts), np.concatenate(group_b_parts)


def build_split_table(sample_matching: pd.DataFrame, ratio: float, resample: bool,
                      seed: int) -> pd.DataFrame:
    """split_donors() plus per-donor metadata, collapsed to one row per
    (group, donor_id) with a 'copies' count -- under resample=True a donor
    can be drawn more than once into the same group; 'copies' says how many
    times, rather than repeating the row (repeat count is what a caller
    needs to rebuild a bootstrap expression matrix via
    np.repeat(donor_id, copies) / column reindexing).

    Args:
        sample_matching: table from load_sample_matching_table().
        ratio: see split_donors().
        resample: see split_donors().
        seed: seeds this replicate's np.random.default_rng(seed).

    Returns:
        pd.DataFrame: columns ['donor_id', 'group', 'copies', 'dataset',
            'diagnosis', 'region_coverage'], sorted by (group, donor_id).
            group is 'A' (ratio fraction) or 'B' (the remainder).
    """
    rng = np.random.default_rng(seed)
    group_a, group_b = split_donors(sample_matching, ratio, resample, rng)

    rows = []
    for group_label, donor_ids in (('A', group_a), ('B', group_b)):
        for donor_id, copies in pd.Series(donor_ids).value_counts().items():
            row = sample_matching.loc[donor_id]
            rows.append({
                'donor_id': donor_id,
                'group': group_label,
                'copies': int(copies),
                'dataset': _dataset_abbr_of_donor(donor_id),
                'diagnosis': row['disease'],
                'region_coverage': '+'.join(_coverage_pattern(row)),
            })
    return pd.DataFrame(rows).sort_values(['group', 'donor_id']).reset_index(drop=True)


def output_path(output_dir: str, ratio: float, resample: bool, seed: int) -> str:
    resample_label = 'resample' if resample else 'no-resample'
    return os.path.join(output_dir, f'donor_split_ratio#{ratio:.2f}_{resample_label}_seed#{seed}.csv')


def main(ratio: float = 0.5, resample: bool = False, seed: int = 1,
         output_dir: str = str(DEFAULT_SPLIT_SAVE_DIR)) -> pd.DataFrame:
    sample_matching = load_sample_matching_table()
    split_table = build_split_table(sample_matching, ratio, resample, seed)

    os.makedirs(output_dir, exist_ok=True)
    save_path = output_path(output_dir, ratio, resample, seed)
    split_table.to_csv(save_path, index=False)

    draws = split_table.groupby('group')['copies'].sum()
    print(f'ratio={ratio}, resample={resample}, seed={seed}: wrote {save_path} '
          f'(group A: {draws.get("A", 0)} draws, group B: {draws.get("B", 0)} draws)')
    return split_table


if __name__ == '__main__':
    # ! default settings - remove when modifying.
    initial_seed = 20260818 # seed range boundary
    selector_seed = 20261007 # seed for fixed output
    seeds_generated = np.random.default_rng(selector_seed).integers(
        initial_seed, initial_seed * 2, size=10, endpoint=True)
    for seed in seeds_generated:
        main(0.5, False, seed)
    # main(
    #     float(sys.argv[1]) if len(sys.argv) > 1 else 0.5,
    #     (sys.argv[2] == 'True') if len(sys.argv) > 2 else False,
    #     int(sys.argv[3]) if len(sys.argv) > 3 else 1,
    #     sys.argv[4] if len(sys.argv) > 4 else str(DEFAULT_SPLIT_SAVE_DIR),
    # )
