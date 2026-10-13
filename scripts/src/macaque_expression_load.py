"""Loads the macaque expression data and sample-matching table.

Used by: macaque_oneshot.py, macaque_abundance.py, macaque_all_gene_correlation.py.
"""

import os
import sys

import pandas as pd
import numpy as np


from pathlib import Path
import yaml
with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)
PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

from project_variables import MACAQUE_REGIONS


MACAQUE_RESIDUALS_DIR = os.path.join(PROJECT_ROOT, 
    SETTINGS['project_structure_settings']['expression_settings']['macaque_expression_directory']
    )
MACAQUE_SAMPLE_MATCHING_PATH = os.path.join(PROJECT_ROOT,
    SETTINGS['project_structure_settings']['metadata_settings']['metadata_directory'], 
    SETTINGS['project_structure_settings']['metadata_settings']['macaque_sample_matching_file']
    )
                                                    

def load_macaque_log2_cpm(expr_dir=MACAQUE_RESIDUALS_DIR, filtered: bool = True) -> dict[str, pd.DataFrame]:
    """
    Returns:
        dict[str, pd.DataFrame]: dictionary of monkey expression values
            (log2 cpms), keyed by region (see project_variables.MACAQUE_REGIONS).
    """
    regression_residuals_filename = (
        SETTINGS['project_structure_settings']['expression_settings']['macaque_expression_filename']['filtered']
        if filtered
        else SETTINGS['project_structure_settings']['expression_settings']['macaque_expression_filename']['no_filtered']
    )
    
    monkey_exprs_dict = {}
    for region in MACAQUE_REGIONS:
        this_exprs = pd.read_csv(os.path.join(expr_dir, region, regression_residuals_filename))
        if 'Unnamed: 0' in this_exprs.columns:
            this_exprs = this_exprs.set_index('Unnamed: 0')
            this_exprs.index.name = None
        monkey_exprs_dict[region] = this_exprs
    return monkey_exprs_dict


def load_macaque_sample_matching(path=MACAQUE_SAMPLE_MATCHING_PATH) -> pd.DataFrame:
    """
    Returns:
        pd.DataFrame: monkey-by-region sample matching table, indexed by
            monkey id. Columns are MACAQUE_REGIONS (each monkey's raw sample
            id in that region, or NaN if absent).
    """
    return pd.read_csv(path, index_col=0)



def draw_resampled_monkey_ids(monkey_ids, n: int, with_replacement: bool,
                               rng: np.random.Generator) -> list[str]:
    """One resampling draw of `n` monkey ids from `monkey_ids`.

    Args:
        monkey_ids: population to draw from -- pass
            load_macaque_sample_matching().index (or any pd.Index/array-like
            of monkey ids). Always draw from the REAL population; never pass
            an already-resampled id list back in.
        n: number of monkeys to draw.
        with_replacement: True draws a bootstrap replicate (the same monkey
            may be drawn more than once, so `n` can exceed the population
            size). False draws a subsample without replacement (`n` distinct
            monkeys; `n` must not exceed the population size).
        rng: caller-owned generator (e.g. np.random.default_rng(seed)) --
            pass an independently-seeded rng per replicate.

    Returns:
        list[str]: n monkey ids, in draw order. May contain repeats iff
            with_replacement=True.
    """
    population = np.asarray(monkey_ids)
    if not with_replacement and n > len(population):
        raise ValueError(
            f'n={n} exceeds population size {len(population)} for with_replacement=False'
        )
    return rng.choice(population, size=n, replace=with_replacement).tolist()


# 2nd+ copies of a monkey drawn more than once get this suffix + copy number on
# their sample ids, so each copy is its own column (a shared label is selected k
# times by label-based indexing, i.e. weighted k^2 instead of k).
BOOTSTRAP_COPY_SEP = '__bootstrap_copy'


def resample_sample_matching(sample_matching: pd.DataFrame, monkey_ids: list[str]) -> pd.DataFrame:
    """Row-resampled sample_matching table: one row per id in `monkey_ids`,
    in that exact order. A bootstrap draw's repeated id yields repeated
    (duplicate-index) rows; the index is left as-is, since no consumer keys
    off it. The sample ids of every 2nd+ copy are relabeled
    '<sample_id>__bootstrap_copy<j>' (j = 1, 2, ...), so each copy is a
    distinct sample label: matched_sample_ids / panel_matrix /
    calculate_synthesis_gene_expressions select by label, and a label shared
    by k copies would be picked k times per occurrence (k^2 weight). A draw
    without repeats is returned unchanged.

    Args:
        sample_matching: from load_macaque_sample_matching() -- the REAL
            table; never an already-resampled copy.
        monkey_ids: from draw_resampled_monkey_ids(). May contain repeats or
            not -- this function does not take a with_replacement parameter;
            duplicate-vs-unique row behavior is inherited entirely from
            whether `monkey_ids` itself repeats.

    Returns:
        pd.DataFrame: sample_matching.loc[monkey_ids] -- same columns,
            len(monkey_ids) rows in draw order; index may be non-unique;
            repeated copies' sample ids carry the BOOTSTRAP_COPY_SEP suffix.
            Same shape-contract as load_macaque_sample_matching()'s return
            value -- drop-in replacement.
    """
    resampled = sample_matching.loc[monkey_ids].copy()
    drawn_ids = pd.Series(list(monkey_ids))
    copy_number = drawn_ids.groupby(drawn_ids).cumcount().to_numpy()  # 0 = first copy
    if (copy_number > 0).any():
        for region in resampled.columns:
            values = resampled[region].to_numpy(dtype=object)
            relabel = (copy_number > 0) & pd.notna(values)
            values[relabel] = [f'{sample_id}{BOOTSTRAP_COPY_SEP}{j}'
                               for sample_id, j in zip(values[relabel], copy_number[relabel])]
            resampled[region] = values
    return resampled


def resample_macaque_exprs(monkey_exprs: dict[str, pd.DataFrame],
                            resampled_sample_matching: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Column-resampled expression dict: for each region, selects that
    region's resampled sample ids -- in the same row order as
    `resampled_sample_matching` -- out of the REAL, already-loaded
    `monkey_exprs`. A bootstrap draw's repeated monkey yields one column per
    copy, holding identical values: a relabeled copy
    ('<sample_id>__bootstrap_copy<j>', see resample_sample_matching) is read
    from its source column and keeps its relabeled id as the column label,
    so every column label is unique.

    Args:
        monkey_exprs: from load_macaque_log2_cpm() -- the REAL, unmodified
            per-region expression tables (unique columns); never pass an
            already-resampled dict back in.
        resampled_sample_matching: from resample_sample_matching() -- its
            row order defines the output's column order.

    Returns:
        dict[str, pd.DataFrame]: region -> expression table, columns = that
            region's non-NaN entries of resampled_sample_matching[region]
            whose source sample exists in monkey_exprs[region].columns (in
            resampled_sample_matching's row order, one unique-label column
            per copy); rows (genes) unchanged. Same keys and shape-contract
            as load_macaque_log2_cpm()'s return value -- drop-in replacement.
    """
    resampled_exprs = {}
    for region, region_exprs in monkey_exprs.items():
        region_ids = resampled_sample_matching[region].dropna()
        source_ids = region_ids.astype(str).str.split(BOOTSTRAP_COPY_SEP).str[0]
        keep = source_ids.isin(region_exprs.columns)
        selected = region_exprs[source_ids[keep].tolist()]
        selected.columns = region_ids[keep].tolist()
        resampled_exprs[region] = selected
    return resampled_exprs


if __name__ == '__main__':
    monkey_exprs = load_macaque_log2_cpm()
    print(monkey_exprs.keys())
    print(load_macaque_sample_matching().shape)
