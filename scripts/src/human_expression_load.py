"""Loads the human expression data and sample-matching table, groups samples by region, cohort
and diagnosis, and shuffles diagnosis labels for the label-shuffle null.

Used by: human_abundance.py, human_oneshot.py, human_corr_contrib.py,
         human_group_abundance_compare.py, human_donor_split.py.
"""

import pandas as pd
import os
import sys
import numpy as np

from pathlib import Path
import yaml
with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)
PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))


HUMAN_RESIDUALS_DIR =  os.path.join(PROJECT_ROOT, 
    SETTINGS['project_structure_settings']['expression_settings']['human_expression_directory']
    )
HUMAN_SAMPLE_MATCHING_TABLE_PATH =  os.path.join(PROJECT_ROOT,
    SETTINGS['project_structure_settings']['metadata_settings']['metadata_directory'] ,
    SETTINGS['project_structure_settings']['metadata_settings']['human_sample_matching_file']
    )


from project_variables import (
    HUMAN_REGIONS,
    HUMAN_DATASET_ABBR,
    DISEASE_ABBR,
    HUMAN_DATASET_REGION_TO_RESIDUALS_FOLDER
    )


def load_human_residuals() -> dict[str, pd.DataFrame]:
    """
    Returns:
        dict[str, pd.DataFrame]: dictionary of human expression values (log2 cpms),
            keyed 'DATASET#REGION' (e.g. 'BS#PFC'); columns are that dataset's
            own raw sample ids, exactly as they appear in the matching table.
            Dataset-region folders absent from disk (e.g. controlled-access
            cohorts in the public toy example) are skipped with a printed notice.
    """
    regression_residuals_filename = \
        SETTINGS['project_structure_settings']['expression_settings']['human_expression_filename']
    human_exprs_dict = {}
    skipped = []
    for x in HUMAN_DATASET_REGION_TO_RESIDUALS_FOLDER:
        folder_name = HUMAN_DATASET_REGION_TO_RESIDUALS_FOLDER[x]
        path = os.path.join(HUMAN_RESIDUALS_DIR, folder_name, regression_residuals_filename)
        if not os.path.exists(path):
            skipped.append(x)
            continue
        this_exprs = pd.read_csv(path)
        if 'Unnamed: 0' in this_exprs.columns:
            this_exprs = this_exprs.set_index('Unnamed: 0')
            this_exprs.index.name = None
        human_exprs_dict[x] = this_exprs
    if skipped:
        print(f'load_human_residuals: {len(skipped)} dataset-region folder(s) not on disk, skipped: {skipped}')
    return human_exprs_dict


def load_sample_matching_table() -> pd.DataFrame:
    """
    Returns:
        pd.DataFrame: metadata/human_sample_matching_table.csv, indexed by
            donor_id. Columns are HUMAN_REGIONS (each donor's raw sample id in
            that region's residuals file, or NaN if absent for that donor)
            plus 'disease' (Control/Schizophrenia).
    """
    return pd.read_csv(HUMAN_SAMPLE_MATCHING_TABLE_PATH, index_col='donor_id')


def separate_donors_by_disease(sample_matching: pd.DataFrame) -> dict[str, pd.Index]:
    """Separates donors into control and schizophrenia groups.

    Args:
        sample_matching: table from load_sample_matching_table().

    Returns:
        dict[str, pd.Index]: DISEASE_ABBR value ('CTL'/'SCZ') -> donor ids.
    """
    return {
        DISEASE_ABBR[disease]: sample_matching.index[sample_matching['disease'] == disease]
        for disease in DISEASE_ABBR
    }


def merge_human_expression_by_region(human_exprs_dict: dict[str, pd.DataFrame],
                                     sample_matching: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Merges every dataset's per-region residuals table (on the column axis)
    into a single table per region, renaming each dataset's raw sample id
    columns to the universal donor_id so samples stay unique/comparable
    across cohorts.

    Args:
        human_exprs_dict: from load_human_residuals(), keyed 'DATASET#REGION'.
        sample_matching: from load_sample_matching_table().

    Returns:
        dict[str, pd.DataFrame]: region -> expression table (genes x donor_id),
            pooling every dataset that has data for that region.
    """
    merged_by_region = {}
    for region in HUMAN_REGIONS:
        region_sample_ids = sample_matching[region].dropna()
        sample_id_to_donor_id = pd.Series(region_sample_ids.index, index=region_sample_ids.values).to_dict()

        region_frames = []
        for key, dataset_region_exprs in human_exprs_dict.items():
            if not key.endswith(f'#{region}'):
                continue
            matched_columns = dataset_region_exprs.columns.intersection(sample_id_to_donor_id.keys())
            region_frames.append(dataset_region_exprs[matched_columns].rename(columns=sample_id_to_donor_id))
        merged_by_region[region] = pd.concat(region_frames, axis=1)
    return merged_by_region


def split_region_expression_by_disease(merged_by_region: dict[str, pd.DataFrame],
                                       disease_groups: dict[str, pd.Index]) -> dict[str, dict[str, pd.DataFrame]]:
    """Splits each region's merged expression table into disease groups.

    Args:
        merged_by_region: from merge_human_expression_by_region().
        disease_groups: from separate_donors_by_disease().

    Returns:
        dict[str, dict[str, pd.DataFrame]]: region -> {'CTL': df, 'SCZ': df},
            each restricted to the donor ids of that disease group that are
            actually present in that region.
    """
    return {
        region: {
            disease: exprs[exprs.columns.intersection(donor_ids)]
            for disease, donor_ids in disease_groups.items()
        }
        for region, exprs in merged_by_region.items()
    }


def split_dataset_specific_expression_by_disease(human_exprs_dict: dict[str, pd.DataFrame],
                                                 sample_matching: pd.DataFrame,
                                                 disease_groups: dict[str, pd.Index]) -> dict[str, dict[str, pd.DataFrame]]:
    """Splits each dataset's own per-region expression table (raw sample id
    columns, no cross-cohort merge) into disease groups, using
    `sample_matching` to translate each dataset's raw sample ids to the donor
    ids that key disease_groups.

    Args:
        human_exprs_dict: from load_human_residuals(), keyed 'DATASET#REGION'.
        sample_matching: from load_sample_matching_table().
        disease_groups: from separate_donors_by_disease().

    Returns:
        dict[str, dict[str, pd.DataFrame]]: 'DATASET#REGION' -> {'CTL': df, 'SCZ': df},
            each restricted to the raw sample ids whose donor belongs to that
            disease group and is present in that dataset-region.
    """
    split_by_dataset_region = {}
    for key, dataset_region_exprs in human_exprs_dict.items():
        _, region = key.split('#')
        region_sample_ids = sample_matching[region].dropna()
        sample_id_by_donor_id = pd.Series(region_sample_ids.values, index=region_sample_ids.index)

        split_by_dataset_region[key] = {
            disease: dataset_region_exprs[
                dataset_region_exprs.columns.intersection(sample_id_by_donor_id.reindex(donor_ids).dropna())
            ]
            for disease, donor_ids in disease_groups.items()
        }
    return split_by_dataset_region


def load_human_expression_by_region_and_disease(sample_matching: pd.DataFrame = None) -> dict[str, dict[str, pd.DataFrame]]:
    """Full pipeline: loads residuals and the sample matching table, merges
    every cohort's table per region (column axis), then separates each
    region's merged table into control/schizophrenia.

    Args:
        sample_matching: optional override (e.g. a shuffle_disease_labels()
            replicate) used in place of load_sample_matching_table().

    Returns:
        dict[str, dict[str, pd.DataFrame]]: region -> {'CTL': df, 'SCZ': df}.
    """
    human_exprs_dict = load_human_residuals()
    if sample_matching is None:
        sample_matching = load_sample_matching_table()
    disease_groups = separate_donors_by_disease(sample_matching)
    merged_by_region = merge_human_expression_by_region(human_exprs_dict, sample_matching)
    return split_region_expression_by_disease(merged_by_region, disease_groups)


def load_human_expression_by_dataset_and_disease(sample_matching: pd.DataFrame = None) -> dict[str, dict[str, pd.DataFrame]]:
    """full pipeline: loads residuals and the sample matching table, no merge (per dataset),
    separates each dataset into control/schizophrenia.

    Args:
        sample_matching: optional override (e.g. a shuffle_disease_labels()
            replicate) used in place of load_sample_matching_table().

    Returns:
        dict[str, dict[str, pd.DataFrame]]: dataset#region -> {'CTL': df, 'SCZ': df},
            each restricted to the donor ids of that disease group that are
            actually present in that dataset.
    """
    human_exprs_dict = load_human_residuals()
    if sample_matching is None:
        sample_matching = load_sample_matching_table()
    disease_groups = separate_donors_by_disease(sample_matching)
    return split_dataset_specific_expression_by_disease(human_exprs_dict, sample_matching, disease_groups)


def _dataset_abbr_of_donor(donor_id: str) -> str:
    """BS_/HBCC_/GSE53987_/GSE80655_ donor ids carry an explicit prefix; the
    3 SMRI collections (AC/NPC/NC) use their raw sample id as donor_id
    directly (A<code>/C<code>/M<code>)."""
    for abbr in ('BS', 'HBCC', 'GSE53987', 'GSE80655'):
        if donor_id.startswith(f'{abbr}_'):
            return abbr
    return {'A': 'AC', 'C': 'NPC', 'M': 'NC'}[donor_id[0]]


def _coverage_pattern(row: pd.Series) -> tuple[str, ...]:
    return tuple(region for region in HUMAN_REGIONS if pd.notna(row[region]))


def shuffle_disease_labels(sample_matching: pd.DataFrame, rng: np.random.Generator,
                          stratified: bool = True) -> pd.DataFrame:
    """One permutation replicate of the disease column.

    By default (stratified=True), permutes within (dataset,
    region-coverage-pattern) blocks so every region's, every region-pair's,
    and every dataset-specific real CTL/SCZ count is preserved exactly. A
    single unstratified (global) shuffle instead draws, for any such subset,
    a hypergeometric sample centered on the *global* CTL:SCZ ratio --
    injecting spurious variance/bias into the null distribution instead of
    reflecting the real per-subset composition (per-dataset ratios range
    from ~50:50 to ~61:39, and vary further by region-coverage pattern
    within a dataset). Always call on the real table from
    load_sample_matching_table(), never on an already-shuffled copy.

    Args:
        sample_matching: from load_sample_matching_table().
        rng: caller-owned generator (e.g. np.random.default_rng(seed)) --
            pass an independently-seeded rng per replicate.
        stratified: if True (default), permute within (dataset,
            region-coverage-pattern) blocks. If False, permute the whole
            'disease' column as a single global block instead -- kept for
            comparison/diagnostic use; it does NOT preserve per-region or
            per-dataset CTL:SCZ counts.

    Returns:
        pd.DataFrame: copy of sample_matching with 'disease' permuted;
            drop-in replacement for every other function in this module.
    """
    if stratified:
        block_keys = pd.Series(
            [(_dataset_abbr_of_donor(donor_id), _coverage_pattern(row)) for donor_id, row in sample_matching.iterrows()],
            index=sample_matching.index,
        )
    else:
        block_keys = pd.Series(0, index=sample_matching.index)
    shuffled = sample_matching.copy()
    for _, donor_ids in shuffled.groupby(block_keys).groups.items():
        shuffled.loc[donor_ids, 'disease'] = rng.permutation(sample_matching.loc[donor_ids, 'disease'].to_numpy())
    return shuffled


def resolve_sample_matching(label_shuffle_seed: int = None) -> pd.DataFrame:
    """The matching table one run uses: the real one for label_shuffle_seed=None,
    else that seed's stratified shuffle. human_abundance.py and human_oneshot.py
    both call this, so a given seed gives a replicate's abundance and its
    correlations the same shuffled labels.

    Args:
        label_shuffle_seed: None for the real labels, or an integer seed for
            np.random.default_rng.

    Returns:
        pd.DataFrame: same layout as load_sample_matching_table().
    """
    real_sample_matching = load_sample_matching_table()
    if label_shuffle_seed is None:
        return real_sample_matching
    rng = np.random.default_rng(label_shuffle_seed)
    return shuffle_disease_labels(real_sample_matching, rng, stratified=True)


if __name__ == '__main__':
    region_disease_exprs = load_human_expression_by_region_and_disease()
    print(region_disease_exprs.keys())
    dataset_disease_exprs = load_human_expression_by_dataset_and_disease()
    print(dataset_disease_exprs.keys())