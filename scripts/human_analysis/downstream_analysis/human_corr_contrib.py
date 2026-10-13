"""Human stage 4: gene-level contributions for the connections flagged by stage 3.

Input:  stage 3's delta_quantile_flagged.csv; human expression and sample-matching table.
Output: label_shuffle_summary/<JOB>/corr_contrib/ (gene-pair and per-gene contribution tables)
Usage:  see HUMAN_PIPELINE.md, section 4.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

with open(Path(__file__).resolve().parent.parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'human_analysis'))

import src.correlation as cor
import src.quantile as qu
from src.human_expression_load import (
    load_sample_matching_table,
    load_human_expression_by_region_and_disease,
)
from src.project_variables import DISEASES, NT_TO_GENE_SET
import human_oneshot


DEFAULT_P_VALUE_METHOD = human_oneshot.DEFAULT_P_VALUE_METHOD

DEFAULT_JOB_NAME = SETTINGS['human_label_shuffle_wrapper']['initial_job_name']
DEFAULT_FLAGGED_PATH = os.path.join(
    PROJECT_ROOT, SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_delta_quantile_report']['report_save_dir'], DEFAULT_JOB_NAME,
    SETTINGS['human_delta_quantile_report']['delta_quantile_flagged_filename'],
)
GENE_PAIR_FILENAME = SETTINGS['human_corr_contrib']['gene_pair_filename']
GENE_ROLLUP_FILENAME = SETTINGS['human_corr_contrib']['gene_rollup_filename']
GENE_PAIR_DIFF_FILENAME = SETTINGS['human_corr_contrib']['gene_pair_diff_filename']
GENE_ROLLUP_DIFF_FILENAME = SETTINGS['human_corr_contrib']['gene_rollup_diff_filename']


def resolve_flagged_path(flagged_path: str) -> str:
    """Accept either a delta_quantile_flagged.csv file path directly, or the
    job directory containing one (e.g. label_shuffle_summary/<JOB_NAME>) --
    the latter is the natural thing to pass when switching jobs. On Windows,
    pointing pandas.read_csv at a bare directory fails with a misleading
    PermissionError (open() on a directory raises EACCES on Windows, not
    IsADirectoryError like on Linux) instead of a clear 'not a file'
    message, so resolve it to the file inside instead of passing it through."""
    if os.path.isdir(flagged_path):
        return os.path.join(flagged_path, SETTINGS['human_delta_quantile_report']['delta_quantile_flagged_filename'])
    return flagged_path


def load_flagged_combos(flagged_path: str = DEFAULT_FLAGGED_PATH) -> list[dict]:
    """One dict per unique (gene_set, neurotransmitter, sender_region,
    receiver_region) named in a delta_quantile_flagged.csv row -- its
    status/disease-comparison columns aren't needed here, only which combo to
    recompute the raw correlation block for."""
    flagged = pd.read_csv(resolve_flagged_path(flagged_path))
    combos = flagged[['neurotransmitter', 'sender_region', 'receiver_region']].drop_duplicates()
    return [
        {
            'gene_set': NT_TO_GENE_SET[row.neurotransmitter],
            'neurotransmitter': row.neurotransmitter,
            'sender_region': row.sender_region,
            'receiver_region': row.receiver_region,
        }
        for row in combos.itertuples(index=False)
    ]


def _spec_by_key() -> tuple[list[str], dict]:
    final_genes = cor.load_final_genes()
    specs = qu._load_gene_set_specs(final_genes)
    return final_genes, {spec['key']: spec for spec in specs}


def gene_pair_contribution(r: np.ndarray, p: np.ndarray, sender_genes: list[str], receiver_genes: list[str]) -> pd.DataFrame:
    """Long-form sender_gene x receiver_gene contribution table. `contribution`
    is `r` where the pair qualifies for human_oneshot.py's qualifying sum
    (r > 0), else 0 -- the same rule src/quantile.py's
    _non_filtered_positive_count_and_sum applies; summing this column
    reproduces human_oneshot.py's real_value for this exact cell."""
    mask = r > 0
    contribution = np.where(mask, r, 0.0)
    n_sender, n_receiver = r.shape
    return pd.DataFrame({
        'sender_gene': np.repeat(sender_genes, n_receiver),
        'receiver_gene': np.tile(receiver_genes, n_sender),
        'r': r.ravel(),
        'p_value': p.ravel(),
        'qualifying': mask.ravel(),
        'contribution': contribution.ravel(),
    })


def gene_rollup(pair_df: pd.DataFrame) -> pd.DataFrame:
    """One row per gene per role (sender/receiver): its contributions summed
    across every partner in the other role, its qualifying-partner count, and
    its share of this combo's total qualifying sum -- ranked descending
    within each role."""
    total = pair_df['contribution'].sum()
    rollups = []
    for role, gene_col in (('sender', 'sender_gene'), ('receiver', 'receiver_gene')):
        grouped = pair_df.groupby(gene_col).agg(
            total_contribution=('contribution', 'sum'),
            n_qualifying_partners=('qualifying', 'sum'),
            n_partners=('qualifying', 'size'),
        ).reset_index().rename(columns={gene_col: 'gene'})
        grouped['role'] = role
        grouped['pct_of_total'] = grouped['total_contribution'] / total if total else np.nan
        grouped = grouped.sort_values('total_contribution', ascending=False).reset_index(drop=True)
        grouped['rank'] = grouped.index + 1
        rollups.append(grouped)
    return pd.concat(rollups, ignore_index=True)[
        ['role', 'gene', 'total_contribution', 'pct_of_total', 'n_qualifying_partners', 'n_partners', 'rank']]


def compute_pair_diff(pair_all: pd.DataFrame) -> pd.DataFrame:
    """SCZ - CTL contribution per (gene_set, sender_region, receiver_region,
    sender_gene, receiver_gene). The sender/receiver gene grid for a given
    combo is fixed by gene-set membership, not by disease, so CTL and SCZ
    always share the exact same set of (sender_gene, receiver_gene) keys --
    this is a plain aligned subtraction, never a mismatched-key join."""
    id_cols = ['gene_set', 'neurotransmitter', 'sender_region', 'receiver_region', 'sender_gene', 'receiver_gene']
    ctl = pair_all[pair_all['disease'] == 'CTL'].set_index(id_cols)
    scz = pair_all[pair_all['disease'] == 'SCZ'].set_index(id_cols)
    diff = pd.DataFrame({
        'r_CTL': ctl['r'], 'r_SCZ': scz['r'],
        'contribution_CTL': ctl['contribution'], 'contribution_SCZ': scz['contribution'],
    })
    diff['DIFF'] = diff['contribution_SCZ'] - diff['contribution_CTL']
    return diff.reset_index()


def compute_gene_diff(rollup_all: pd.DataFrame) -> pd.DataFrame:
    """SCZ - CTL total_contribution AND pct_of_total per (gene_set,
    sender_region, receiver_region, role, gene) -- same fixed-gene-grid
    reasoning as compute_pair_diff, at the rolled-up-per-gene level.

    DIFF (raw total_contribution_SCZ - total_contribution_CTL) is NOT
    comparable across gene sets: a gene's raw total_contribution is a sum
    over its partner genes in the other role, so it scales with that other
    role's gene-set size (GENE_SET_TO_USE's n_cat1/n_cat2 range from single
    digits to the low 20s) -- a gene from a large gene set can out-rank a
    truly stronger driver from a small one purely from having more terms to
    sum, not genuine effect size.

    PCT_DIFF (pct_of_total_SCZ - pct_of_total_CTL) instead compares each
    gene's SHARE of its own combo's total qualifying sum between diseases --
    bounded in [-1, 1] regardless of gene-set size, and answers 'did this
    gene's relative importance within its own connection grow or shrink',
    which is what makes it comparable across gene sets/combos. This -- not
    DIFF -- is the metric downstream driver-prioritization should rank/
    threshold on."""
    id_cols = ['gene_set', 'neurotransmitter', 'sender_region', 'receiver_region', 'role', 'gene']
    ctl = rollup_all[rollup_all['disease'] == 'CTL'].set_index(id_cols)
    scz = rollup_all[rollup_all['disease'] == 'SCZ'].set_index(id_cols)
    diff = pd.DataFrame({
        'total_contribution_CTL': ctl['total_contribution'],
        'total_contribution_SCZ': scz['total_contribution'],
        'pct_of_total_CTL': ctl['pct_of_total'],
        'pct_of_total_SCZ': scz['pct_of_total'],
    })
    diff['DIFF'] = diff['total_contribution_SCZ'] - diff['total_contribution_CTL']
    diff['PCT_DIFF'] = diff['pct_of_total_SCZ'] - diff['pct_of_total_CTL']
    diff = diff.reset_index().sort_values(
        ['gene_set', 'sender_region', 'receiver_region', 'role', 'PCT_DIFF'],
        ascending=[True, True, True, True, False],
    )
    return diff.reset_index(drop=True)


def compute_combo_contribution(combo: dict, final_genes: list[str], spec: dict, disease: str,
                               human_exprs_by_region: dict, sample_matching: pd.DataFrame,
                               p_value_method: str = DEFAULT_P_VALUE_METHOD) -> tuple[pd.DataFrame, pd.DataFrame]:
    if p_value_method == 'permutation':
        raise ValueError(
            "p_value_method='permutation' is deprecated for this pipeline (see HUMAN_PIPELINE.md, "
            "section 7) -- use 'transformation'."
        )
    sender_region, receiver_region = combo['sender_region'], combo['receiver_region']
    sender_genes = [final_genes[i] for i in spec['cat2_idx']]  # cat2 = sender genes
    receiver_genes = [final_genes[i] for i in spec['cat1_idx']]  # cat1 = receiver genes

    human_exprs = {
        sender_region: human_exprs_by_region[sender_region][disease],
        receiver_region: human_exprs_by_region[receiver_region][disease],
    }
    # human_oneshot.py keys sender/receiver on the SAME donor_id (not
    # cor.matched_sample_ids' macaque-style raw-sample-id matching) -- reuse
    # its own helper so the matched donor set is identical to what produced
    # the real_value this module is decomposing.
    sample_ids_sender, sample_ids_receiver = human_oneshot._matched_donor_ids(
        human_exprs, sample_matching, sender_region, receiver_region)

    X = cor.panel_matrix(human_exprs, sender_region, sender_genes, sample_ids_sender)
    Y = cor.panel_matrix(human_exprs, receiver_region, receiver_genes, sample_ids_receiver)
    r, p, n_donors = cor.cross_correlation_and_pvalues(X, Y)

    pair_df = gene_pair_contribution(r, p, sender_genes, receiver_genes)
    rollup_df = gene_rollup(pair_df)
    for key in ('gene_set', 'neurotransmitter', 'sender_region', 'receiver_region'):
        pair_df[key] = combo[key]
        rollup_df[key] = combo[key]
    pair_df['disease'] = disease
    pair_df['n_donors'] = n_donors
    rollup_df['disease'] = disease

    return pair_df, rollup_df


def compute_all(combos: list[dict],
                p_value_method: str = DEFAULT_P_VALUE_METHOD) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pure computation, no file I/O: (pair_all, rollup_all, pair_diff,
    gene_diff) for every combo x DISEASES. Loads final_genes/gene-set specs/
    sample matching/expression ONCE and reuses them across every combo --
    shared, callable by any caller with its own combo list (e.g. a flagged-
    report subset, or every gene_set x directed region pair for a full
    sweep), not just main()'s own flagged-report path."""
    final_genes, specs_by_key = _spec_by_key()
    sample_matching = load_sample_matching_table()
    human_exprs_by_region = load_human_expression_by_region_and_disease(sample_matching=sample_matching)

    pair_frames, rollup_frames = [], []
    for combo in combos:
        spec = specs_by_key[combo['gene_set']]
        for disease in DISEASES:
            pair_df, rollup_df = compute_combo_contribution(
                combo, final_genes, spec, disease, human_exprs_by_region, sample_matching,
                p_value_method)
            pair_frames.append(pair_df)
            rollup_frames.append(rollup_df)
        print(f"{combo['gene_set']} {combo['sender_region']}->{combo['receiver_region']}: done")

    pair_all = pd.concat(pair_frames, ignore_index=True)
    rollup_all = pd.concat(rollup_frames, ignore_index=True)
    pair_diff = compute_pair_diff(pair_all)
    gene_diff = compute_gene_diff(rollup_all)
    return pair_all, rollup_all, pair_diff, gene_diff


def main(flagged_path: str = DEFAULT_FLAGGED_PATH,
         output_dir: str = None,
         p_value_method: str = DEFAULT_P_VALUE_METHOD) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flagged_path = resolve_flagged_path(flagged_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(flagged_path), SETTINGS['human_corr_contrib']['report_save_dir'])

    combos = load_flagged_combos(flagged_path)
    if not combos:
        raise ValueError(f'No flagged combos found in {flagged_path}')

    pair_all, rollup_all, pair_diff, gene_diff = compute_all(combos, p_value_method)

    pair_path = os.path.join(output_dir, GENE_PAIR_FILENAME)
    rollup_path = os.path.join(output_dir, GENE_ROLLUP_FILENAME)
    pair_diff_path = os.path.join(output_dir, GENE_PAIR_DIFF_FILENAME)
    gene_diff_path = os.path.join(output_dir, GENE_ROLLUP_DIFF_FILENAME)
    os.makedirs(output_dir, exist_ok=True)
    pair_all.to_csv(pair_path, index=False, float_format='%.6g', compression='gzip')
    rollup_all.to_csv(rollup_path, index=False, float_format='%.6g')
    pair_diff.to_csv(pair_diff_path, index=False, float_format='%.6g', compression='gzip')
    gene_diff.to_csv(gene_diff_path, index=False, float_format='%.6g')

    print(f'{len(combos)} flagged combos x {len(DISEASES)} diseases: '
          f'wrote {len(pair_all)} gene-pair rows to {pair_path}, '
          f'{len(rollup_all)} gene-rollup rows to {rollup_path}, '
          f'{len(pair_diff)} gene-pair DIFF rows to {pair_diff_path}, '
          f'{len(gene_diff)} gene DIFF rows to {gene_diff_path}')
    return pair_all, rollup_all, pair_diff, gene_diff


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FLAGGED_PATH,
        sys.argv[2] if len(sys.argv) > 2 else None,
        sys.argv[3] if len(sys.argv) > 3 else DEFAULT_P_VALUE_METHOD,
    )
