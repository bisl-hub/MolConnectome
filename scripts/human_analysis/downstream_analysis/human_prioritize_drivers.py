"""Human stage 5: driver genes for every connection flagged by stage 3.

Input:  stage 3's delta_quantile_flagged.csv, stage 2's gene_contribution_diff_full.csv, and the
        human expression (gene-level changes are recomputed for the flagged connections).
Output: label_shuffle_summary/<JOB>/driver_gene_prioritization.csv
Usage:  see HUMAN_PIPELINE.md, section 4.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys
from pathlib import Path

import pandas as pd
import yaml

with open(Path(__file__).resolve().parent.parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'human_analysis'))

import human_corr_contrib as cc
import human_corr_landscape as landscape
from src.project_variables import NT_TO_GENE_SET
from src.connection_analyzer import _is_direction_consistent


DEFAULT_JOB_NAME = SETTINGS['human_label_shuffle_wrapper']['initial_job_name']
DEFAULT_FLAGGED_PATH = os.path.join(
    PROJECT_ROOT, SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_delta_quantile_report']['report_save_dir'], DEFAULT_JOB_NAME,
    SETTINGS['human_delta_quantile_report']['delta_quantile_flagged_filename'],
)
# human_corr_landscape.py's own full-sweep output -- the fixed, job-independent
# gene-connection-instance population the global |PCT_DIFF| quantile threshold
# is drawn from (computed once from the real, unshuffled expression data, so
# it doesn't vary with which label-shuffle job's flagged report is processed).
DEFAULT_LANDSCAPE_DIFF_PATH = os.path.join(landscape.DEFAULT_OUTPUT_DIR, landscape.GENE_ROLLUP_DIFF_FILENAME)

TOP_QUANTILE = SETTINGS['human_prioritize_drivers']['top_quantile']
OUTPUT_FILENAME = SETTINGS['human_prioritize_drivers']['output_filename']


def compute_global_threshold(top_quantile: float = TOP_QUANTILE,
                             landscape_diff_path: str = DEFAULT_LANDSCAPE_DIFF_PATH) -> float:
    """Minimum |PCT_DIFF| a gene-connection instance needs to sit in the top
    `top_quantile` share of the full landscape sweep -- e.g. top_quantile=0.3
    means the value at the 70th percentile of |PCT_DIFF| across every gene x
    role x gene_set x directed-region-pair instance in the full sweep.
    PCT_DIFF (not raw DIFF) is used so gene sets of very different sizes
    (GENE_SET_TO_USE's n_cat1/n_cat2 range from single digits to the low 20s)
    are compared on a common, size-independent scale -- see
    human_corr_contrib.compute_gene_diff."""
    if not os.path.exists(landscape_diff_path):
        raise FileNotFoundError(
            f'{landscape_diff_path} not found -- run human_corr_landscape.py first to generate '
            f'the full-sweep gene DIFF landscape this quantile threshold is drawn from.'
        )
    full_gene_diff = pd.read_csv(landscape_diff_path)
    return float(full_gene_diff['PCT_DIFF'].abs().quantile(1 - top_quantile))


def compute_local_threshold(gene_diff_combo: pd.DataFrame, role: str, top_quantile: float = TOP_QUANTILE) -> float:
    """Like compute_global_threshold, but scoped to just ONE connection's own
    gene pool for the given role -- top_quantile=0.3 means the value at the
    70th percentile of |PCT_DIFF| among just this combo's own sender (or
    receiver) genes, instead of the fixed cross-connection landscape bar.
    Recomputed fresh per (combo, role): a connection's own top 30% is only
    ever compared against its own genes, never another connection's."""
    subset = gene_diff_combo[gene_diff_combo['role'] == role]
    return float(subset['PCT_DIFF'].abs().quantile(1 - top_quantile))


def _sign_matches_status(pct_diff_value: float, status: str) -> bool:
    """hyperactive: a driver gene must have GROWN its share of the total in
    SCZ (PCT_DIFF > 0). hypoactive: a driver gene must have SHRUNK its share
    in SCZ (PCT_DIFF < 0) -- the sign half of what makes a gene a 'driver'."""
    if status == 'hyperactive':
        return pct_diff_value > 0
    if status == 'hypoactive':
        return pct_diff_value < 0
    return False


def prioritized_genes(gene_diff_combo: pd.DataFrame, role: str, status: str, threshold: float) -> list[str]:
    """Genes of the given role (sender/receiver) within one combo's gene_diff
    rows that are both large enough (|PCT_DIFF| >= threshold) and signed
    consistently with `status`, ranked by |PCT_DIFF| descending."""
    subset = gene_diff_combo[gene_diff_combo['role'] == role]
    matches_sign = subset['PCT_DIFF'].apply(lambda d: _sign_matches_status(d, status))
    qualifying = subset[(subset['PCT_DIFF'].abs() >= threshold) & matches_sign]
    qualifying = qualifying.loc[qualifying['PCT_DIFF'].abs().sort_values(ascending=False).index]
    return qualifying['gene'].tolist()


def merge_prioritized(global_genes: list[str], connectionwise_genes: list[str]) -> list[str]:
    """Union of both methods' picks -- a gene counts as a driver if EITHER
    the global or the connection-wise threshold flags it. Order: global's
    own |PCT_DIFF|-descending order first, then any additional genes only
    the connection-wise method found, in its own order -- not re-sorted into
    one shared ranking, since a gene that clears only one of the two
    thresholds isn't on a directly comparable scale to one that clears both."""
    return list(dict.fromkeys(global_genes + connectionwise_genes))


def main(flagged_path: str = DEFAULT_FLAGGED_PATH,
         output_dir: str = None,
         top_quantile: float = TOP_QUANTILE,
         landscape_diff_path: str = DEFAULT_LANDSCAPE_DIFF_PATH,
         p_value_method: str = cc.DEFAULT_P_VALUE_METHOD) -> pd.DataFrame:
    # accepts either a delta_quantile_flagged.csv path or its job directory
    # (see human_corr_contrib.resolve_flagged_path) -- same convention.
    flagged_path = cc.resolve_flagged_path(flagged_path)
    # written directly into the flagged report's own job directory -- no
    # corr_contrib-style subfolder, per the requested output location.
    if output_dir is None:
        output_dir = os.path.dirname(flagged_path)

    flagged = pd.read_csv(flagged_path)
    combos = cc.load_flagged_combos(flagged_path)
    if not combos:
        raise ValueError(f'No flagged combos found in {flagged_path}')

    _, _, _, gene_diff = cc.compute_all(combos, p_value_method)
    global_threshold = compute_global_threshold(top_quantile, landscape_diff_path)
    print(f'top_quantile={top_quantile} -> global |PCT_DIFF| threshold={global_threshold:.6g} (from {landscape_diff_path})')

    sender_global, receiver_global = [], []
    sender_connectionwise, receiver_connectionwise = [], []
    for row in flagged.itertuples(index=False):
        gene_set = NT_TO_GENE_SET[row.neurotransmitter]
        combo_diff = gene_diff[
            (gene_diff['gene_set'] == gene_set)
            & (gene_diff['sender_region'] == row.sender_region)
            & (gene_diff['receiver_region'] == row.receiver_region)
        ]
        sender_global.append(prioritized_genes(combo_diff, 'sender', row.status, global_threshold))
        receiver_global.append(prioritized_genes(combo_diff, 'receiver', row.status, global_threshold))

        # Each connection's own top_quantile, recomputed fresh per (combo,
        # role) -- NOT the fixed global_threshold above.
        local_sender_threshold = compute_local_threshold(combo_diff, 'sender', top_quantile)
        local_receiver_threshold = compute_local_threshold(combo_diff, 'receiver', top_quantile)
        sender_connectionwise.append(prioritized_genes(combo_diff, 'sender', row.status, local_sender_threshold))
        receiver_connectionwise.append(prioritized_genes(combo_diff, 'receiver', row.status, local_receiver_threshold))

    sender_merged = [merge_prioritized(g, c) for g, c in zip(sender_global, sender_connectionwise)]
    receiver_merged = [merge_prioritized(g, c) for g, c in zip(receiver_global, receiver_connectionwise)]

    result = flagged.copy()
    result['sender_prioritized'] = sender_merged
    result['receiver_prioritized'] = receiver_merged
    result['sender_prioritized_global'] = sender_global
    result['receiver_prioritized_global'] = receiver_global
    result['sender_prioritized_connectionwise'] = sender_connectionwise
    result['receiver_prioritized_connectionwise'] = receiver_connectionwise
    result['is_direction_consistent'] = flagged.apply(
        lambda row: _is_direction_consistent(
            row['status'], row['sender_abundance_direction'], row['sender_abundance_significance']),
        axis=1)

    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(output_path, index=False)

    n_rows = len(flagged)
    print(f'{n_rows} flagged rows -- '
          f'global: {sum(1 for g in sender_global if g)} w/ sender driver, '
          f'{sum(1 for g in receiver_global if g)} w/ receiver driver; '
          f'connection-wise: {sum(1 for g in sender_connectionwise if g)} w/ sender driver, '
          f'{sum(1 for g in receiver_connectionwise if g)} w/ receiver driver; '
          f'merged: {sum(1 for g in sender_merged if g)} w/ sender driver, '
          f'{sum(1 for g in receiver_merged if g)} w/ receiver driver -- wrote {output_path}')
    return result


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FLAGGED_PATH,
        sys.argv[2] if len(sys.argv) > 2 else None,
        float(sys.argv[3]) if len(sys.argv) > 3 else TOP_QUANTILE,
        sys.argv[4] if len(sys.argv) > 4 else DEFAULT_LANDSCAPE_DIFF_PATH,
        sys.argv[5] if len(sys.argv) > 5 else cc.DEFAULT_P_VALUE_METHOD,
    )
