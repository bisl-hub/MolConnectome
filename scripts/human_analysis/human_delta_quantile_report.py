"""Human stage 3: tests every SCZ - CTL connectivity difference against the label-shuffle null.

Input:  stage 1's unshuffled/ run, every run_NNNNNN folder of the job, and P1's
        group_abundance_stats.csv.
Output: human_analysis_results/label_shuffle_summary/<JOB>/delta_quantile_report.csv and
        delta_quantile_flagged.csv
Usage:  see HUMAN_PIPELINE.md, section 4.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys
import re
import glob

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import scipy.stats

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

import src.quantile as qu
import human_oneshot
import human_weighted_connectivity
import human_label_shuffle_wrapper
from src.project_variables import HUMAN_REGIONS, GENE_SET_TO_USE, NT_TO_GENE_SET, DISEASES

DEFAULT_UNSHUFFLED_DIR = human_label_shuffle_wrapper.DEFAULT_UNSHUFFLED_DIR
DEFAULT_JOB_DIR = os.path.join(human_label_shuffle_wrapper.QUANTILE_SAVE_PATH,
                               human_label_shuffle_wrapper.JOB_NAME)
# per-(region, neurotransmitter) SCZ-vs-CTL abundance stats from
# prerequisite/human_group_abundance_compare.py, joined by (sender_region, neurotransmitter).
DEFAULT_GROUP_ABUNDANCE_STATS_PATH = os.path.join(
    PROJECT_ROOT, SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_group_abundance_compare']['report_save_dir'],
    SETTINGS['human_group_abundance_compare']['stats_filename'])

# quantile_weighted's two-tailed flag thresholds -- based on quantile_weighted
# only (not quantile_non_weighted), per the confirmed scope of this flag.
HYPOACTIVE_QUANTILE_THRESHOLD = SETTINGS['human_delta_quantile_report']['hypoactive_quantile_threshold']
HYPERACTIVE_QUANTILE_THRESHOLD = SETTINGS['human_delta_quantile_report']['hyperactive_quantile_threshold']

DEFAULT_REPORT_SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_delta_quantile_report']['report_save_dir'],
)

# Epsilon for "is this real value structurally zero" checks (e.g. a
# combination's SCZ_weighted/CTL_weighted are both exactly 0.0 because that
# sender region has zero measured abundance for this neurotransmitter) --
# deliberately a tolerance rather than an exact ==0.0 comparison.
FDR_EPSILON = SETTINGS['human_delta_quantile_report']['structural_zero_epsilon']


def _activity_status(quantile_weighted: float) -> str:
    """quantile_weighted's rank against the shuffled null, thresholded at
    both tails: < 0.05 means the real (SCZ - CTL) weighted delta is unusually
    LOW relative to the permuted null ('hypoactive' in SCZ); > 0.95 means
    unusually HIGH ('hyperactive' in SCZ). None (not flagged) otherwise."""
    if quantile_weighted < HYPOACTIVE_QUANTILE_THRESHOLD:
        return 'hypoactive'
    if quantile_weighted > HYPERACTIVE_QUANTILE_THRESHOLD:
        return 'hyperactive'
    return None


def _abundance_direction(mean_diff: float) -> str:
    """group_abundance_stats.csv's mean_diff_schizophrenia_minus_control,
    thresholded at 0. NaN (no abundance data at all -- see
    _abundance_significance) or exactly 0 (no measurable difference) leave
    this undefined (None) rather than forcing a direction."""
    if pd.isna(mean_diff):
        return None
    if mean_diff > 0:
        return 'abundance elevated in SCZ'
    if mean_diff < 0:
        return 'abundance decreased in SCZ'
    return None


def _abundance_significance(p_value: float) -> str:
    """group_abundance_stats.csv's own t-test p_value, thresholded at 0.05.
    NaN happens when a (region, neurotransmitter) has zero abundance in both
    groups (no synthesis genes covered for that region at all) -- no t-test
    is defined there, so this stays undefined (None) rather than 'not
    significant'."""
    if pd.isna(p_value):
        return None
    if p_value < 0.05:
        return 'significantly_different_abundance'
    if p_value > 0.05:
        return 'not_significantly_different'
    return None


def _quantile_to_pvalue(quantile: float) -> float:
    """Two-sided p-value from a rank-based quantile in (0,1): small when the
    real (unshuffled) delta sits near EITHER tail of the shuffled null --
    matching hypoactive (quantile near 0) and hyperactive (near 1) both
    being "significant" in the same sense."""
    return 2.0 * min(quantile, 1.0 - quantile)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Standard Benjamini-Hochberg step-up FDR-adjusted q-values, returned in
    the same order as `p_values`: sort ascending, q_(i) = p_(i) * m / rank_i,
    then take the running minimum from the largest rank down so q stays
    monotonically non-decreasing as p increases."""
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    raw_q = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw_q[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_values = np.empty(m)
    q_values[order] = adjusted
    return q_values


def _add_fdr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds p_value_{weighted,non_weighted} (two-sided, from the rank-based
    quantile) and q_value_{weighted,non_weighted} (Benjamini-Hochberg FDR,
    each metric its own family/denominator). Per metric, a combination is
    excluded from ITS OWN FDR family (q_value left NaN) when both its real
    SCZ and CTL values round to zero within FDR_EPSILON -- a structurally
    missing connection (e.g. zero sender-region abundance for the weighted
    metric) carries no information and would only dilute the correction if
    counted as a test."""
    df = df.copy()
    df['p_value_weighted'] = df['quantile_weighted'].apply(_quantile_to_pvalue)
    df['p_value_non_weighted'] = df['quantile_non_weighted'].apply(_quantile_to_pvalue)

    valid_weighted = ~((df['SCZ_weighted'].abs() < FDR_EPSILON) & (df['CTL_weighted'].abs() < FDR_EPSILON))
    valid_non_weighted = ~((df['SCZ_non_weighted'].abs() < FDR_EPSILON) & (df['CTL_non_weighted'].abs() < FDR_EPSILON))

    df['q_value_weighted'] = np.nan
    df.loc[valid_weighted, 'q_value_weighted'] = _benjamini_hochberg(
        df.loc[valid_weighted, 'p_value_weighted'].to_numpy())
    df['q_value_non_weighted'] = np.nan
    df.loc[valid_non_weighted, 'q_value_non_weighted'] = _benjamini_hochberg(
        df.loc[valid_non_weighted, 'p_value_non_weighted'].to_numpy())
    return df


def _discover_replicate_seeds(job_dir: str) -> list[int]:
    """Every label_shuffle_seed with a run_NNNNNN directory under job_dir,
    sorted -- discovered by glob rather than assumed, so this tracks however
    many replicates actually exist on disk."""
    seeds = []
    for path in glob.glob(os.path.join(job_dir, 'run_*')):
        match = re.fullmatch(r'run_(\d+)', os.path.basename(path))
        if match and os.path.isdir(path):
            seeds.append(int(match.group(1)))
    return sorted(seeds)


def _load_replicate(run_dir: str, seed: int, background_source: str, p_value_method: str) -> dict:
    """{disease: (non_weighted_dict[gene_set_id -> 5x5 region df],
    weighted_sheets[sheet_name -> 5x5 region df])} for one replicate
    (seed=None for the unshuffled baseline)."""
    result = {}
    for disease in DISEASES:
        signaling_path = human_oneshot.output_path(run_dir, background_source, p_value_method, seed, disease)
        signaling_data = pd.read_csv(signaling_path)
        non_weighted = human_weighted_connectivity.build_gene_set_adjacency(
            signaling_data, GENE_SET_TO_USE, HUMAN_REGIONS)

        weighted_path = human_weighted_connectivity.weighted_connectivity_output_paths(
            run_dir, background_source, p_value_method, disease)['weighted_connectivity']
        weighted = pd.read_excel(weighted_path, sheet_name=None, index_col=0)  # all sheets in one read

        result[disease] = (non_weighted, weighted)
    return result


def _join_abundance_stats(flagged: pd.DataFrame, group_abundance_stats_path: str) -> pd.DataFrame:
    """Cross-references each flagged row's own (sender_region,
    neurotransmitter) -- NOT receiver_region, since abundance is a per-region
    property with no directionality -- against group_abundance_stats.csv's
    per-(region, neurotransmitter) SCZ-vs-CTL mean abundance comparison."""
    abundance_stats = pd.read_csv(group_abundance_stats_path)
    abundance_stats = abundance_stats.rename(columns={'region': 'sender_region'})
    merged = flagged.merge(
        abundance_stats[['sender_region', 'neurotransmitter', 'mean_diff_schizophrenia_minus_control', 'p_value', 'hedges_g']],
        on=['sender_region', 'neurotransmitter'], how='left',
    )
    merged['sender_abundance_direction'] = merged['mean_diff_schizophrenia_minus_control'].apply(_abundance_direction)
    merged['sender_abundance_significance'] = merged['p_value'].apply(_abundance_significance)
    return merged


def main(unshuffled_dir: str = DEFAULT_UNSHUFFLED_DIR,
         job_dir: str = DEFAULT_JOB_DIR,
         background_source: str = human_oneshot.DEFAULT_BACKGROUND_SOURCE,
         p_value_method: str = human_oneshot.DEFAULT_P_VALUE_METHOD,
         output_path: str = None,
         flagged_output_path: str = None,
         group_abundance_stats_path: str = DEFAULT_GROUP_ABUNDANCE_STATS_PATH) -> pd.DataFrame:
    this_job_name = Path(job_dir).name
    if output_path is None:
        output_path = os.path.join(DEFAULT_REPORT_SAVE_DIR, 
                                   this_job_name, 
                                   SETTINGS['human_delta_quantile_report']['delta_quantile_report_filename'])
    if flagged_output_path is None:
        flagged_output_path = os.path.join(DEFAULT_REPORT_SAVE_DIR, 
                                this_job_name, 
                                SETTINGS['human_delta_quantile_report']['delta_quantile_flagged_filename'])

    real = _load_replicate(unshuffled_dir, None, background_source, p_value_method)

    seeds = _discover_replicate_seeds(job_dir)
    if not seeds:
        raise ValueError(f'No run_NNNNNN replicate directories found under {job_dir}')
    shuffled = [
        _load_replicate(os.path.join(job_dir, f'run_{seed:06d}'), seed, background_source, p_value_method)
        for seed in seeds
    ]
    n_null = len(shuffled)

    rows = []
    for nt_name, gene_set_id in NT_TO_GENE_SET.items():
        weighted_sheet_name = f'weighted_{nt_name}'
        for region1 in HUMAN_REGIONS:
            for region2 in HUMAN_REGIONS:
                if region1 == region2:
                    continue

                ctl_weighted = real['CTL'][1][weighted_sheet_name].loc[region1, region2]
                scz_weighted = real['SCZ'][1][weighted_sheet_name].loc[region1, region2]
                ctl_non_weighted = real['CTL'][0][gene_set_id].loc[region1, region2]
                scz_non_weighted = real['SCZ'][0][gene_set_id].loc[region1, region2]

                delta_weighted = scz_weighted - ctl_weighted
                delta_non_weighted = scz_non_weighted - ctl_non_weighted

                null_weighted = np.array([
                    rep['SCZ'][1][weighted_sheet_name].loc[region1, region2]
                    - rep['CTL'][1][weighted_sheet_name].loc[region1, region2]
                    for rep in shuffled
                ])
                null_non_weighted = np.array([
                    rep['SCZ'][0][gene_set_id].loc[region1, region2]
                    - rep['CTL'][0][gene_set_id].loc[region1, region2]
                    for rep in shuffled
                ])

                delta_weighted_mean = float(np.mean(null_weighted))
                delta_weighted_std = float(np.std(null_weighted))
                delta_weighted_skewness = float(scipy.stats.skew(null_weighted))
                delta_weighted_kurtosis = float(scipy.stats.kurtosis(null_weighted))

                delta_non_weighted_mean = float(np.mean(null_non_weighted))
                delta_non_weighted_std = float(np.std(null_non_weighted))
                delta_non_weighted_skewness = float(scipy.stats.skew(null_non_weighted))
                delta_non_weighted_kurtosis = float(scipy.stats.kurtosis(null_non_weighted))

                # A structurally zero-abundance sender region (see
                # HUMAN_PIPELINE.md, section 7) makes every replicate's weighted value exactly
                # 0, so both delta_weighted and delta_weighted_std are 0 --
                # 0/0 is genuinely undefined, left as NaN rather than forced
                # to a number.
                with np.errstate(invalid='ignore'):
                    delta_weighted_normalized = (delta_weighted - delta_weighted_mean) / delta_weighted_std

                rank_weighted = qu._tie_adjusted_rank(null_weighted, delta_weighted, 'mid-rank')
                quantile_weighted = (rank_weighted + 0.5) / (n_null + 1)
                rank_non_weighted = qu._tie_adjusted_rank(null_non_weighted, delta_non_weighted, 'mid-rank')
                quantile_non_weighted = (rank_non_weighted + 0.5) / (n_null + 1)

                rows.append({
                    'neurotransmitter': nt_name,
                    'sender_region': region1,
                    'receiver_region': region2,
                    'SCZ_weighted': scz_weighted,
                    'CTL_weighted': ctl_weighted,
                    'SCZ_non_weighted': scz_non_weighted,
                    'CTL_non_weighted': ctl_non_weighted,

                    # statistics of null distribution
                    'delta_weighted': delta_weighted,
                    'delta_weighted_mean': delta_weighted_mean,
                    'delta_weighted_std': delta_weighted_std,
                    'delta_weighted_skewness': delta_weighted_skewness,
                    'delta_weighted_kurtosis': delta_weighted_kurtosis,
                    'delta_weighted_normalized': delta_weighted_normalized,

                    'delta_non_weighted': delta_non_weighted,
                    'delta_non_weighted_mean': delta_non_weighted_mean,
                    'delta_non_weighted_std': delta_non_weighted_std,
                    'delta_non_weighted_skewness': delta_non_weighted_skewness,
                    'delta_non_weighted_kurtosis': delta_non_weighted_kurtosis,
                   
                    'quantile_weighted': quantile_weighted,
                    'quantile_non_weighted': quantile_non_weighted,
                })

    df = pd.DataFrame(rows)
    df = _add_fdr_columns(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'n_null={n_null}: {len(df)} rows written to {output_path}')

    flagged = df.assign(status=df['quantile_weighted'].apply(_activity_status))
    flagged = flagged[flagged['status'].notna()]
    flagged = _join_abundance_stats(flagged, group_abundance_stats_path)
    os.makedirs(os.path.dirname(flagged_output_path), exist_ok=True)
    flagged.to_csv(flagged_output_path, index=False)
    print(f'{len(flagged)} flagged rows (quantile_weighted <{HYPOACTIVE_QUANTILE_THRESHOLD} or '
          f'>{HYPERACTIVE_QUANTILE_THRESHOLD}) written to {flagged_output_path}')

    return df


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_UNSHUFFLED_DIR,
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_JOB_DIR,
        sys.argv[3] if len(sys.argv) > 3 else human_oneshot.DEFAULT_BACKGROUND_SOURCE,
        sys.argv[4] if len(sys.argv) > 4 else human_oneshot.DEFAULT_P_VALUE_METHOD,
        sys.argv[5] if len(sys.argv) > 5 else None,
        sys.argv[6] if len(sys.argv) > 6 else None,
        sys.argv[7] if len(sys.argv) > 7 else DEFAULT_GROUP_ABUNDANCE_STATS_PATH,
    )
