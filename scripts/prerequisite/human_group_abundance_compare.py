"""Human prerequisite P1: per-donor ligand abundance, CTL vs SCZ t-test and Hedges' g per region.

Input:  human expression and sample-matching table.
Output: human_analysis_results/group_abundance_comparison/group_abundance_stats.csv and
        individual_abundance_boxplots.png
Usage:  see HUMAN_PIPELINE.md, section 4.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import yaml

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

import src.calculate_abundance as abund
from src.project_variables import HUMAN_REGIONS, NAMED_NEUROTRANSMITTERS, DISEASES
from src.human_expression_load import load_human_expression_by_region_and_disease
from src.visualization.custom_colors import DISEASE_COLORS


DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_group_abundance_compare']['report_save_dir'])
DEFAULT_PLOT_FILENAME = SETTINGS['human_group_abundance_compare']['plot_filename']
DEFAULT_STATS_FILENAME = SETTINGS['human_group_abundance_compare']['stats_filename']

DISEASE_DISPLAY_NAMES = {'CTL': 'Control', 'SCZ': 'Schizophrenia'}
expression_log_base = SETTINGS['analysis_global_settings']['log_expression_base']


def calculate_individual_abundance(region_disease_exprs: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Per-donor neurotransmitter abundance, long form.

    Args:
        region_disease_exprs: from load_human_expression_by_region_and_disease(),
            region -> {'CTL': df, 'SCZ': df}, each df genes x donor_id.

    Returns:
        pd.DataFrame: columns region, neurotransmitter, disease, donor_id,
            abundance -- one row per (region, neurotransmitter, donor).
    """
    long_frames = []
    for region in HUMAN_REGIONS:
        for disease in DISEASES:
            exprs = region_disease_exprs[region][disease]
            genes_to_extract = np.intersect1d(abund.essential_synthesis_genes, exprs.index)
            # same +2.0 positivity shift calculate_synthesis_gene_expressions applies
            # before calculate_ligand_abundance's multiplicative, fractional-power
            # class combination -- without it, per-donor (higher-variance than a
            # cross-donor mean) negative residuals risk NaNs from fractional powers
            # of negative bases.
            essential_exprs = exprs.loc[genes_to_extract] + expression_log_base
            abundance = abund.calculate_ligand_abundance(essential_exprs)
            abundance = abundance.loc[NAMED_NEUROTRANSMITTERS]
            abundance.index.name = 'neurotransmitter'
            long_ = abundance.reset_index().melt(
                id_vars='neurotransmitter', var_name='donor_id', value_name='abundance')
            long_['region'] = region
            long_['disease'] = disease
            long_frames.append(long_)
    long_abundance = pd.concat(long_frames, ignore_index=True)
    return long_abundance[['region', 'neurotransmitter', 'disease', 'donor_id', 'abundance']]


def _hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Hedges' g (bias-corrected Cohen's d) for group_a relative to group_b."""
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_sd
    small_sample_correction = 1 - (3 / (4 * (n_a + n_b) - 9))
    return float(cohens_d * small_sample_correction)


def compute_group_statistics(long_abundance: pd.DataFrame) -> pd.DataFrame:
    """Control-vs-Schizophrenia comparison for every (region, neurotransmitter).

    Args:
        long_abundance: from calculate_individual_abundance().

    Returns:
        pd.DataFrame: one row per (region, neurotransmitter) with group means,
            mean difference (Schizophrenia - Control), Student's t-test
            statistic/p-value, and Hedges' g (same direction as the mean
            difference).
    """
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for region in HUMAN_REGIONS:
            for neurotransmitter in NAMED_NEUROTRANSMITTERS:
                subset = long_abundance[
                    (long_abundance['region'] == region) & (long_abundance['neurotransmitter'] == neurotransmitter)]
                control_values = subset.loc[subset['disease'] == 'CTL', 'abundance'].dropna().to_numpy()
                schizo_values = subset.loc[subset['disease'] == 'SCZ', 'abundance'].dropna().to_numpy()
                n_control, n_schizo = len(control_values), len(schizo_values)

                mean_control = float(np.mean(control_values)) if n_control else np.nan
                mean_schizo = float(np.mean(schizo_values)) if n_schizo else np.nan

                if n_control > 1 and n_schizo > 1:
                    t_stat, p_value = scipy.stats.ttest_ind(schizo_values, control_values, equal_var=True)
                    hedges_g = _hedges_g(schizo_values, control_values)
                else:
                    t_stat, p_value, hedges_g = np.nan, np.nan, np.nan

                rows.append({
                    'region': region,
                    'neurotransmitter': neurotransmitter,
                    'n_control': n_control,
                    'n_schizophrenia': n_schizo,
                    'mean_control': mean_control,
                    'mean_schizophrenia': mean_schizo,
                    'mean_diff_schizophrenia_minus_control': mean_schizo - mean_control,
                    't_stat': float(t_stat) if pd.notna(t_stat) else np.nan,
                    'p_value': float(p_value) if pd.notna(p_value) else np.nan,
                    'hedges_g': hedges_g,
                })
    return pd.DataFrame(rows)


def _annotate_significance(ax: plt.Axes, subset: pd.DataFrame, nt_stats: pd.DataFrame) -> None:
    """Draws a Control-vs-Schizophrenia significance bracket per region on
    `ax`, reusing the exact p-values already computed by
    compute_group_statistics (rather than letting statannotations recompute
    its own test) so the plot never disagrees with the saved stats CSV.
    Regions with no valid test (e.g. a neurotransmitter whose synthesis genes
    are entirely unmeasured there -> both groups constant at 0.0, see
    HUMAN_PIPELINE.md, section 6) are skipped rather than annotated."""
    valid_regions = [region for region in HUMAN_REGIONS if pd.notna(nt_stats.loc[region, 'p_value'])]
    if not valid_regions:
        return
    pairs = [((region, 'CTL'), (region, 'SCZ')) for region in valid_regions]
    pvalues = [nt_stats.loc[region, 'p_value'] for region in valid_regions]

    annotator = Annotator(
        ax, pairs, data=subset, x='region', y='abundance', hue='disease',
        order=HUMAN_REGIONS, hue_order=DISEASES, verbose=False,
    )
    annotator.configure(test=None, text_format='star', loc='inside', verbose=False,
                        line_height=0.02, line_width=1.0, fontsize=8)
    annotator.set_pvalues_and_annotate(pvalues)


def plot_individual_abundance_distributions(long_abundance: pd.DataFrame, stats_df: pd.DataFrame,
                                             output_path: str, n_cols: int = 4) -> None:
    """One boxplot panel per neurotransmitter (region on x-axis, Control/
    Schizophrenia as hue), arranged in a single grid figure, with a
    significance bracket (statannotations, star notation) per region built
    from stats_df's own t-test p-values (see compute_group_statistics)."""
    n_neurotransmitters = len(NAMED_NEUROTRANSMITTERS)
    n_rows = int(np.ceil(n_neurotransmitters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

    legend_handles, legend_labels = None, None
    for idx, neurotransmitter in enumerate(NAMED_NEUROTRANSMITTERS):
        ax = axes[idx // n_cols][idx % n_cols]
        subset = long_abundance[long_abundance['neurotransmitter'] == neurotransmitter]
        sns.boxplot(
            data=subset, x='region', y='abundance', hue='disease',
            order=HUMAN_REGIONS, hue_order=DISEASES, palette=DISEASE_COLORS,
            width=0.6, fliersize=2, linewidth=1, ax=ax,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        nt_stats = stats_df.loc[stats_df['neurotransmitter'] == neurotransmitter].set_index('region')
        _annotate_significance(ax, subset, nt_stats)

        ax.set_title(neurotransmitter)
        ax.set_xlabel('')
        ax.set_ylabel('Abundance' if idx % n_cols == 0 else '')
        ax.tick_params(axis='x', rotation=30)

    for idx in range(n_neurotransmitters, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    fig.legend(legend_handles, [DISEASE_DISPLAY_NAMES[label] for label in legend_labels],
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle('Individual-level neurotransmitter ligand abundance by region and disease group', y=1.07)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main(output_dir: str = DEFAULT_OUTPUT_DIR):
    plot_path = os.path.join(output_dir, DEFAULT_PLOT_FILENAME)
    stats_path = os.path.join(output_dir, DEFAULT_STATS_FILENAME)

    region_disease_exprs = load_human_expression_by_region_and_disease()
    long_abundance = calculate_individual_abundance(region_disease_exprs)
    stats_df = compute_group_statistics(long_abundance)

    plot_individual_abundance_distributions(long_abundance, stats_df, plot_path)

    os.makedirs(output_dir, exist_ok=True)
    stats_df.to_csv(stats_path, index=False)

    print(f'wrote {plot_path} and {stats_path}')
    return long_abundance, stats_df


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR)
