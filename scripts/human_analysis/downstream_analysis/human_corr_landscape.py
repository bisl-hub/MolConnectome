"""Human stage 2: gene-level contributions and SCZ - CTL changes for every connection (real labels).

Input:  human expression and sample-matching table, the curated gene list and gene sets.
Output: human_analysis_results/corr_landscape/ (gene-pair and per-gene contribution tables,
        concentration curves and a plot)
Usage:  see HUMAN_PIPELINE.md, section 4.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

with open(Path(__file__).resolve().parent.parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'human_analysis'))

import human_corr_contrib as cc
from src.project_variables import HUMAN_REGIONS, DISEASES
from src.visualization.custom_colors import (
    CONTRIBUTION_COLOR, DIFF_COLOR, BASELINE_COLOR, GRID_COLOR,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, SURFACE_COLOR,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_corr_landscape']['report_save_dir'],
)
GENE_PAIR_FILENAME = SETTINGS['human_corr_landscape']['gene_pair_filename']
GENE_ROLLUP_FILENAME = SETTINGS['human_corr_landscape']['gene_rollup_filename']
GENE_PAIR_DIFF_FILENAME = SETTINGS['human_corr_landscape']['gene_pair_diff_filename']
GENE_ROLLUP_DIFF_FILENAME = SETTINGS['human_corr_landscape']['gene_rollup_diff_filename']
CONTRIBUTION_CURVE_FILENAME = SETTINGS['human_corr_landscape']['contribution_curve_filename']
DIFF_CURVE_FILENAME = SETTINGS['human_corr_landscape']['diff_curve_filename']
PLOT_FILENAME = SETTINGS['human_corr_landscape']['plot_filename']

CHECKPOINTS = (50, 80, 90)


def generate_all_combos() -> list[dict]:
    """Every (gene_set, sender_region, receiver_region) combo -- 9 gene sets
    x 20 directed region pairs = 180 connections.
    itertools.permutations(HUMAN_REGIONS, 2) generates the same 20 directed
    pairs human_delta_quantile_report.py's nested region1/region2 loop
    (skipping region1==region2) does."""
    _, specs_by_key = cc._spec_by_key()
    return [
        {
            'gene_set': gene_set_key,
            'neurotransmitter': spec['neurotransmitter'],
            'sender_region': sender_region,
            'receiver_region': receiver_region,
        }
        for gene_set_key, spec in specs_by_key.items()
        for sender_region, receiver_region in permutations(HUMAN_REGIONS, 2)
    ]


def build_cumulative_curve(values: pd.Series) -> pd.DataFrame:
    """Sort `values` descending, then rank / pct_items / cumulative /
    pct_cumulative -- the standard concentration-curve shape: pct_items =
    100*rank/n on the x axis, pct_cumulative = 100*cumsum/total on the y
    axis. A pathological all-zero `values` (total == 0) is guarded to 0.0
    rather than left as the 0/0 division would otherwise produce."""
    sorted_vals = values.sort_values(ascending=False).reset_index(drop=True)
    cumulative = sorted_vals.cumsum()
    total = sorted_vals.sum()
    n = len(sorted_vals)
    return pd.DataFrame({
        'rank': np.arange(1, n + 1),
        'pct_items': np.arange(1, n + 1) / n * 100,
        'value': sorted_vals.to_numpy(),
        'cumulative': cumulative.to_numpy(),
        'pct_cumulative': (cumulative / total * 100).to_numpy() if total else np.zeros(n),
    })


def checkpoint_table(curve: pd.DataFrame, checkpoints: tuple = CHECKPOINTS) -> pd.DataFrame:
    """For each target cumulative %, the smallest rank/pct_items that reaches
    it -- the first row (in ascending-rank order) where pct_cumulative meets
    the threshold."""
    rows = []
    for threshold in checkpoints:
        hit = curve.loc[curve['pct_cumulative'] >= threshold]
        if hit.empty:
            rows.append({'threshold_pct': threshold, 'rank': np.nan, 'pct_items': np.nan})
        else:
            first = hit.iloc[0]
            rows.append({'threshold_pct': threshold, 'rank': int(first['rank']), 'pct_items': float(first['pct_items'])})
    return pd.DataFrame(rows)


def plot_cumulative_curves(contribution_curve: pd.DataFrame, diff_curve: pd.DataFrame,
                           contribution_checkpoints: pd.DataFrame, diff_checkpoints: pd.DataFrame,
                           output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)

    ax.plot([0, 100], [0, 100], linestyle='--', linewidth=1.5, color=BASELINE_COLOR,
            label='No concentration (uniform)', zorder=1)
    ax.plot(contribution_curve['pct_items'], contribution_curve['pct_cumulative'],
            linewidth=2, color=CONTRIBUTION_COLOR, label='Raw contribution (structure)', zorder=3)
    ax.plot(diff_curve['pct_items'], diff_curve['pct_cumulative'],
            linewidth=2, color=DIFF_COLOR, label='|Δ share of total| (relative change)', zorder=3)

    # Headline 80% Pareto checkpoint: drop line + end marker per curve. A
    # concentration curve (sorted descending) always sits on/above the y=x
    # diagonal, so the region below it (y < x) is guaranteed free of both
    # curves -- that's where the checkpoint text box below is placed.
    ax.axhline(80, linestyle=':', linewidth=1, color=TEXT_MUTED, alpha=0.6, zorder=2)
    for curve, checkpoints, color in (
        (contribution_curve, contribution_checkpoints, CONTRIBUTION_COLOR),
        (diff_curve, diff_checkpoints, DIFF_COLOR),
    ):
        row = checkpoints.loc[checkpoints['threshold_pct'] == 80].iloc[0]
        if pd.notna(row['pct_items']):
            ax.plot([row['pct_items'], row['pct_items']], [0, 80], linestyle=':',
                    linewidth=1, color=color, alpha=0.6, zorder=2)
            ax.plot(row['pct_items'], 80, marker='o', markersize=8, color=color,
                    markeredgecolor=SURFACE_COLOR, markeredgewidth=2, zorder=4)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('% of gene-connection instances (ranked by magnitude)', color=TEXT_SECONDARY)
    ax.set_ylabel('Cumulative % of total', color=TEXT_SECONDARY)
    ax.set_title('How much of the correlation structure does a small share of genes explain?',
                color=TEXT_PRIMARY, fontsize=12)
    ax.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BASELINE_COLOR)
    ax.tick_params(colors=TEXT_MUTED)

    checkpoint_lines = []
    for label, checkpoints in (('Raw contribution', contribution_checkpoints), ('|Δ share|', diff_checkpoints)):
        parts = ', '.join(
            f'{int(row.threshold_pct)}%→{row.pct_items:.1f}%' if pd.notna(row.pct_items) else f'{int(row.threshold_pct)}%→n/a'
            for row in checkpoints.itertuples()
        )
        checkpoint_lines.append(f'{label}: {parts}')
    ax.text(0.97, 0.03, '\n'.join(checkpoint_lines), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color=TEXT_SECONDARY,
            bbox=dict(boxstyle='round', facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02),
              frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main(output_dir: str = DEFAULT_OUTPUT_DIR,
         p_value_method: str = cc.DEFAULT_P_VALUE_METHOD) -> tuple:
    combos = generate_all_combos()
    print(f'{len(combos)} connections generated (gene sets x directed region pairs)')

    pair_all, rollup_all, pair_diff, gene_diff = cc.compute_all(combos, p_value_method)

    pair_path = os.path.join(output_dir, GENE_PAIR_FILENAME)
    rollup_path = os.path.join(output_dir, GENE_ROLLUP_FILENAME)
    pair_diff_path = os.path.join(output_dir, GENE_PAIR_DIFF_FILENAME)
    gene_diff_path = os.path.join(output_dir, GENE_ROLLUP_DIFF_FILENAME)
    os.makedirs(output_dir, exist_ok=True)
    pair_all.to_csv(pair_path, index=False, float_format='%.6g', compression='gzip')
    rollup_all.to_csv(rollup_path, index=False, float_format='%.6g')
    pair_diff.to_csv(pair_diff_path, index=False, float_format='%.6g', compression='gzip')
    gene_diff.to_csv(gene_diff_path, index=False, float_format='%.6g')
    print(f'{len(combos)} connections x {len(DISEASES)} diseases: '
          f'wrote {len(pair_all)} gene-pair rows to {pair_path}, '
          f'{len(rollup_all)} gene-rollup rows to {rollup_path}, '
          f'{len(pair_diff)} gene-pair DIFF rows to {pair_diff_path}, '
          f'{len(gene_diff)} gene DIFF rows to {gene_diff_path}')

    contribution_curve = build_cumulative_curve(rollup_all['total_contribution'])
    # PCT_DIFF (share-of-combo-total change), not raw DIFF -- gene sets vary
    # widely in size (n_cat1/n_cat2 from single digits to the low 20s across
    # GENE_SET_TO_USE), so raw DIFF (a sum over partner genes) systematically
    # favors genes from larger gene sets. PCT_DIFF is bounded in [-1, 1]
    # regardless of gene-set size, so it's the metric comparable across
    # combos -- see human_corr_contrib.compute_gene_diff's docstring.
    diff_curve = build_cumulative_curve(gene_diff['PCT_DIFF'].abs())
    contribution_checkpoints = checkpoint_table(contribution_curve)
    diff_checkpoints = checkpoint_table(diff_curve)

    contribution_curve_path = os.path.join(output_dir, CONTRIBUTION_CURVE_FILENAME)
    diff_curve_path = os.path.join(output_dir, DIFF_CURVE_FILENAME)
    contribution_curve.to_csv(contribution_curve_path, index=False, float_format='%.6g')
    diff_curve.to_csv(diff_curve_path, index=False, float_format='%.6g')

    plot_path = os.path.join(output_dir, PLOT_FILENAME)
    plot_cumulative_curves(contribution_curve, diff_curve, contribution_checkpoints, diff_checkpoints, plot_path)

    print('Raw contribution curve checkpoints (n={}):'.format(len(contribution_curve)))
    print(contribution_checkpoints.to_string(index=False))
    print('|Δ share of total| (PCT_DIFF) curve checkpoints (n={}):'.format(len(diff_curve)))
    print(diff_checkpoints.to_string(index=False))
    print(f'wrote cumulative curves to {contribution_curve_path}, {diff_curve_path}, and plot to {plot_path}')

    return pair_all, rollup_all, pair_diff, gene_diff, contribution_curve, diff_curve


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR,
        sys.argv[2] if len(sys.argv) > 2 else cc.DEFAULT_P_VALUE_METHOD,
    )
