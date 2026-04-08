import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
import datetime
import scipy.stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from typing import Any


sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency"))
import plot_config



RESULTS_SAVED_DIR = os.path.join(PROJECT_ROOT, "test", "figure_inputs")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "test", "plots")


def fit_gamma_and_plot(contribution_data_array: 'np.ndarray[float]', lb=0.01, ub = 0.99) -> 'plt.Figure':
    figure = plt.figure(figsize=(5, 5), layout='constrained')
    ax = figure.subplots(1, 1)
    a, loc, scale = scipy.stats.gamma.fit(contribution_data_array)
    fitted_gamma_x = np.linspace(
        scipy.stats.gamma.ppf(lb, a, loc=loc, scale=scale), 
        scipy.stats.gamma.ppf(ub, a, loc=loc, scale=scale), 
        100
    )
    fitted_gamma_y = scipy.stats.gamma.pdf(fitted_gamma_x, a, loc=loc, scale=scale)
    ax.hist(contribution_data_array, bins=20, color='royalblue', density=True)
    ax.set_xlabel("Correlation contribution")
    ax.set_ylabel("Probablity density")
    ax.set_xlim(0, ax.get_xlim()[1])
    legend_obj = [
        mpatches.Rectangle((0,0), 0.3, 0.3, color='royalblue', ),
    ]
    ax.legend(handles=legend_obj, 
                        labels=['Density',],# 'Estimated gamma distribution with ' + '\u03B1=' + f'{a:1.3f}'],
                        numpoints=1,
                        loc='upper right', 
                        fontsize=plot_config.SUBTITLE_SIZE/2+2,)
    plt.show()
    return figure


def plot_ecdf_and_line(contribution_data_array: 'np.ndarray[float]',) -> 'plt.Figure':
    figure = plt.figure(figsize=(5, 5), layout='constrained')
    ax = figure.subplots(1, 1)
    # ax.ecdf(contribution_data_array, color='royalblue')
    sorted_data = np.sort(contribution_data_array)[::-1]
    top_50_count = int(0.5 * len(sorted_data))
    top_50_value = sorted_data[top_50_count]
    # quantile = np.quantile(contribution_data_array, 0.5)
    ax.axvline(0.5, color='red')
    cumulative_sum = np.cumsum(sorted_data)
    real_y_value = cumulative_sum / np.sum(contribution_data_array)
    real_x_value = np.linspace(0, 1, contribution_data_array.shape[0])
    ax.plot(real_x_value, real_y_value, color='royalblue')
    legend_obj = [
        mlines.Line2D([],[], color='royalblue', marker='None', linestyle='solid',),
        mlines.Line2D([],[], color='red', marker='None', linestyle='solid',),
    ]
    ax.legend(handles=legend_obj, 
                        labels=['Cumulative sum', 'Top 50% cutoff'],
                        numpoints=1,
                        loc='lower right', 
                        fontsize=plot_config.SUBTITLE_SIZE/2+2,)
    ax.set_xlabel("Top observation ratio")
    ax.set_ylabel("Cumulative ratio")
    plt.show()
    return figure


def analyze_corr_contrib(correlation_contribution_data: pd.DataFrame):
    positive_diff_corr_contribs = correlation_contribution_data.groupby('difference_sign').get_group('positive')
    positive_corr_contribs = positive_diff_corr_contribs['correlation_contribution'].to_numpy()
    real_positives = positive_corr_contribs[positive_corr_contribs > 1e-8]

    negative_diff_corr_contribs = correlation_contribution_data.groupby('difference_sign').get_group('negative')
    negative_corr_contribs = negative_diff_corr_contribs['correlation_contribution'].to_numpy()
    negative_corr_contribs = np.negative(negative_corr_contribs)
    real_negatives = negative_corr_contribs[negative_corr_contribs > 1e-8]

    # total
    total_corr_contribs = np.concatenate([real_positives, real_negatives])
    sorted_data = np.sort(total_corr_contribs)[::-1]

    # Calculate what % of the data is in the top 20%
    top_20_count = int(0.5 * len(sorted_data))
    top_20_sum = np.sum(sorted_data[:top_20_count])
    total_sum = np.sum(sorted_data)
    share = top_20_sum / total_sum
    print(f"The top 50% of observations account for {share:.1%} of the total value.")

    '''Suppl figure S1A - distribution and fitted gamma dist for positives'''
    total_dist_fig = fit_gamma_and_plot(total_corr_contribs, lb=0.2)
    # fit_power_and_plot(total_corr_contribs, lb=0.1, ub=0.999)
    '''Suppl figure S2A - cumulative sum'''
    total_cumsum_fig = plot_ecdf_and_line(total_corr_contribs)

    total_dist_fig.savefig(os.path.join(PLOT_SAVE_DIR, 'figure_S1A_total_dist.svg'), format='svg', dpi=400)
    total_cumsum_fig.savefig(os.path.join(PLOT_SAVE_DIR, 'figure_S1B_total_cumsum.svg'), format='svg', dpi=400)

    return None


def main():
    corr_contrib_data = pd.read_csv(os.path.join(RESULTS_SAVED_DIR, 'correlation_contributions.csv'))
    analyze_corr_contrib(corr_contrib_data)


if __name__ == '__main__':
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Initializing...")
    main()
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Finished!")