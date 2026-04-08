import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Any


sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency"))
import plot_config
import plot_utils


RESULTS_SAVED_DIR = os.path.join(PROJECT_ROOT, "test", "figure_inputs")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "test", "plots")

def draw_connectome_heatmap_with_cbar(adjacency_matrix: pd.DataFrame,
                                      figure_args_dict: dict[str, Any]) -> None:
    this_large_fig = plt.figure(figsize=figure_args_dict['figure_size'], 
                                layout='constrained')
    subfigures_array_for_cbar_and_heatmap = \
        this_large_fig.subfigures(1, 2, wspace=0.05, hspace=0.05, width_ratios=[11, 1],)
    heatmap_figure_obj: plt.Figure = subfigures_array_for_cbar_and_heatmap[0]
    plot_utils.draw_heatmap(adjacency_matrix, heatmap_figure_obj, figure_args_dict)
    cbar_figure_obj: plt.Figure = subfigures_array_for_cbar_and_heatmap[1]
    adj_values = adjacency_matrix.to_numpy().flatten()
    adj_values_notnan = adj_values[~np.isnan(adj_values)]
    max_adj_value = adj_values_notnan.max()
    min_adj_value = adj_values_notnan.min()
    plot_utils.draw_vertical_cbars(cbar_figure_obj, 
                        min_adj_value, max_adj_value, tick_numbers= 5)
    this_large_fig.suptitle(figure_args_dict['figure_title'] , fontsize=plot_config.SUBTITLE_SIZE)
    plt.plot()
    this_large_fig.savefig(os.path.join(PLOT_SAVE_DIR, figure_args_dict['figure_filename']),
                           format='svg', transparent=True, dpi=400)
    plt.show()
    return None


def main():
    CM_all = pd.read_csv(os.path.join(RESULTS_SAVED_DIR, 'cocomac_GS.csv'), index_col=0)
    CM_all.astype(np.float64)
    figure_args_dict = {
        'figure_size': (19, 16), # total figure size
        'label_font_size': 17, # label size
        'draw_hemisphere_divisor': False,
        'divisor_color': 'lightgrey',
        'annotate': True,
        'figure_title': 'Matched golden standard CoCoMac connectome \n No hemisphere division',
        'figure_filename': 'CoCoMac_GS_heatmap.svg'
    }
    draw_connectome_heatmap_with_cbar(CM_all, figure_args_dict)

    VB_all = pd.read_csv(os.path.join(
        RESULTS_SAVED_DIR, 'virtualbrain_GS.csv'), index_col=0)
    VB_all.astype(np.float64)
    figure_args_dict['figure_title'] = 'Matched golden standard TheVirtualBrain connectome \n Right hemisphere only'
    figure_args_dict['figure_filename'] = 'TVB_GS_heatmap.svg'
    draw_connectome_heatmap_with_cbar(VB_all, figure_args_dict)



if __name__ == '__main__':
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Initializing...")
    main()
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Finished!")

