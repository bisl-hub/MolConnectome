import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
import itertools
import datetime

import matplotlib.colors
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from statannotations.Annotator import Annotator
from typing import Any

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency"))
import plot_config
import plot_utils
import core_config


RESULTS_SAVED_DIR = os.path.join(PROJECT_ROOT, "test", "figure_inputs")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "test", "plots")
NT_TO_DRAW = ['DA', '5-HT', 'Glu', 'GABA', 'NE', 'ACh', 'Gly', 'epi', 'Gas_CO', 'Gas_NO', 'NP', 'Syn']


def draw_abundance_boxplot_wrapper(abundance_dataframe: pd.DataFrame, 
                                   boxplot_drawing_args_dict: dict[str, Any]):
    figure_object = plt.figure(figsize=boxplot_drawing_args_dict['figure_size'], layout='constrained')
    ax_obj = figure_object.subplots(1, 1)
    sns.boxplot(abundance_dataframe, x='Dataset', y='Abundance', hue='Disease', 
                order=core_config.HUMAN_REGIONS_TO_DRAW, legend=False, 
                palette={'control': matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['salmon']),
                         'schizo': matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['deepskyblue']),},
                width=0.8, gap=0.1, 
                ax=ax_obj)
    iter_ = list(itertools.product(core_config.HUMAN_REGIONS_TO_DRAW, ['control', 'schizo']))
    pairs_to_compare = [(iter_[0], iter_[1]), (iter_[2], iter_[3]),
                        (iter_[4], iter_[5]), (iter_[6], iter_[7]), 
                        (iter_[8], iter_[9]),]
    annotator = Annotator(ax_obj, pairs_to_compare, data=abundance_dataframe,
                           x='Dataset', y='Abundance', hue='Disease', order=core_config.HUMAN_REGIONS_TO_DRAW, )
    annotator.configure(test='t-test_ind', text_format='star', loc='inside',)# hide_non_significant=True, )
    annotator.apply_and_annotate()
    ax_obj.set_xlabel('Region'); ax_obj.set_ylabel('Estimated abundance');
    # 레전드 만들기.
    legend_obj = [ 
        mpatches.Patch(edgecolor='black', facecolor='salmon', label='Control'), 
        mpatches.Patch(edgecolor='black', facecolor='deepskyblue', label='Schizophrenia'), ]
    ax_obj.legend(handles=legend_obj, 
               labels=['Control', 'Schizophrenia'],
               loc='best', 
               title='Condition',
               fontsize=plot_config.CAPTION_SIZE,)
    figure_object.suptitle(core_config.NEUROTRANSMITTER_NAME_DICT[boxplot_drawing_args_dict['figure_title']])
    figure_file_name = 'figure_S3_abundance_' + boxplot_drawing_args_dict['figure_title'] + '.svg'
    figure_object.savefig(os.path.join(PLOT_SAVE_DIR, figure_file_name),
                transparent=True, dpi=400, format='svg', bbox_inches='tight')
    plt.show()
    return None


def main():
    boxplot_data = pd.read_csv(os.path.join(RESULTS_SAVED_DIR, "neurotransmitter_abundance_longform.csv"), index_col=0)
    boxplot_data['Abundance'] = boxplot_data['Abundance'].astype(np.float64)
    drawing_args_dict = {
        'figure_size' : (8, 5)
    }
    for neurotransmitter_ in NT_TO_DRAW:
        abundance_data = boxplot_data[boxplot_data['Neurotransmitter'] == neurotransmitter_]
        drawing_args_dict['figure_title'] = neurotransmitter_
        draw_abundance_boxplot_wrapper(abundance_data, drawing_args_dict) # #FF0000


if __name__ == '__main__':
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Initializing...")
    main()
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Completed!")