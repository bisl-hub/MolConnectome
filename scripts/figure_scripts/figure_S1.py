import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
import pickle
import datetime
import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import scipy.stats


sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency"))
import plot_config
import plot_utils


RESULTS_SAVED_DIR = os.path.join(PROJECT_ROOT, "test", "figure_inputs")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "test", "plots")
CMAPS = plot_utils.make_custom_cmap_dict()

VISp_regions = ['L23_IT', 'L4', 'L5_IT', 'L5_PT', 'L6_CT', 'L6_IT', 'L6b',]
ALM_regions = ['L23_IT', 'L5_IT', 'L5_PT', 'L6_CT', 'L6_IT',]
VISp_target_regions = ['VISp', 'RSP', 'ACA',]
ALM_target_regions = ['SSs', 'SSp', 'RSP', 'ORB', 'MOp', 'ALM',]


def make_golden_standard_adjacency(model_region:str, include_controversial: True):
    '''
    VISp_regions = ['L23_IT' , 'L4', 'L5_IT', 'L5_PT', 'L6_CT', 'L6_IT', 'L6b',] \n
    ALM_regions  = ['L23_IT' , 'L5_IT', 'L5_PT', 'L6_CT', 'L6_IT',] \n
    VISp_target_regions = ['VISp', 'RSP', 'ACA',] \n
    ALM_target_regions = ['SSs', 'SSp', 'RSP', 'ORB', 'MOp', 'ALM',] \n
    '''
    if model_region == 'VISp':
        source_regions = VISp_regions
        target_regions = VISp_target_regions
        base_dataframe = pd.DataFrame(0.0, index=source_regions, columns=target_regions)
        base_dataframe.loc['L23_IT', ['VISp', 'RSP', 'ACA']] = 1.0
        if include_controversial:
            base_dataframe.loc['L4', ['VISp', 'RSP',]] = 1.0
        else:
            base_dataframe.loc['L4', ['RSP',]] = 1.0
        base_dataframe.loc['L5_IT', ['VISp', 'RSP', 'ACA']] = 1.0
        base_dataframe.loc['L5_PT', ['RSP',]] = 1.0
        base_dataframe.loc['L6_CT', ['VISp',]] = 1.0
        base_dataframe.loc['L6_IT', ['VISp', 'RSP', 'ACA']] = 1.0
        base_dataframe.loc['L6b', ['ACA']] = 1.0
    elif model_region == 'ALM':
        source_regions = ALM_regions
        target_regions = ALM_target_regions
        base_dataframe = pd.DataFrame(0.0, index=source_regions, columns=target_regions)
        base_dataframe.loc['L23_IT', ['SSs', 'MOp', 'ALM',]] = 1.0
        base_dataframe.loc['L5_IT', ['SSs', 'SSp', 'ORB', 'MOp', 'ALM',]] = 1.0
        base_dataframe.loc['L5_PT', ['RSP',]] = 1.0
        base_dataframe.loc['L6_IT',['SSs', 'SSp', 'ALM',]] = 1.0
    return base_dataframe


def calculate_classifier_metrics(predicted_:pd.DataFrame, 
                                 true_value_:pd.DataFrame) -> dict[str, Any]:
    predicted_flattened = predicted_.stack(future_stack=True)
    true_flattened = true_value_.stack(future_stack=True)
    #! 이제 해야할 일: 정답을 binarization 하고 "없는거" 빼기.
    #! 우선 정답지에 없는 것부터 빼기
    new_indices = []
    for index in true_flattened.index:
        if pd.isna(true_flattened.loc[index]):
            pass
        else:
            if index[0] == index[1]:
                pass
            else:
                new_indices.append(index)
    predicted_filtered = predicted_flattened.loc[new_indices]
    true_filtered = true_flattened.loc[new_indices]
    true_binarized = [1 if x > 1e-8 else 0 
                      for x in true_filtered.to_numpy() ]
    predicted_array = predicted_filtered.to_numpy()
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_binarized, predicted_array)
    prec, recall, thresholds_ = sklearn.metrics.precision_recall_curve(true_binarized, predicted_array)
    auroc_ = sklearn.metrics.roc_auc_score(true_binarized, predicted_array)
    auprc_ = sklearn.metrics.average_precision_score(true_binarized, predicted_array)
    # accuracy_ = sklearn.metrics.accuracy_score(true_binarized, predicted_array)
    # precision_ = sklearn.metrics.precision_score(true_binarized, predicted_array)
    # recall_ = sklearn.metrics.recall_score(true_binarized, predicted_array)
    prevalence_positives = np.sum(true_binarized) / len(true_binarized)
    metrics_ = {
        'roc_fpr' : fpr,
        'roc_tpr' : tpr,
        'roc_auroc' : auroc_,
        'prc_precision' : prec,
        'prc_recall' : recall,
        'prc_auprc': auprc_,
        'positive_prevalence': prevalence_positives,
        # 'accuracy': accuracy_, 
        # 'precision': precision_, 
        # 'recall': recall_, 
        }
    return metrics_
    

def ROC_curve_plotter(predicted_value: pd.DataFrame, 
                      true_value: pd.DataFrame,
                      figure_args_dict: dict[str, Any]):
    large_figure_object = plt.figure(figsize=figure_args_dict['figure_size'], 
                                     layout='constrained')
    this_curve_figure: plt.Figure = large_figure_object
    each_subplot_array = this_curve_figure.subplots(2, 1)# ,width_ratios=[3, 3, 2])
    auroc_curve_ax: plt.Axes = each_subplot_array[0]
    prc_curve_ax: plt.Axes = each_subplot_array[1]
    # metrics_container_ax: plt.Axes = each_subplot_array[2]
    this_true_labels = true_value
    metrics_dict_ = calculate_classifier_metrics(predicted_value, this_true_labels)
    auroc_curve_ax.plot(metrics_dict_['roc_fpr'], metrics_dict_['roc_tpr'], 
                        color=CMAPS['teals'](0.99))
    prc_curve_ax.plot(metrics_dict_['prc_recall'], metrics_dict_['prc_precision'], 
                        color=CMAPS['teals'](0.99))
    roc_auroc = metrics_dict_['roc_auroc']; prc_auprc = metrics_dict_['prc_auprc']
    auroc_curve_ax.plot((0, 1), (0, 1), color='dimgrey', linestyle='dashed')
    prc_curve_ax.plot((0, 1), (metrics_dict_['positive_prevalence'], metrics_dict_['positive_prevalence']), 
                        color='dimgrey', linestyle='dashed')
    auroc_curve_ax.set_xlim(-0.05, 1.05)
    auroc_curve_ax.set_ylim(-0.05, 1.05)
    prc_curve_ax.set_xlim(-0.05, 1.05)
    prc_curve_ax.set_ylim(-0.05, 1.05)
    auroc_curve_ax.text(
            0.5, 1.05, 'ROC curve\nauROC: ' + f'{roc_auroc:1.4f}',
            horizontalalignment='center',
            verticalalignment='bottom', 
            multialignment='center',
            fontsize=plot_config.SUBTITLE_SIZE - 3 ,
            transform=auroc_curve_ax.transAxes, 
            fontweight='bold',
            # bbox=dict(edgecolor='black', boxstyle='round', facecolor='white', ),
            )
    prc_curve_ax.text(
            0.5, 1.05, 'PR curve\nauPRC: ' + f'{prc_auprc:1.4f}',
            horizontalalignment='center',
            verticalalignment='bottom', 
            multialignment='center',
            fontsize=plot_config.SUBTITLE_SIZE - 3 ,
            transform=prc_curve_ax.transAxes, 
            fontweight='bold',
            # bbox=dict(edgecolor='black', boxstyle='round', facecolor='white', ),
            )
    # this_curve_figure.text(
    #         0.5, 1.08, construct_figure_title_string(figure_args_dict),
    #         horizontalalignment='center',
    #         verticalalignment='top', 
    #         multialignment='center',
    #         fontsize=vis.SUBTITLE_SIZE -5,
    #         fontweight='bold',
    #         # bbox=dict(edgecolor='black', boxstyle='round', facecolor='white', ),
    #         )
    this_curve_figure.savefig(os.path.join(PLOT_SAVE_DIR, figure_args_dict['figure_save_title']), 
                              format='svg', transparent=True, dpi=400)
    plt.show()
    return None


def main():
    # ! VISp 
    FIGURE_SIZE = (4, 8)
    VISp_neuronchat_result = pd.read_csv(os.path.join(RESULTS_SAVED_DIR, 'VISp_NC_pseudobulk.csv'), index_col=0)
    VISp_neuronchat_result.astype(np.float64)
    VISp_neuronchat_GS = make_golden_standard_adjacency('VISp', include_controversial=True)
    figure_plotting_args_dict = {
        'figure_size': FIGURE_SIZE,
        'model': 'VISp',
        'figure_save_title': 'Neuronchat_pseudobulk_VISp' + '.svg'
    }
    ROC_curve_plotter(VISp_neuronchat_result, VISp_neuronchat_GS, figure_plotting_args_dict)
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Finished!")
        
    # ! ALM
    ALM_neuronchat_result = pd.read_csv(os.path.join(RESULTS_SAVED_DIR, 'ALM_NC_pseudobulk.csv'), index_col=0)
    ALM_neuronchat_result.astype(np.float64)
    ALM_neuronchat_GS = make_golden_standard_adjacency('ALM', include_controversial=False)
    figure_plotting_args_dict = {
        'figure_size': FIGURE_SIZE,
        'model': 'ALM',
        'figure_save_title': 'Neuronchat_pseudobulk_ALM' + '.svg'
    }
    ROC_curve_plotter(ALM_neuronchat_result, ALM_neuronchat_GS, figure_plotting_args_dict)
    return None


if __name__ == '__main__':
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Initializing...")
    main()
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Finished!")
    