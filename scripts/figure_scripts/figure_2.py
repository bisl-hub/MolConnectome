import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
import datetime

from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.colors

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency"))
import plot_config
import plot_utils


RESULTS_SAVED_DIR = os.path.join(PROJECT_ROOT, "test", "figure_inputs")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "test", "plots")
CMAPS = plot_utils.make_custom_cmap_dict()
CMAP_compare = {
    'NeuronChat':  matplotlib.colors.to_rgb(matplotlib.colors.CSS4_COLORS['royalblue']),
    'Ours':  matplotlib.colors.to_rgb(matplotlib.colors.CSS4_COLORS['crimson']),
}
COLOR_LIST = [
    matplotlib.colors.to_rgb(matplotlib.colors.CSS4_COLORS['royalblue']),
    matplotlib.colors.to_rgb('#f9d9a9'),
    matplotlib.colors.to_rgb('#ef905d'),
    matplotlib.colors.to_rgb('#eb512e'),
    matplotlib.colors.to_rgb(matplotlib.colors.CSS4_COLORS['crimson']),
]
TICK_LABEL_SIZE = plot_config.CAPTION_SIZE + 3
LABEL_SIZE = plot_config.CAPTION_SIZE + 5
FIGURE_TITLE_SIZE = plot_config.CAPTION_SIZE + 7


def calculate_classifier_metrics(predicted_: pd.DataFrame,
                                 true_value_: pd.DataFrame,
                                 permutation_seed=None) -> dict[str, Any]:
    predicted_flattened = predicted_.stack(future_stack=True)
    true_flattened = true_value_.stack(future_stack=True)
    #! 이제 해야할 일: 정답을 binarization 하고 "없는거" 빼기.
    #! 우선 정답지에 없는 것부터 빼기
    new_indices_true = []
    for index in true_flattened.index:
        if pd.isna(true_flattened.loc[index]):
            pass
        else:
            if index[0] == index[1]:
                pass
            else:
                new_indices_true.append(index)
    # 추가적으로 예측값이 NaN 인 경우 - 즉 커넥션이 고려되지 않은 경우 도 빼기.
    new_indices_predicted = []
    for index in predicted_flattened.index:
        if pd.isna(predicted_flattened.loc[index]):
            pass
        else:
            if index[0] == index[1]:
                pass
            else:
                new_indices_predicted.append(index)
    new_indices_true = pd.MultiIndex.from_tuples(new_indices_true)
    new_indices_predicted = pd.MultiIndex.from_tuples(new_indices_predicted)
    new_indices = np.intersect1d(new_indices_true, new_indices_predicted)
    try:
        predicted_filtered = predicted_flattened.loc[new_indices]
        true_filtered = true_flattened.loc[new_indices]
    except KeyError or ValueError:
        raise ValueError('Wrong index', new_indices_true,
                         new_indices_predicted)
    # print(predicted_filtered, predicted_filtered.shape)
    # print(true_filtered, true_filtered.shape)
    true_binarized = [1 if x > 1e-8 else 0
                      for x in true_filtered.to_numpy()]
    predicted_array = [x if x > 1e-8 else 0
                       for x in predicted_filtered.to_numpy()]
    if permutation_seed:
        rng = np.random.default_rng(seed=permutation_seed)
        shuffled_true_labels = rng.permutation(true_binarized)
        true_binarized = shuffled_true_labels
    try:
        fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(
            true_binarized, predicted_array)
        prec, recall, prc_thresholds_ = sklearn.metrics.precision_recall_curve(
            true_binarized, predicted_array)
        auroc_ = sklearn.metrics.roc_auc_score(true_binarized, predicted_array)
        auprc_ = sklearn.metrics.average_precision_score(
            true_binarized, predicted_array)
        prevalence_positives = np.sum(true_binarized) / len(true_binarized)
        metrics_ = {
            'roc_fpr': fpr,
            'roc_tpr': tpr,
            'roc_auroc': auroc_,
            'prc_precision': prec,
            'prc_recall': recall,
            'prc_auprc': auprc_,
            'positive_prevalence': prevalence_positives,
            'roc_thresholds': roc_thresholds,
            'prc_thresholds': prc_thresholds_,
            # 'accuracy': accuracy_,
            # 'precision': precision_,
            # 'recall': recall_,
        }
    except:
        # ! 비어있으면 오류가 남
        # ! 패스해야 한다고 리턴해 주어야 함.
        metrics_ = {
            'roc_fpr': np.nan,
            'roc_tpr': np.nan,
            'roc_auroc': np.nan,
            'prc_precision': np.nan,
            'prc_recall': np.nan,
            'prc_auprc': np.nan,
            'positive_prevalence': np.nan,
        }
        # return False
    # accuracy_ = sklearn.metrics.accuracy_score(true_binarized, predicted_array)
    # precision_ = sklearn.metrics.precision_score(true_binarized, predicted_array)
    # recall_ = sklearn.metrics.recall_score(true_binarized, predicted_array)
    return metrics_


def overlapped_ROC_curve_plotter(predicted_values_collection: 'dict[str, pd.DataFrame]',
                                 true_values_collection: 'dict[str, pd.DataFrame]',
                                 figure_args_dict: 'dict[str, Any]'):
    '''해야 할 것
    1. 각각의 피규어를 따로 따로 그려서 저장해야 함.'''
    large_figure_container: 'list[plt.Figure]' = [
        [plt.figure(figsize=figure_args_dict['figure_size'], layout='constrained')
         for _ in range(2)]
        for _ in true_values_collection.keys()]
    # (지금 상황에서는 (2, 2) 사이즈의 피규어로 구성된 리스트임.)
    for e, tlabel_ in enumerate(true_values_collection):
        this_true_labels = true_values_collection[tlabel_]
        auroc_figure: 'plt.Figure' = large_figure_container[e][0]
        auprc_figure: 'plt.Figure' = large_figure_container[e][1]
        auroc_curve_ax: plt.Axes = auroc_figure.subplots(1, 1)
        prc_curve_ax: plt.Axes = auprc_figure.subplots(1, 1)
        legend_obj = [mlines.Line2D([], [], color=CMAP_compare[pred_value_],
                      marker='None', linestyle='solid', linewidth=2.5)
                      for k, pred_value_ in enumerate(predicted_values_collection)]
        auroc_collection_ = []
        auprc_collection = []
        for j, pred_value_ in enumerate(predicted_values_collection):
            metrics_dict_ = calculate_classifier_metrics(predicted_values_collection[pred_value_],
                                                                       this_true_labels)
            auroc_curve_ax.plot(metrics_dict_['roc_fpr'], metrics_dict_['roc_tpr'],
                                # color=CMAPS['teal_crimson']( j / len(list(predicted_values_collection.keys()))),
                                color=CMAP_compare[pred_value_], alpha=1.0, linewidth=2.5,
                                )
            prc_curve_ax.plot(metrics_dict_['prc_recall'], metrics_dict_['prc_precision'],
                              #   color=CMAPS['teal_crimson']( j / len(list(predicted_values_collection.keys()))),
                              color=CMAP_compare[pred_value_], alpha=1.0, linewidth=2.5,
                              )
            auroc_collection_.append(metrics_dict_['roc_auroc'])
            auprc_collection.append(metrics_dict_['prc_auprc'])
        auroc_labels = [pred_value_ + ', auROC=' + f'{auroc_collection_[e]:1.4f}'
                        for e, pred_value_ in enumerate(predicted_values_collection)]
        auprc_labels = [pred_value_ + ', auPRC=' + f'{auprc_collection[e]:1.4f}'
                        for e, pred_value_ in enumerate(predicted_values_collection)]
        auroc_curve_ax.legend(handles=legend_obj,
                              labels=auroc_labels,
                              numpoints=1,
                              loc='lower right',
                              fontsize=TICK_LABEL_SIZE - 1,)
        prc_curve_ax.legend(handles=legend_obj,
                            labels=auprc_labels,
                            numpoints=1,
                            loc='best',
                            fontsize=TICK_LABEL_SIZE - 1,)
        auroc_curve_ax.plot(
            (0, 1), (0, 1), color='dimgrey', linestyle='dashed')
        prc_curve_ax.plot((0, 1), (metrics_dict_['positive_prevalence'], metrics_dict_['positive_prevalence']),
                          color='dimgrey', linestyle='dashed')
        auroc_curve_ax.set_xlim(-0.05, 1.05)
        auroc_curve_ax.set_ylim(-0.05, 1.05)
        prc_curve_ax.set_xlim(-0.05, 1.05)
        prc_curve_ax.set_ylim(-0.05, 1.05)
        auroc_curve_ax.set_xlabel('FPR', fontdict=dict(fontsize=LABEL_SIZE,))
        auroc_curve_ax.set_ylabel('TPR', fontdict=dict(fontsize=LABEL_SIZE,))
        prc_curve_ax.set_xlabel('Recall', fontdict=dict(fontsize=LABEL_SIZE,))
        prc_curve_ax.set_ylabel(
            'Precision', fontdict=dict(fontsize=LABEL_SIZE,))
        auroc_curve_ax.set_xticks(np.linspace(0, 1, 6))
        auroc_curve_ax.set_yticks(np.linspace(0, 1, 6))
        prc_curve_ax.set_xticks(np.linspace(0, 1, 6))
        prc_curve_ax.set_yticks(np.linspace(0, 1, 6))
        auroc_curve_ax.set_xticklabels(
            auroc_curve_ax.get_xticklabels(), fontdict=dict(fontsize=TICK_LABEL_SIZE,))
        auroc_curve_ax.set_yticklabels(
            auroc_curve_ax.get_yticklabels(), fontdict=dict(fontsize=TICK_LABEL_SIZE,))
        prc_curve_ax.set_xticklabels(
            prc_curve_ax.get_xticklabels(), fontdict=dict(fontsize=TICK_LABEL_SIZE,))
        prc_curve_ax.set_yticklabels(
            prc_curve_ax.get_yticklabels(), fontdict=dict(fontsize=TICK_LABEL_SIZE,))
        auroc_curve_ax.text(
            0.5, 1.1, 'ROC curve',
            horizontalalignment='center', verticalalignment='top', multialignment='center',
            fontsize=FIGURE_TITLE_SIZE, transform=auroc_curve_ax.transAxes,
            # fontweight='bold',
            # bbox=dict(edgecolor='black', boxstyle='round', facecolor='white', ),
        )
        prc_curve_ax.text(
            0.5, 1.1, 'PR curve',
            horizontalalignment='center', verticalalignment='top', multialignment='center',
            fontsize=FIGURE_TITLE_SIZE, transform=prc_curve_ax.transAxes,
            # fontweight='bold',
            # bbox=dict(edgecolor='black', boxstyle='round', facecolor='white', ),
        )
    plt.show()
    for e, tlabel_ in enumerate(true_values_collection):
        auroc_figure: 'plt.Figure' = large_figure_container[e][0]
        auprc_figure: 'plt.Figure' = large_figure_container[e][1]
        auroc_figure_name = tlabel_ + '_auROC.svg'
        auroc_figure.savefig(os.path.join(PLOT_SAVE_DIR, auroc_figure_name),
                             transparent=True,
                             dpi=400,
                             format='svg',
                             bbox_inches='tight')
        auprc_figure_name = tlabel_ + '_auPRC.svg'
        auprc_figure.savefig(os.path.join(PLOT_SAVE_DIR, auprc_figure_name),
                             transparent=True,
                             dpi=400,
                             format='svg',
                             bbox_inches='tight')
    plt.close()
    return None


def bar_style_metrics(predicted_values_collection: 'dict[str, pd.DataFrame]',
                      true_values_collection: 'dict[str, pd.DataFrame]',
                      figure_args_dict: 'dict[str, Any]'):
    large_figure_container: 'list[plt.Figure]' = [
        [plt.figure(figsize=figure_args_dict['figure_size'], layout='constrained')
         for _ in range(2)]
        for _ in true_values_collection.keys()]
    # (지금 상황에서는 (2, 2) 사이즈의 피규어로 구성된 리스트임.)
    for e, tlabel_ in enumerate(true_values_collection):
        this_true_labels = true_values_collection[tlabel_]
        auroc_figure: 'plt.Figure' = large_figure_container[e][0]
        auprc_figure: 'plt.Figure' = large_figure_container[e][1]
        auroc_curve_ax: plt.Axes = auroc_figure.subplots(1, 1)
        prc_curve_ax: plt.Axes = auprc_figure.subplots(1, 1)
        auroc_collection_ = []
        auprc_collection = []
        for j, pred_value_ in enumerate(predicted_values_collection):
            metrics_dict_ = calculate_classifier_metrics(
                predicted_values_collection[pred_value_], this_true_labels)
            auroc_collection_.append(metrics_dict_['roc_auroc'])
            auprc_collection.append(metrics_dict_['prc_auprc'])
        metrics_dataframe = pd.DataFrame.from_dict(
            {'Prediction method': list(predicted_values_collection.keys()),
             'auROC': auroc_collection_, 'auPRC': auprc_collection, }, orient='columns',
        )
        auroc_curve_ax.bar(metrics_dataframe['Prediction method'], metrics_dataframe['auROC'],
                           width=0.4, color=COLOR_LIST)
        prc_curve_ax.bar(metrics_dataframe['Prediction method'], metrics_dataframe['auPRC'],
                         width=0.4, color=COLOR_LIST)
        xlim_ = prc_curve_ax.get_xlim()
        prc_curve_ax.plot(xlim_, (metrics_dict_['positive_prevalence'], metrics_dict_['positive_prevalence']),
                          color='dimgrey', linestyle='dashed', linewidth=2.5)
        if tlabel_ == 'CoCoMac':
            auroc_curve_ax.set_ylim(0.5, 0.75)
            prc_curve_ax.set_ylim(0.0, 0.4)
        elif tlabel_ == 'TheVirtualBrain':
            auroc_curve_ax.set_ylim(0.5, 0.75)
            prc_curve_ax.set_ylim(0.0, 0.75)
        prc_curve_ax.set_xlim(xlim_[0], xlim_[1])
        auroc_curve_ax.bar_label(auroc_curve_ax.containers[0],
                                 fontsize=TICK_LABEL_SIZE,
                                 fmt='{:1.4f}', )
        prc_curve_ax.bar_label(prc_curve_ax.containers[0],
                               fontsize=TICK_LABEL_SIZE,
                               fmt='{:1.4f}', )
        auroc_curve_ax.set_xlabel('Prediction method', labelpad=5.0,
                                  fontdict=dict(fontsize=LABEL_SIZE,))
        auroc_curve_ax.set_ylabel('auROC', labelpad=5.0,
                                  fontdict=dict(fontsize=LABEL_SIZE,))
        prc_curve_ax.set_xlabel('Prediction method', labelpad=5.0,
                                fontdict=dict(fontsize=LABEL_SIZE,))
        prc_curve_ax.set_ylabel('auPRC', labelpad=5.0,
                                fontdict=dict(fontsize=LABEL_SIZE,))
        auroc_curve_ax.set_xticks(auroc_curve_ax.get_xticks())
        auroc_curve_ax.set_yticks(auroc_curve_ax.get_yticks())
        prc_curve_ax.set_xticks(prc_curve_ax.get_xticks())
        prc_curve_ax.set_yticks(prc_curve_ax.get_yticks())
        auroc_curve_ax.set_xticklabels(auroc_curve_ax.get_xticklabels(),
                                       fontdict=dict(fontsize=TICK_LABEL_SIZE,))
        auroc_curve_ax.set_yticklabels(auroc_curve_ax.get_yticklabels(),
                                       fontdict=dict(fontsize=TICK_LABEL_SIZE,))
        prc_curve_ax.set_xticklabels(prc_curve_ax.get_xticklabels(),
                                     fontdict=dict(fontsize=TICK_LABEL_SIZE,))
        prc_curve_ax.set_yticklabels(prc_curve_ax.get_yticklabels(),
                                     fontdict=dict(fontsize=TICK_LABEL_SIZE,))
        # auroc_curve_ax.bar(list(predicted_values_collection.keys()), auroc_collection_
        #                    )
        # prc_curve_ax.bar(list(predicted_values_collection.keys()), auprc_collection
        #                    )
    plt.show()
    for e, tlabel_ in enumerate(true_values_collection):
        auroc_figure: 'plt.Figure' = large_figure_container[e][0]
        auprc_figure: 'plt.Figure' = large_figure_container[e][1]
        auroc_figure_name = tlabel_ + '_auROC_bar.svg'
        auroc_figure.savefig(os.path.join(PLOT_SAVE_DIR, auroc_figure_name),
                             transparent=True,
                             dpi=400,
                             format='svg',
                             bbox_inches='tight')
        auprc_figure_name = tlabel_ + '_auPRC_bar.svg'
        auprc_figure.savefig(os.path.join(PLOT_SAVE_DIR, auprc_figure_name),
                             transparent=True,
                             dpi=400,
                             format='svg',
                             bbox_inches='tight')
    return None


def main():
    # 정답 불러오기
    CM_all = pd.read_csv(os.path.join(RESULTS_SAVED_DIR, 'cocomac_GS.csv'), index_col=0)
    CM_all.astype(np.float64)
    VB_all = pd.read_csv(os.path.join(
        RESULTS_SAVED_DIR, 'virtualbrain_GS.csv'), index_col=0)
    VB_all.astype(np.float64)
    true_labels_dict = {
        'CoCoMac': CM_all, 'TheVirtualBrain': VB_all,
    }
    # 예측값 불러오기
    neuronchat_adjacency = pd.read_csv(os.path.join(
        RESULTS_SAVED_DIR, 'neuronchat_predicted.csv'), index_col=0)
    neuronchat_adjacency.astype(np.float64)
    weighted_sum_mean = pd.read_csv(os.path.join(
        RESULTS_SAVED_DIR, 'ours_predicted.csv'), index_col=0)
    weighted_sum_mean.astype(np.float64)
    curve_plotter_dict = {
        'NeuronChat': neuronchat_adjacency,
        'Ours': weighted_sum_mean,
    }
    figure_plotting_args_dict = {
        'figure_size': (4, 4),
    }
    overlapped_ROC_curve_plotter(
        curve_plotter_dict, true_labels_dict, figure_plotting_args_dict)

    # 예측값 불러오기
    simple_corr_simple_average = pd.read_csv(
        os.path.join(RESULTS_SAVED_DIR, 'pcc_sum_predicted.csv'), index_col=0)
    simple_corr_simple_average.astype(np.float64)
    simple_multiplied_weighted_sum = pd.read_csv(
        os.path.join(RESULTS_SAVED_DIR, 'pcc_weighted_predicted.csv'), index_col=0)
    simple_multiplied_weighted_sum.astype(np.float64)
    our_method_simple_average = pd.read_csv(
        os.path.join(RESULTS_SAVED_DIR, 'ours_sum_predicted.csv'), index_col=0)
    our_method_simple_average.astype(np.float64)
    final_dict = {
        'NeuronChat': neuronchat_adjacency,
        'PCC_sum':  simple_corr_simple_average,
        'PCC_weighted': simple_multiplied_weighted_sum,
        'Ours_sum': our_method_simple_average,
        'Ours': weighted_sum_mean,
    }
    figure_plotting_args_dict = {
        'figure_size': (6, 3),
    }
    bar_style_metrics(final_dict, true_labels_dict, figure_plotting_args_dict)


if __name__ == '__main__':
    print(datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Initializing...")
    main()
    print(datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Finished!")
