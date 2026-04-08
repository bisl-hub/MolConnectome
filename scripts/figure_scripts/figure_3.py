import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import itertools
import datetime
import sys
import matplotlib.colors

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from functools import reduce
from typing import Any


sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency"))
import core_config
import plot_config
import plot_utils



RESULTS_SAVED_DIR = os.path.join(PROJECT_ROOT, "test", "figure_inputs")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "test", "plots")
BLUE = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['royalblue'])
RED = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['crimson'])
GREEN = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['lime'])
NETWORK_CMAP = plot_utils.make_complicated_oklch_colormap(BLUE, RED, GREEN, chroma=True)
NOT_TO_DRAW = ['weighted_DA_13', 'weighted_5-HT_1', 'weighted_5-HT_16', 
               'weighted_Glu_11', 'weighted_GABA_7', 'weighted_Syn_11', 'weighted_Syn_21',]


def load_significant_differences():
    significance_file = os.path.join(RESULTS_SAVED_DIR, 'human_differences.csv')
    significants_ = pd.read_csv(significance_file)
    if 'Unnamed: 0' in significants_.columns:
        significants_.set_index('Unnamed: 0', inplace=True)
        significants_.index.name = None
    node_order = [x.split('_')[2] for x in significants_.index]
    significants_['node_order'] = np.array(node_order).astype(int)
    significants_ = significants_[['source','target','difference','significance', 'node_order']]
    return significants_


def draw_pentagon_multiple_edges(figure_object: 'plt.Figure | plt.SubFigure',
                                 edge_data: pd.DataFrame, 
                                 network_drawing_args_dict: dict[str, Any]):
    '''변수 할당'''
    figure_size = network_drawing_args_dict['figure_size']
    figure_title_string = network_drawing_args_dict['figure_title_string']
    # figure_filename = network_drawing_args_dict['figure_save_file_name']
    rotation = network_drawing_args_dict['rotate_position'] # 어디로 몇도만큼 돌릴지. 각도로 표기되어야 함.
    '''차이가 나야 하는 부분, 이전에는 M-by-M 모양의 adjacency matrix 를 그렸다면,
    이번에는 long-form (즉 meltdown 된) 데이터를 그려야 함.'''
    adj_values = edge_data['difference'].to_numpy().flatten()
    adj_values_notnan = adj_values[~np.isnan(adj_values)]
    max_adj_value = adj_values_notnan.max()
    min_adj_value = adj_values_notnan.min()
    if network_drawing_args_dict['colormap_interval']:
        (min_adj_value, max_adj_value) = network_drawing_args_dict['colormap_interval']
    network_drawing_args_dict['colormap_rescaling'] = \
        lambda x: (x - min_adj_value) / (max_adj_value - min_adj_value)
    remained_nodes = np.unique(np.concatenate([edge_data['source'].to_numpy(), edge_data['target'].to_numpy(), ]))
    # 네트워크 만들기
    node_list = [x for x in core_config.HUMAN_REGIONS_TO_DRAW if x in remained_nodes]
    G = nx.Graph(); G.add_nodes_from(node_list)
    positions = nx.circular_layout(G); # print(positions);
    if rotation:
        rotate_degree = (np.pi) * (rotation / 180) # default for triangle is -30 degree
        rotation_mat = np.array([[np.cos(rotate_degree), -np.sin(rotate_degree)], 
                    [np.sin(rotate_degree), np.cos(rotate_degree)]])
        positions = {x: np.matmul(rotation_mat, positions[x]) for x in positions}; # print(positions)
    nodelist_to_draw = [x for x in node_list if x in remained_nodes] # 없는 것들 제거!
    #! 그림 그리기
    large_figure_obj = figure_object
    large_figure_obj.set_dpi(400)
    ax_obj = large_figure_obj.subplots(1, 1)
    font_weight_dict = {x: plot_config.human_network_style_dict[x]['labelcolor'] for x in node_list }
    node_color_list_to_draw = [plot_config.human_network_style_dict[x]['facecolor'] for x in nodelist_to_draw ]
    nx.draw_networkx_nodes(G, positions, 
                           nodelist = nodelist_to_draw, node_color = node_color_list_to_draw,
                           margins=0.1, node_size=network_drawing_args_dict['node_size'], edgecolors='black', 
                           ax=ax_obj)
    nx.draw_networkx_labels(G, positions, 
                            font_size = network_drawing_args_dict['node_label_size'],
                            font_color = font_weight_dict,
                            ax=ax_obj)
    edge_radian_dict_three = {
        0: 0.0, 1: -0.2, 2: 0.2
    }
    edge_radian_dict_two = {
        0: 0.1, 1: 0.1,
    }
    edge_radian_dict_two_reverse = {
        0: -0.1, 1: 0.1,
    }
    for perm_ in itertools.combinations(node_list, 2): # for pair of regions
        try:
            edges_1 = edge_data.groupby('source').get_group(perm_[0]).groupby('target').get_group(perm_[1])
            edges_2 = edge_data.groupby('source').get_group(perm_[1]).groupby('target').get_group(perm_[0])
            edges_merged = pd.concat([edges_1, edges_2], axis=0)
        except KeyError:
            try:
                edges_1 = edge_data.groupby('source').get_group(perm_[0]).groupby('target').get_group(perm_[1])
                edges_merged = edges_1
            except KeyError:
                try:
                    edges_2 = edge_data.groupby('source').get_group(perm_[1]).groupby('target').get_group(perm_[0])
                    edges_merged = edges_2
                except KeyError: # 둘 다 없는 경우.
                    edges_merged = None
        if isinstance(edges_merged, pd.DataFrame):
            # print(edges_merged)
            for e, index_ in enumerate(edges_merged.index):
                node_pair = edges_merged.loc[index_][['source', 'target']].to_numpy().tolist()
                delta_conn = edges_merged.loc[index_]['difference']
                significance = edges_merged.loc[index_]['significance']
                diff_pvalue = 1 - significance if significance > 0.5 + 1e-8 else significance
                edge_thickness = np.negative(np.log10(diff_pvalue))
                edge_class = index_.split('_')[1]
                if edge_class != 'Syn':
                    difference_descriptor = edge_class + ', \u0394=' +  f'{delta_conn:1.3f}'
                else:
                    difference_descriptor = 'Syn, ' + '\u0394=' +  f'{delta_conn:1.3f}'
                # print(node_pair, delta_conn)
                if edges_merged.shape[0] == 1:
                    if (node_pair == ['HP', 'NAc']):
                        network_drawing_args_dict['edge_curve_radian'] = -0.15
                    elif (node_pair == ['CN', 'CC']):
                        network_drawing_args_dict['edge_curve_radian'] = 0.0 # #FF0000
                    else:
                        network_drawing_args_dict['edge_curve_radian'] = 0.0
                        # network_drawing_args_dict['edge_curve_radian'] = 0.0
                elif edges_merged.shape[0] == 2:
                    # print(node_pair, perm_, e)
                    if tuple(node_pair) == perm_: # 정방향
                        network_drawing_args_dict['edge_curve_radian'] = edge_radian_dict_two[e]
                    else: # 역방향, 원복.
                        network_drawing_args_dict['edge_curve_radian'] = edge_radian_dict_two_reverse[e]
                elif edges_merged.shape[0] == 3:
                    if tuple(node_pair) == perm_: # 정방향
                        network_drawing_args_dict['edge_curve_radian'] = 0.0
                    else: # 역방향, 원복.
                        network_drawing_args_dict['edge_curve_radian'] = edge_radian_dict_three[e]
                plot_utils.draw_non_self_loop_edge(G, positions, node_pair, difference_descriptor, 
                                             edge_thickness**2, ax_obj, network_drawing_args_dict)
        else:
            pass
    ax_obj.set_xlim(-1.3, 1.3) 
    ax_obj.set_ylim(-1.3, 1.3)
    ax_obj.spines['top'].set_visible(True)
    ax_obj.spines['bottom'].set_visible(True)
    ax_obj.spines['left'].set_visible(True)
    ax_obj.spines['right'].set_visible(True)
    return None


def draw_vertical_cbar(fig_obj: 'plt.Figure | plt.SubFigure', 
                        min_adj_value: float,
                        max_adj_value: float,
                        colormap: 'matplotlib.colors.ListedColormap',
                        tick_numbers = 4
                        ) -> None:
    cbar_subplot = fig_obj.subplots(1, 1)
    this_colormap = colormap
    fig_obj.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=this_colormap), 
                        cax=cbar_subplot)
    tick_array = np.linspace(min_adj_value, max_adj_value, tick_numbers)
    tick_label_array = [f'{tick_:1.3f}' for tick_ in tick_array]
    tick_pos_array = np.linspace(0.0, 1.0, tick_numbers)
    cbar_subplot.set_yticks(tick_pos_array, labels = tick_label_array)
    cbar_subplot.tick_params(axis='y', labelsize=plot_config.SUBTITLE_SIZE)
    return fig_obj


def main():
    sample_shuffled_differences_data = load_significant_differences()
    # ! 포지티브랑 네가티브 나누기
    significance_threshold = 0.05
    no_sums_idx = [x for x in sample_shuffled_differences_data.index
                          if x.split('_')[1] not in ['sum',]]
    no_sums = sample_shuffled_differences_data.loc[no_sums_idx]
    positive_conns = no_sums[(no_sums['significance'] < significance_threshold)]
    negative_conns = no_sums[(no_sums['significance'] > (1-significance_threshold))]
    positive_index_to_draw = [x for x in positive_conns.index if x not in NOT_TO_DRAW]
    negative_index_to_draw = [x for x in negative_conns.index if x not in NOT_TO_DRAW]
    positive_data = positive_conns.loc[positive_index_to_draw]
    negative_data = negative_conns.loc[negative_index_to_draw]

    '''이제 이거로 그려야 할 것은 다음과 같음.
    1. 파싱 해서, "동일한" 커넥션일 경우에는 arc의 radian을 차이나게 해서 그리기.
    2. 각각의 엣지에 이름 넣기 - 예를 들어, DA: -1.5 와 같은 느낌으로.
    3. 구조 변경 - 두개의 피규어는 서브피규어로 묶어야 함. 이유는 민-맥스 코넥션 값을 보여 줘야 하기 때문.
    '''
    network_drawing_args_dict = {
        'edge_curve_radian': 0.1,
        'deafult_edge_width': 0.1,
        'edge_alpha': 1.0,
        'network_cmap': NETWORK_CMAP,
        'draw_edge_label': True,
        'node_size': 3500, 
        'rotate_position': 90,
        'figure_size' : (15, 7),
        'node_label_size': plot_config.TITLE_SIZE, 
        'edge_caption_size': plot_config.CAPTION_SIZE + 4.5,
    }
    min_conn = np.nanmin([np.nanmin(positive_data['difference'].to_numpy().flatten()), 
                       np.nanmin(negative_data['difference'].to_numpy().flatten())])
    max_conn = np.nanmax([np.nanmax(positive_data['difference'].to_numpy().flatten()), 
                       np.nanmax(negative_data['difference'].to_numpy().flatten())])
    network_drawing_args_dict['colormap_interval'] = (min_conn, max_conn)

    large_figure_object = plt.figure(figsize=network_drawing_args_dict['figure_size'], layout='constrained')
    subfigures_array = large_figure_object.subfigures(1, 3, width_ratios=[7, 7, 1], wspace=0.01)
    positive_figure_object: 'plt.Figure' = subfigures_array[0]; 
    network_drawing_args_dict['figure_title_string'] = 'A'
    # network_drawing_args_dict['figure_save_file_name'] = 'figure_3a_NT_differences.svg'
    draw_pentagon_multiple_edges(positive_figure_object, positive_data, network_drawing_args_dict)
    positive_figure_object.suptitle(network_drawing_args_dict['figure_title_string'], x=0.03, y=0.97, 
                    horizontalalignment='left',
                    verticalalignment='top', multialignment='center',
                    fontsize=plot_config.TITLE_SIZE + 2,
                    fontweight='bold',)
    negative_figure_object: 'plt.Figure' = subfigures_array[1]; 
    network_drawing_args_dict['figure_title_string'] = 'B'
    # network_drawing_args_dict['figure_save_file_name'] = 'figure_3b_synapse_differences.svg'
    draw_pentagon_multiple_edges(negative_figure_object, negative_data, network_drawing_args_dict)
    negative_figure_object.suptitle(network_drawing_args_dict['figure_title_string'], x=0.03, y=0.97, 
                    horizontalalignment='left',
                    verticalalignment='top', multialignment='center',
                    fontsize=plot_config.TITLE_SIZE + 2,
                    fontweight='bold',)
    draw_vertical_cbar(subfigures_array[2], min_conn, max_conn, NETWORK_CMAP)

    # large_figure_object.suptitle(network_args_dict['figure_title_string'], x=-0.02, y=1.00, 
    #                 horizontalalignment='right',
    #                 verticalalignment='top', multialignment='center',
    #                 fontsize=plot_config.TITLE_SIZE + 2,
    #                 fontweight='bold',)
    # figure_title = network_args_dict['figure_save_file_name']
    plt.show()
    figure_title = 'figure_3_left_positive_right_negative.svg'
    large_figure_object.savefig(os.path.join(PLOT_SAVE_DIR, figure_title),
                transparent=True, dpi=400, format='svg', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Initializing...")
    main()
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Finished!")
