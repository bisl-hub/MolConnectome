import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import itertools
import datetime
import sys
import pickle

import matplotlib.colors
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Any


sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency"))
import plot_config
import plot_utils
import core_config


RESULTS_SAVED_DIR = os.path.join(PROJECT_ROOT, "test", "figure_inputs")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "test", "plots")
BLUE = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['royalblue'])
RED = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['crimson'])
GREEN = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['lime'])
NETWORK_CMAP = plot_utils.make_complicated_oklch_colormap(BLUE, RED, GREEN, chroma=True)
RESULTS_SAVED_DIR = os.path.join(PROJECT_ROOT, "test", "figure_inputs")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "test", "plots")
XLIM = (-1.3, 1.3); YLIM = (-1.3, 1.5); TITLE_POSITION = (0.5, 0.97)


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


def draw_pentagon(figure_object: 'plt.Figure | plt.SubFigure',
                  adjacency_matrix: pd.DataFrame, 
                  network_drawing_args_dict: dict[str, Any]):
    '''변수 할당'''
    figure_title_string = network_drawing_args_dict['figure_title_string']
    rotation = network_drawing_args_dict['rotate_position'] # 어디로 몇도만큼 돌릴지. 각도로 표기되어야 함.
    adj_values = adjacency_matrix.to_numpy().flatten()
    adj_values_notnan = adj_values[~np.isnan(adj_values)]
    max_adj_value = adj_values_notnan.max()
    min_adj_value = adj_values_notnan.min()
    if network_drawing_args_dict['colormap_interval']:
        (min_adj_value, max_adj_value) = network_drawing_args_dict['colormap_interval']
    network_drawing_args_dict['colormap_rescaling'] = \
        lambda x: (x - min_adj_value) / (max_adj_value - min_adj_value)
    # 네트워크 만들기
    node_list = adjacency_matrix.index # #FF0000
    node_list = [x for x in core_config.HUMAN_REGIONS_TO_DRAW if x in node_list] # #FF0000
    G = nx.Graph(); G.add_nodes_from(node_list)
    positions = nx.circular_layout(G); # print(positions);
    if rotation:
        rotate_degree = (np.pi) * (rotation / 180) # default for triangle is -30 degree
        rotation_mat = np.array([[np.cos(rotate_degree), -np.sin(rotate_degree)], 
                    [np.sin(rotate_degree), np.cos(rotate_degree)]])
        positions = {x: np.matmul(rotation_mat, positions[x]) for x in positions}; # print(positions)
    nodelist_to_draw = [x for x in node_list 
                        if x in (np.union1d(adjacency_matrix.dropna(how='all').index, 
                                 adjacency_matrix.T.dropna(how='all').index))] # 없는 것들 제거!
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
    node_list = list(adjacency_matrix.index)
    for perm_ in itertools.product(node_list, node_list): # for pair of regions
        # 여기에서 이제 파싱해온 엣지 웨이트를 사용해야 함.
        structual_connectivity = adjacency_matrix.loc[perm_[0], perm_[1]] # will be a float
        if -1e-8 < structual_connectivity < 1e-8 : # 0인 경우
            pass
        elif np.isnan(structual_connectivity): # 없는 경우
            pass
        else:
            plot_utils.draw_non_self_loop_edge(G, positions, perm_, structual_connectivity, 
                                         2.0, ax_obj, network_drawing_args_dict)
    ax_obj.set_xlim(*XLIM); ax_obj.set_ylim(*YLIM)
    ax_obj.spines['top'].set_visible(True)
    ax_obj.spines['bottom'].set_visible(True)
    ax_obj.spines['left'].set_visible(True)
    ax_obj.spines['right'].set_visible(True)
    ax_obj.text(*TITLE_POSITION, figure_title_string, 
                        transform=ax_obj.transAxes,
                        horizontalalignment='center',
                        verticalalignment='top', multialignment='center',
                        fontsize=plot_config.SUBTITLE_SIZE + 2,
                        fontweight='bold',
                        )
    return None


def draw_significant_difference(figure_object: 'plt.Figure | plt.SubFigure',
                                 edge_data: pd.DataFrame, 
                                 network_drawing_args_dict: dict[str, Any]):
    '''변수 할당'''
    rotation = network_drawing_args_dict['rotate_position'] # 어디로 몇도만큼 돌릴지. 각도로 표기되어야 함.
    figure_title_string = network_drawing_args_dict['figure_title_string']

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
    node_list = core_config.HUMAN_REGIONS_TO_DRAW  # #FF0000
    G = nx.Graph(); G.add_nodes_from(node_list)
    positions = nx.circular_layout(G); # print(positions);
    if rotation:
        rotate_degree = (np.pi) * (rotation / 180) # default for triangle is -30 degree
        rotation_mat = np.array([[np.cos(rotate_degree), -np.sin(rotate_degree)], 
                    [np.sin(rotate_degree), np.cos(rotate_degree)]])
        positions = {x: np.matmul(rotation_mat, positions[x]) for x in positions}; # print(positions)
    nodelist_to_draw = [x for x in node_list if x in remained_nodes] # 없는 것들 제거!
    nodelist_to_draw = core_config.HUMAN_REGIONS_TO_DRAW  # #FF0000
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
            for e, index_ in enumerate(edges_merged.index):
                node_pair = edges_merged.loc[index_][['source', 'target']].to_numpy().tolist()
                delta_conn = edges_merged.loc[index_]['difference']
                significance = edges_merged.loc[index_]['significance']
                sign__ = '+' if delta_conn > 1e-8 else ''
                diff_pvalue = 1 - significance if significance > 0.5 + 1e-8 else significance
                edge_thickness = np.negative(np.log10(diff_pvalue))
                edge_class = index_.split('_')[1]
                awesome_format = np.format_float_scientific(diff_pvalue, precision=3)
                difference_descriptor = '\u0394=' + sign__ + f'{delta_conn:1.3f}' + '\n' + 'p=' + awesome_format
                # print(node_pair, delta_conn)
                if edge_class == 'Syn' and node_pair == ['CC', 'PFC']:
                    network_drawing_args_dict['edge_curve_radian'] = -0.1
                elif edge_class == 'Syn' and node_pair == ['NAc', 'CC']:
                    network_drawing_args_dict['edge_curve_radian'] = -0.1
                elif edge_class == 'Syn' and node_pair == ['CC', 'HP']:
                    network_drawing_args_dict['edge_curve_radian'] = 0.15
                elif edge_class == 'Syn' and node_pair == ['HP', 'CC']:
                    network_drawing_args_dict['edge_curve_radian'] = 0.15
                elif edge_class == 'GABA' and node_pair == ['CC', 'PFC']:
                    network_drawing_args_dict['edge_curve_radian'] = -0.1
                else:
                    network_drawing_args_dict['edge_curve_radian'] = 0.1
                plot_utils.draw_non_self_loop_edge(G, positions, node_pair, difference_descriptor, 
                                             edge_thickness**2, ax_obj, network_drawing_args_dict)
        else:
            pass

    '''# edge_radian_dict_three = {
    #     0: 0.0, 1: -0.1, 2: 0.1
    # }
    # edge_radian_dict_two = {
    #     0: 0.1, 1: 0.1,
    # }
    # edge_radian_dict_two_reverse = {
    #     0: -0.1, 1: 0.1,
    # }
    # for perm_ in itertools.combinations(node_list, 2): # for pair of regions
    #     try:
    #         edges_1 = edge_data.groupby('source').get_group(perm_[0]).groupby('target').get_group(perm_[1])
    #         edges_2 = edge_data.groupby('source').get_group(perm_[1]).groupby('target').get_group(perm_[0])
    #         edges_merged = pd.concat([edges_1, edges_2], axis=0)
    #     except KeyError:
    #         try:
    #             edges_1 = edge_data.groupby('source').get_group(perm_[0]).groupby('target').get_group(perm_[1])
    #             edges_merged = edges_1
    #         except KeyError:
    #             try:
    #                 edges_2 = edge_data.groupby('source').get_group(perm_[1]).groupby('target').get_group(perm_[0])
    #                 edges_merged = edges_2
    #             except KeyError: # 둘 다 없는 경우.
    #                 edges_merged = None
    #     if isinstance(edges_merged, pd.DataFrame):
    #         # print(edges_merged)
    #         for e, index_ in enumerate(edges_merged.index):
    #             node_pair = edges_merged.loc[index_][['source', 'target']].to_numpy().tolist()
    #             delta_conn = edges_merged.loc[index_]['difference']
    #             edge_class = index_.split('_')[1]
    #             if edge_class != 'Syn':
    #                 difference_descriptor = edge_class + ', \u0394=' +  f'{delta_conn:1.3f}'
    #             else:
    #                 difference_descriptor = '\u0394=' +  f'{delta_conn:1.3f}'
    #             # print(node_pair, delta_conn)
    #             if edges_merged.shape[0] == 1:
    #                 if edge_class != 'Syn':
    #                     if (node_pair == ['HP', 'NAc']):
    #                         network_drawing_args_dict['edge_curve_radian'] = -0.1
    #                     elif (node_pair == ['HP', 'CN']):
    #                         network_drawing_args_dict['edge_curve_radian'] = 0.1
    #                     else:
    #                         network_drawing_args_dict['edge_curve_radian'] = 0.0
    #                 else:
    #                     network_drawing_args_dict['edge_curve_radian'] = 0.0
    #             elif edges_merged.shape[0] == 2:
    #                 # print(node_pair, perm_, e)
    #                 if tuple(node_pair) == perm_: # 정방향
    #                     network_drawing_args_dict['edge_curve_radian'] = edge_radian_dict_two[e]
    #                 else: # 역방향, 원복.
    #                     network_drawing_args_dict['edge_curve_radian'] = edge_radian_dict_two_reverse[e]
    #             elif edges_merged.shape[0] == 3:
    #                 if tuple(node_pair) == perm_: # 정방향
    #                     network_drawing_args_dict['edge_curve_radian'] = 0.0
    #                 else: # 역방향, 원복.
    #                     network_drawing_args_dict['edge_curve_radian'] = edge_radian_dict_three[e]
    #             plot_utils.draw_non_self_loop_edge(G, positions, node_pair, 
    #                                          difference_descriptor, ax_obj, network_drawing_args_dict)
    #     else:
    #         pass'''
    ax_obj.set_xlim(*XLIM); ax_obj.set_ylim(*YLIM)
    ax_obj.spines['top'].set_visible(True)
    ax_obj.spines['bottom'].set_visible(True)
    ax_obj.spines['left'].set_visible(True)
    ax_obj.spines['right'].set_visible(True)
    ax_obj.text(*TITLE_POSITION, figure_title_string, 
                    transform=ax_obj.transAxes,
                    horizontalalignment='center',
                    verticalalignment='top', multialignment='center',
                    fontsize=plot_config.SUBTITLE_SIZE + 2,
                    fontweight='bold',
                    )
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
    tick_array = np.linspace(min_adj_value, max_adj_value, tick_numbers)  # #FF0000
    tick_label_array = [f'{tick_:1.3f}' for tick_ in tick_array]
    tick_position_array = np.linspace(0.0, 1.0, tick_numbers)  # #FF0000
    cbar_subplot.set_yticks(tick_position_array, labels = tick_label_array)
    cbar_subplot.tick_params(axis='y', labelsize=plot_config.SUBTITLE_SIZE)
    return fig_obj


def draw_pentagon_wrapper(control_adjacency_matrix: 'pd.DataFrame', 
                          schizo_adjacency_matrix: 'pd.DataFrame', 
                          network_args_dict: 'dict[str, Any]'):
    if network_args_dict['NT_system_name'] in ['ACh', 'NP']:
        min_conn = np.nanmin([np.nanmin(control_adjacency_matrix.to_numpy().flatten()), 
                        np.nanmin(schizo_adjacency_matrix.to_numpy().flatten()), 
                        ])
        max_conn = np.nanmax([np.nanmax(control_adjacency_matrix.to_numpy().flatten()), 
                        np.nanmax(schizo_adjacency_matrix.to_numpy().flatten()),
                        ])
    else:
        sample_shuffled_differences_data = load_significant_differences()
        transmitters_index = [x for x in sample_shuffled_differences_data.index
                            if x.split('_')[1] == network_args_dict['NT_system_name']]
        neurotransmitter_data = sample_shuffled_differences_data.loc[transmitters_index]
        significance_threshold = 0.05
        significant_NT_conns = neurotransmitter_data[
            (neurotransmitter_data['significance'] > (1-significance_threshold)) | 
            (neurotransmitter_data['significance'] < significance_threshold)]
        min_conn = np.nanmin([np.nanmin(control_adjacency_matrix.to_numpy().flatten()), 
                        np.nanmin(schizo_adjacency_matrix.to_numpy().flatten()), 
                        np.nanmin(significant_NT_conns['difference'].to_numpy().flatten()), 
                        ])
        max_conn = np.nanmax([np.nanmax(control_adjacency_matrix.to_numpy().flatten()), 
                        np.nanmax(schizo_adjacency_matrix.to_numpy().flatten()),
                        np.nanmax(significant_NT_conns['difference'].to_numpy().flatten()), 
                        ])
    network_args_dict['colormap_interval'] = (min_conn, max_conn)
    large_figure_object = plt.figure(figsize= network_args_dict['figure_size'], layout='constrained')
    subfigures_array = large_figure_object.subfigures(1, 4, width_ratios=[7, 7, 7, 1], wspace=0.01)
    network_args_dict['figure_title_string'] = 'Control'
    draw_pentagon(subfigures_array[0], control_adjacency_matrix, network_args_dict)
    network_args_dict['figure_title_string'] = 'Schizophrenia'
    draw_pentagon(subfigures_array[2],schizo_adjacency_matrix, network_args_dict)
    if network_args_dict['NT_system_name'] in ['ACh', 'NP']:
        network_args_dict['figure_title_string'] = figure_title_string = 'No significant differences'
        temp_ax: 'plt.Axes' = subfigures_array[1].subplots(1, 1)
        temp_ax.text(0.5, 0.5, figure_title_string, 
                    transform=temp_ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='top', multialignment='center',
                    fontsize=plot_config.SUBTITLE_SIZE + 2,
                    fontweight='bold',
                    )
        temp_ax.get_xaxis().set_visible(False)
        temp_ax.get_yaxis().set_visible(False)
    else:
        network_args_dict['figure_title_string'] = 'Connectivity difference'
        draw_significant_difference(subfigures_array[1], significant_NT_conns, network_args_dict)
    draw_vertical_cbar(subfigures_array[3], min_conn, max_conn, NETWORK_CMAP)
    large_figure_object.suptitle(network_args_dict['large_figure_title_string'], x=-0.01, y=1.00, 
                    horizontalalignment='right',
                    verticalalignment='top', multialignment='center',
                    fontsize=plot_config.TITLE_SIZE + 8,
                    fontweight='bold',)
    figure_title = network_args_dict['figure_save_file_name']
    large_figure_object.savefig(os.path.join(PLOT_SAVE_DIR, figure_title),
                transparent=True, dpi=400, format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
    return None


def main():
    network_drawing_args_dict = {
        'edge_curve_radian': 0.1,
        'deafult_edge_width': 2.5,
        'edge_alpha': 1.0,
        'network_cmap': NETWORK_CMAP,
        'draw_edge_label': True,
        'node_size': 3500, 
        'rotate_position': 90,
        'node_label_size': plot_config.TITLE_SIZE, 
        'edge_caption_size': plot_config.CAPTION_SIZE + 4.5,
        'figure_size' : (22, 7), # 3 in a row
    }
    figure_title_dict = {
        x : chr(x+ 65) for x in range(8)
    }
    with open(file=os.path.join(RESULTS_SAVED_DIR, 'weighted_seperated.pickle'), mode='rb') as f:
        weighted_network_dict = pickle.load(f)
    for e, gene_set_name in enumerate(weighted_network_dict.keys()):
        weighted_control = weighted_network_dict[gene_set_name]['control']
        weighted_schizo = weighted_network_dict[gene_set_name]['schizo']
        network_drawing_args_dict['NT_system_name'] = gene_set_name.split('_')[1];  # 'Glu'
        network_drawing_args_dict['NT_full_name'] = core_config.NEUROTRANSMITTER_NAME_DICT[gene_set_name.split('_')[1]]; # 'Glutamate' 
        network_drawing_args_dict['large_figure_title_string'] = figure_title_dict[e]
        network_drawing_args_dict['figure_save_file_name'] = 'figure_S3_seperated_' + gene_set_name.split('_')[1] + '.svg'
        draw_pentagon_wrapper(weighted_control, weighted_schizo, network_drawing_args_dict)


if __name__ == '__main__':
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Initializing...")
    main()
    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print("Finished!")
