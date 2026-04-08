import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys

import matplotlib.colors

import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from typing import Any


sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency", "downloaded", "python-oklch", "src"))
import oklch as pokl
import plot_config


BLUE = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['royalblue'])
RED = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['crimson'])
GREEN = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['lime'])
WHITE = matplotlib.colors.to_hex(matplotlib.colors.CSS4_COLORS['white'])


def make_simple_oklch_colormap(start_color, end_color='#FFFFFF', 
                               lightness = 'none', chroma = 'none'):
    start_hex_code = matplotlib.colors.to_hex(start_color); start_object = pokl.HEX(start_hex_code)
    end_hex_code = matplotlib.colors.to_hex(end_color); end_object = pokl.HEX(end_hex_code)
    start_oklch = start_object.to_OKLCH(); end_oklch = end_object.to_OKLCH()

    if lightness == 'mean':
        middle_lightness = (start_oklch.l + end_oklch.l ) / 2
        start_oklch.l = middle_lightness; end_oklch.l = middle_lightness; 
    elif lightness == 'start':
        start_oklch.l = start_oklch.l; end_oklch.l = start_oklch.l;
    elif lightness == 'end':
        start_oklch.l = end_oklch.l; end_oklch.l = end_oklch.l;
    elif lightness == 'min':
        start_oklch.l = min(start_oklch.l, end_oklch.l); end_oklch.l = min(start_oklch.l, end_oklch.l);
    elif lightness == 'max':
        start_oklch.l = max(start_oklch.l, end_oklch.l); end_oklch.l = max(start_oklch.l, end_oklch.l);
    else:
        pass

    if chroma == 'mean':
        middle_chroma = (start_oklch.c + end_oklch.c ) / 2
        start_oklch.c = middle_chroma; end_oklch.c = middle_chroma;
    elif chroma == 'start':
        start_oklch.c = start_oklch.c; end_oklch.c = start_oklch.c;
    elif chroma == 'end':
        start_oklch.c = end_oklch.c; end_oklch.c = end_oklch.c;
    elif chroma == 'min':
        start_oklch.c = min(start_oklch.c, end_oklch.c); end_oklch.c = min(start_oklch.c, end_oklch.c);
    elif chroma == 'max':
        start_oklch.c = max(start_oklch.c, end_oklch.c); end_oklch.c = max(start_oklch.c, end_oklch.c);
    else:
        pass

    start_oklch = pokl.gamut_clip_hue_dependent(start_oklch)
    end_oklch = pokl.gamut_clip_hue_dependent(end_oklch)
    linear_space = [
        pokl.interpolate(x, start_oklch, end_oklch) for x in np.linspace(0, 1, 256)
    ]
    linear_hex_codes = [str(x.to_HEX()) for x in linear_space]
    rgb_array = [matplotlib.colors.to_rgb(x) for x in linear_hex_codes]
    rgb_array_final = rgb_array
    colormap = matplotlib.colors.ListedColormap(rgb_array_final, N=256)
    return colormap


def make_complicated_oklch_colormap(start_color, end_color, middle_color='#FFFFFF',
                                    lightness = 'none', chroma = 'none'):
    start_hex_code = matplotlib.colors.to_hex(start_color); start_object = pokl.HEX(start_hex_code)
    end_hex_code = matplotlib.colors.to_hex(end_color); end_object = pokl.HEX(end_hex_code)
    middle_hex_code = matplotlib.colors.to_hex(middle_color); middle_object = pokl.HEX(middle_hex_code)
    start_oklch = start_object.to_OKLCH(); end_oklch = end_object.to_OKLCH(); middle_oklch = middle_object.to_OKLCH()

    if lightness == 'mean' or lightness is True:
        middle_lightness = (start_oklch.l + end_oklch.l + middle_oklch.l) / 3
        start_oklch.l = middle_lightness; end_oklch.l = middle_lightness; middle_oklch.l = middle_lightness; 
    elif lightness == 'start':
        start_oklch.l = start_oklch.l; end_oklch.l = start_oklch.l; middle_oklch.l = start_oklch.l;
    elif lightness == 'end':
        start_oklch.l = end_oklch.l; end_oklch.l = end_oklch.l; middle_oklch.l = end_oklch.l;
    elif lightness == 'middle':
        start_oklch.l = middle_oklch.l; end_oklch.l = middle_oklch.l; middle_oklch.l = middle_oklch.l;
    elif lightness == 'min':
        min_l = min(start_oklch.l, end_oklch.l, middle_oklch.l)
        start_oklch.l = min_l; end_oklch.l = min_l; middle_oklch.l = min_l;
    elif lightness == 'max':
        max_l = max(start_oklch.l, end_oklch.l, middle_oklch.l)
        start_oklch.l = max_l; end_oklch.l = max_l; middle_oklch.l = max_l;
    else:
        pass

    if chroma == 'mean' or chroma is True:
        middle_chroma = (start_oklch.c + end_oklch.c + middle_oklch.c) / 3
        start_oklch.c = middle_chroma; end_oklch.c = middle_chroma; middle_oklch.c = middle_chroma; 
    elif chroma == 'start':
        start_oklch.c = start_oklch.c; end_oklch.c = start_oklch.c; middle_oklch.c = start_oklch.c;
    elif chroma == 'end':
        start_oklch.c = end_oklch.c; end_oklch.c = end_oklch.c; middle_oklch.c = end_oklch.c;
    elif chroma == 'middle':
        start_oklch.c = middle_oklch.c; end_oklch.c = middle_oklch.c; middle_oklch.c = middle_oklch.c;
    elif chroma == 'min':
        min_c = min(start_oklch.c, end_oklch.c, middle_oklch.c)
        start_oklch.c = min_c; end_oklch.c = min_c; middle_oklch.c = min_c;
    elif chroma == 'max':
        max_c = max(start_oklch.c, end_oklch.c, middle_oklch.c)
        start_oklch.c = max_c; end_oklch.c = max_c; middle_oklch.c = max_c;
    else:
        pass
    start_oklch = pokl.gamut_clip_hue_dependent(start_oklch)
    end_oklch = pokl.gamut_clip_hue_dependent(end_oklch)
    middle_oklch = pokl.gamut_clip_hue_dependent(middle_oklch)
    first_linear_space = [
            pokl.interpolate(x, start_oklch, middle_oklch) for x in np.linspace(0, 1, 256)
        ]
    first_linear_hex_codes = [str(x.to_HEX()) for x in first_linear_space]
    first_rgb_array = [matplotlib.colors.to_rgb(x) for x in first_linear_hex_codes]
    second_linear_space = [
            pokl.interpolate(x, end_oklch, middle_oklch) for x in np.linspace(0, 1, 256)
        ]
    second_linear_hex_codes = [str(x.to_HEX()) for x in second_linear_space]
    second_rgb_array = [matplotlib.colors.to_rgb(x) for x in second_linear_hex_codes]
    rgb_array_final = first_rgb_array + list(reversed(second_rgb_array))[1:]
    colormap = matplotlib.colors.ListedColormap(rgb_array_final, N=511)
    return colormap



def make_custom_cmap_dict() -> 'dict[str, matplotlib.colors.Colormap]':
    """returns custom cmap that are not implemented in matplotlib.colormaps

    Returns:
        dict[str, matplotlib.colors.Colormap]: dict of mpl colormaps. \n
        keys are : ['teals', 'salmons', 'teal_salmon', 'olives', 'violets', 'olive_violet']. \n
        each colormaps are callable by float ranging 0.0 ~ 1.0. ex) `custom_cmap_dict['violets'](0.34)`
    """
    teal_rgb_value = matplotlib.colors.to_rgb(
        matplotlib.colors.CSS4_COLORS['teal'])
    salmon_rgb_value = matplotlib.colors.to_rgb(
        matplotlib.colors.CSS4_COLORS['salmon'])
    white_rgb_value = matplotlib.colors.to_rgb(
        matplotlib.colors.CSS4_COLORS['white'])
    olive_rgb_value = matplotlib.colors.to_rgb(
        matplotlib.colors.CSS4_COLORS['olive'])
    violet_rgb_value = matplotlib.colors.to_rgb(
        matplotlib.colors.CSS4_COLORS['mediumorchid'])
    lightsalmon_rgb_value = matplotlib.colors.to_rgb(
        matplotlib.colors.CSS4_COLORS['lightsalmon'])
    crimson_rgb_value = matplotlib.colors.to_rgb(
        matplotlib.colors.CSS4_COLORS['crimson'])
    teal_as_RGBA = [[teal_rgb_value[0] + ((white_rgb_value[0] - teal_rgb_value[0]) * i / 256),
                     teal_rgb_value[1] + ((white_rgb_value[1] -
                                          teal_rgb_value[1]) * i / 256),
                     teal_rgb_value[2] + ((white_rgb_value[2] -
                                          teal_rgb_value[2]) * i / 256),
                     1.0]
                    for i in range(256)
                    ]
    salmon_as_RGBA = [[salmon_rgb_value[0] + ((white_rgb_value[0] - salmon_rgb_value[0]) * i / 256),
                       salmon_rgb_value[1] +
                       ((white_rgb_value[1] - salmon_rgb_value[1]) * i / 256),
                       salmon_rgb_value[2] +
                       ((white_rgb_value[2] -
                         salmon_rgb_value[2]) * i / 256),
                       1.0]
                      for i in range(256)
                      ]
    olive_as_RGBA = [[olive_rgb_value[0] + ((white_rgb_value[0] - olive_rgb_value[0]) * i / 256),
                      olive_rgb_value[1] + ((white_rgb_value[1] -
                                            olive_rgb_value[1]) * i / 256),
                      olive_rgb_value[2] + ((white_rgb_value[2] -
                                            olive_rgb_value[2]) * i / 256),
                      1.0]
                     for i in range(256)
                     ]
    violet_as_RGBA = [[violet_rgb_value[0] + ((white_rgb_value[0] - violet_rgb_value[0]) * i / 256),
                       violet_rgb_value[1] +
                       ((white_rgb_value[1] - violet_rgb_value[1]) * i / 256),
                       violet_rgb_value[2] +
                       ((white_rgb_value[2] -
                         violet_rgb_value[2]) * i / 256),
                       1.0]
                      for i in range(256)
                      ]
    ls_to_crimson_as_RGBA = [[lightsalmon_rgb_value[0] + ((crimson_rgb_value[0] - lightsalmon_rgb_value[0]) * i / 256),
                              lightsalmon_rgb_value[1] + (
                                  (crimson_rgb_value[1] - lightsalmon_rgb_value[1]) * i / 256),
                              lightsalmon_rgb_value[2] + (
                                  (crimson_rgb_value[2] - lightsalmon_rgb_value[2]) * i / 256),
                              1.0]
                             for i in range(256)
                             ]
    teal_to_crimson_as_RGBA = [[teal_rgb_value[0] + ((crimson_rgb_value[0] - teal_rgb_value[0]) * i / 256),
                                teal_rgb_value[1] + ((crimson_rgb_value[1] -
                                                     teal_rgb_value[1]) * i / 256),
                                teal_rgb_value[2] + ((crimson_rgb_value[2] -
                                                     teal_rgb_value[2]) * i / 256),
                                1.0]
                               for i in range(256)
                               ]
    teal_salmons = list(teal_as_RGBA) + list(reversed(salmon_as_RGBA))
    olive_violets = list(olive_as_RGBA) + list(reversed(violet_as_RGBA))
    teal_salmon_black = list(teal_as_RGBA) + \
        [(0, 0, 0, 0)] + list(reversed(salmon_as_RGBA))
    teal_salmon_cmap = matplotlib.colors.ListedColormap(teal_salmons, N=512)
    olive_violet_cmap = matplotlib.colors.ListedColormap(olive_violets, N=512)
    teals = matplotlib.colors.ListedColormap(
        list(reversed(teal_as_RGBA)), N=256)
    salmons = matplotlib.colors.ListedColormap(
        list(reversed(salmon_as_RGBA)), N=256)
    olives = matplotlib.colors.ListedColormap(
        list(reversed(olive_as_RGBA)), N=256)
    violets = matplotlib.colors.ListedColormap(
        list(reversed(violet_as_RGBA)), N=256)
    teal_salmon_black = matplotlib.colors.ListedColormap(
        teal_salmon_black, N=513)
    ls_crimson = matplotlib.colors.ListedColormap(ls_to_crimson_as_RGBA, N=256)
    teal_crimson = matplotlib.colors.ListedColormap(
        teal_to_crimson_as_RGBA, N=256)
    custom_cmap_dict = {
        'teal_salmon': teal_salmon_cmap,
        'teals': teals,
        'salmons': salmons,
        'olive_violet': olive_violet_cmap,
        'olives': olives,
        'violets': violets,
        'teal_salmon_black': teal_salmon_black,
        'orange_red': ls_crimson,
        'teal_crimson': teal_crimson,
    }
    return custom_cmap_dict


CMAPS = make_custom_cmap_dict()

# 히트맵 그리는 함수들
def draw_heatmap(adjacency_matrix_: pd.DataFrame, 
                 figure_obj: plt.Figure, 
                 figure_args_dict_ : dict[str, Any]) -> None:
    heatmap_ax_obj: plt.Axes = figure_obj.subplots(1, 1)
    heatmap_line_width = 0.1
    adjacency_matrix = adjacency_matrix_
    adj_values = adjacency_matrix.to_numpy().flatten()
    adj_values_notnan = adj_values[~np.isnan(adj_values)]
    max_adj_value = adj_values_notnan.max()
    min_adj_value = adj_values_notnan.min()
    try:
        if figure_args_dict_['binarization_threshold']:
            threshold = figure_args_dict_['binarization_threshold']
            adjacency_matrix = adjacency_matrix_[adjacency_matrix > threshold]
            adj_values = adjacency_matrix.to_numpy().flatten()
            adj_values_notnan = adj_values[~np.isnan(adj_values)]
            max_adj_value = adj_values_notnan.max()
            min_adj_value = adj_values_notnan.min()
    except KeyError:
        pass
    try:
        if figure_args_dict_['normalize_by_maximum']:
            adjacency_matrix = adjacency_matrix / max_adj_value
            adj_values = adjacency_matrix.to_numpy().flatten()
            adj_values_notnan = adj_values[~np.isnan(adj_values)]
            max_adj_value = adj_values_notnan.max()
            min_adj_value = adj_values_notnan.min()
    except KeyError:
        pass
    annotate_ = figure_args_dict_['annotate']
    sns.heatmap(np.where(adjacency_matrix.isna(), 0, np.nan), 
            vmin=0.0, vmax=0.0,
            cmap=matplotlib.colors.ListedColormap(['lightgrey']), 
            # linewidths=0.01, linecolor='lightgrey', 
            annot=False, ax=heatmap_ax_obj, cbar=False, 
            # fmt=".3g",
            # annot_kws=dict(fontsize=22, fontweight='bold', )
            )
    sns.heatmap(np.where((adjacency_matrix.gt(-1e-8) & adjacency_matrix.lt(1e-8)), 0, np.nan), 
            vmin=1.0, vmax=1.0,
            cmap=matplotlib.colors.ListedColormap(['dimgrey']), 
            # linewidths=0.01, linecolor='lightgrey', 
            annot=False, ax=heatmap_ax_obj, cbar=False, 
            )
    for_heatmap_ = adjacency_matrix[~(adjacency_matrix.gt(-1e-8) & adjacency_matrix.lt(1e-8))]
    if annotate_:
        sns.heatmap(for_heatmap_, 
                vmin=min_adj_value, vmax=max_adj_value,
                cmap=CMAPS['orange_red'], 
                linewidths=heatmap_line_width, linecolor='lightgrey', 
                annot=True, ax=heatmap_ax_obj, cbar=False, 
                fmt=".3f",
                annot_kws=dict(fontsize=22, fontweight='bold', )
                )
    else:
        sns.heatmap(for_heatmap_, 
                vmin=min_adj_value, vmax=max_adj_value,
                cmap=CMAPS['orange_red'], 
                linewidths=heatmap_line_width, linecolor='lightgrey', 
                annot=False, ax=heatmap_ax_obj, cbar=False, 
                # fmt=".3g",
                # annot_kws=dict(fontsize=22, fontweight='bold', )
                )
    heatmap_ax_obj.tick_params(axis='x', bottom=False, labelbottom=False, top=True, labeltop=True, labelrotation=90.)
    heatmap_ax_obj.tick_params(axis='x', labelsize=figure_args_dict_['label_font_size'])
    heatmap_ax_obj.tick_params(axis='y', right=False, labelright=False, left=True, labelleft=True, labelrotation=0.)
    heatmap_ax_obj.tick_params(axis='y', labelsize=figure_args_dict_['label_font_size'])
    heatmap_ax_obj.spines.top.set_visible(True)
    heatmap_ax_obj.spines.bottom.set_visible(True)
    heatmap_ax_obj.spines.left.set_visible(True)
    heatmap_ax_obj.spines.right.set_visible(True)
    if figure_args_dict_['draw_hemisphere_divisor']:
        left_and_right_divisor = adjacency_matrix.shape[0] // 2
        heatmap_ax_obj.axhline(left_and_right_divisor, 
                            color=figure_args_dict_['divisor_color'], 
                            linewidth=1.5)
        heatmap_ax_obj.axvline(left_and_right_divisor, 
                            color=figure_args_dict_['divisor_color'], 
                            linewidth=1.5)
    return None


def draw_vertical_cbars(fig_obj: plt.Figure, 
                        min_adj_value: float,
                        max_adj_value: float,
                        tick_numbers = 4) -> None:
    cbar_subfig_arrays = fig_obj.subplots(3, 1, 
                                          height_ratios = [10, 1, 1], 
                                          gridspec_kw=dict(wspace=0.1, hspace=0.1))
    this_colormap = CMAPS['orange_red']
    fig_obj.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=this_colormap), 
                        cax=cbar_subfig_arrays[0])
    black_patch = mpatches.FancyBboxPatch((0, 0), 1, 1, 
                                          transform=cbar_subfig_arrays[1].transAxes, 
                                          boxstyle='Square', 
                                          color=matplotlib.colors.to_hex('dimgrey'))
    cbar_subfig_arrays[1].add_patch(black_patch)
    grey_patch = mpatches.FancyBboxPatch((0, 0), 1, 1, 
                                         transform=cbar_subfig_arrays[2].transAxes,
                                         boxstyle='Square', 
                                         color=matplotlib.colors.to_hex('lightgrey'))
    cbar_subfig_arrays[2].add_patch(grey_patch)
    tick_array = np.linspace(min_adj_value, max_adj_value, tick_numbers)
    tick_label_array = [f'{tick_:1.3f}' for tick_ in tick_array]
    tick_pos_array = np.linspace(0.0, 1.0, tick_numbers)
    cbar_subfig_arrays[0].set_yticks(tick_pos_array, labels = tick_label_array)
    # cbar_subfig_arrays[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], labels = ['0.0', '0.25', '0.5', '0.75', '1.0'])
    cbar_subfig_arrays[1].set_yticks([0.5], labels = ['NS'])
    cbar_subfig_arrays[2].set_yticks([0.0], labels = ['NaN'])
    cbar_subfig_arrays[0].tick_params(axis='y', labelsize=plot_config.SUBTITLE_SIZE)
    cbar_subfig_arrays[1].tick_params(axis='x', bottom=False, labelbottom=False, )
    cbar_subfig_arrays[1].tick_params(axis='y', 
                                      left=False, labelleft=False, 
                                      right=True, labelright=True, 
                                      labelsize=plot_config.SUBTITLE_SIZE)
    cbar_subfig_arrays[2].tick_params(axis='x', bottom=False, labelbottom=False, )
    cbar_subfig_arrays[2].tick_params(axis='y', 
                                      left=False, labelleft=False, 
                                      right=True, labelright=True, 
                                      labelsize=plot_config.SUBTITLE_SIZE)
    return None


# 네트워크 그리는 함수들
def get_edge_label_position(x1, y1, x2, y2, rad=0.0):
    x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
    dx, dy = x2 - x1, y2 - y1
    cx, cy = x12 + rad * dy, y12 - rad * dx
    return cx, cy


def get_edge_label_rotation(x1, y1, x2, y2):
    eps_ = 1e-4
    if np.abs(x2 - x1) < eps_: # at the same x position
        if y2 > y1: # up to down
            return 90.0 #* FINAL
        else: # down to up
            return -90.0 #* FINAL
    else:
        distance = np.sqrt(
            ((x2 - x1) ** 2) + ((y2 - y1) ** 2)
        )
        sero_ = (x2 - x1)
        label_rotation_rad = np.arcsin(sero_ / distance) # [-pi/2, pi/2]
        # apply sabunmyeon-dependent degree
        if (x2 > x1) and (y2 > y1) : # 1sabunmyeon
            label_rotation = - (label_rotation_rad / np.pi * 180.0) + 90.0 #* FINAL
        elif (x2 < x1) and (y2 > y1) : # 2sabunmyeon
            label_rotation = - (label_rotation_rad / np.pi * 180.0) - 90.0 #* FINAL
        elif (x2 < x1) and (y2 < y1) : # 3sabunmyeon
            label_rotation = (label_rotation_rad / np.pi * 180.0) + 90.0 #* FINAL
        elif (x2 > x1) and (y2 < y1) : # 4sabunmyeon
            label_rotation = (label_rotation_rad / np.pi * 180.0) - 90.0 #*FINAL
        else:
            label_rotation = (label_rotation_rad / np.pi * 180.0)
            # raise ValueError(x1,y1, x2, y2)
        return label_rotation


def draw_non_self_loop_edge(graph: nx.Graph, 
                            node_positions: dict[str, np.array], 
                            node_pair: tuple[str, str], 
                            edge_weight_label: float | str,
                            edge_width_offset: float, 
                            ax_obj: plt.Axes,
                            drawing_args_dict: dict[str, Any],
                            ) ->  plt.Axes:
    RADIAN = drawing_args_dict['edge_curve_radian']
    conn_style_pos = 'arc3,rad=' + str(RADIAN)
    DEFAULT_WIDTH = drawing_args_dict['deafult_edge_width']
    position = node_positions
    ALPHA = drawing_args_dict['edge_alpha']
    graph.add_edge(node_pair[0], node_pair[1],)
    edge_weight_string = f'{edge_weight_label:1.3f}' if isinstance(edge_weight_label, float) else edge_weight_label
    if isinstance(edge_weight_label, str):
        if len(edge_weight_label.split('=')) == 2: #  '\u0394=' +  f'{delta_conn:1.3f}' case
            edge_weight = float(edge_weight_label.split('=')[-1])
        else:
            edge_weight_another = edge_weight_label.split('p')[0].strip()
            edge_weight = float(edge_weight_another.split('=')[-1])
    else:
        edge_weight = edge_weight_label
    edge_width = DEFAULT_WIDTH + np.abs(edge_width_offset) # #FF0000
    this_drawing_colormap = drawing_args_dict['network_cmap']
    edge_color = this_drawing_colormap( drawing_args_dict['colormap_rescaling'](edge_weight))
    nx.draw_networkx_edges(graph, position, edgelist=[node_pair, ], 
                            width=edge_width,
                            edge_color=edge_color, 
                            arrowstyle='-|>',
                            arrowsize=25,
                            arrows=True,
                            node_size=drawing_args_dict['node_size'],
                            min_target_margin=1,
                            connectionstyle=conn_style_pos, 
                            ax=ax_obj,)
    elabel_x, elabel_y = get_edge_label_position(position[node_pair[0]][0], 
                                            position[node_pair[0]][1], 
                                            position[node_pair[1]][0], 
                                            position[node_pair[1]][1], 
                                            rad=RADIAN/2.0)
    label_rot = get_edge_label_rotation(position[node_pair[0]][0], 
                                            position[node_pair[0]][1], 
                                            position[node_pair[1]][0], 
                                            position[node_pair[1]][1], )
    if drawing_args_dict['draw_edge_label']:
        ax_obj.text(elabel_x, elabel_y, edge_weight_string, 
                            fontsize=drawing_args_dict['edge_caption_size'],  
                            verticalalignment='center', 
                            horizontalalignment='center',
                            multialignment='center', 
                            rotation=label_rot,
                            bbox=dict(edgecolor='black', boxstyle='round', facecolor='white', ),
                            )
    graph.remove_edge(node_pair[0], node_pair[1],)
    return ax_obj
