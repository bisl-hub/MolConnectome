''' 특정한 설정으로 구성 된 네트워크를 만들고, 네트워크를 파일로 그리고, csv 파일로 저장하는 기능을 하게 될 것임.'''

import os
import itertools
import datetime
import sys
import pickle

import scipy.stats

import pandas as pd
import numpy as np
import warnings


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
import calculate_abundance as abund

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_edge_weight(pval_arr_obj: np.ndarray[float], alpha: float, method='ppf'):
    pval_arr = pval_arr_obj
    pval_median = np.median(pval_arr)
    ALPHA = alpha
    if method == 'ppf':
        edge_weight = np.mean([scipy.stats.norm.ppf(x) for x in pval_arr])
    elif method == 'quantile':
        edge_weight = np.mean(pval_arr)
    else:
        raise ValueError('Something is wrong.')
    return edge_weight


def calculate_edge_weight_both_sign(positive_pvalue_array: np.ndarray[float], 
                               negative_pvalue_array: np.ndarray[float],
                               alpha=0.05) -> float:
    positive_pvalue_median = np.median(positive_pvalue_array)
    negative_pvalue_median = np.median(negative_pvalue_array)
    if (positive_pvalue_median > 1-alpha) and (negative_pvalue_median > 1-alpha): 
        # more pos and more neg
        edge_weight = (np.mean([scipy.stats.norm.ppf(x) for x in positive_pvalue_array]) + 
                    np.mean([scipy.stats.norm.ppf(x) for x in negative_pvalue_array]))
    elif (positive_pvalue_median > 1-alpha) and (alpha < negative_pvalue_median <= 1-alpha): 
        # more pos but not more or less neg
        edge_weight = np.mean([scipy.stats.norm.ppf(x) for x in positive_pvalue_array])
    elif (positive_pvalue_median > 1-alpha) and (negative_pvalue_median <= alpha): 
        # more pos but less neg
        edge_weight = (np.mean([scipy.stats.norm.ppf(x) for x in positive_pvalue_array]) -
                    np.mean([scipy.stats.norm.ppf(x) for x in negative_pvalue_array]))
    elif (alpha < positive_pvalue_median <= 1-alpha) and (negative_pvalue_median > 1-alpha): 
        # not more or less pos but more neg
        edge_weight = np.negative(np.mean([scipy.stats.norm.ppf(x) for x in negative_pvalue_array]))
    elif (alpha < positive_pvalue_median <= 1-alpha) and (alpha < negative_pvalue_median <= 1-alpha): # NEUTRAL
        edge_weight = (
            (np.mean([scipy.stats.norm.ppf(x) for x in positive_pvalue_array]) + 
            np.mean([scipy.stats.norm.ppf(x) for x in negative_pvalue_array]))
            / 2.0)
    elif (alpha < positive_pvalue_median <= 1-alpha) and (negative_pvalue_median <= alpha): # not more or less pos but less neg
        edge_weight = (
            (np.mean([scipy.stats.norm.ppf(x) for x in positive_pvalue_array]) - 
            np.mean([scipy.stats.norm.ppf(x) for x in negative_pvalue_array]))
            / 2.0)
    elif (positive_pvalue_median <= alpha) and (negative_pvalue_median > 1-alpha): # less pos but more neg
        edge_weight = (np.mean([scipy.stats.norm.ppf(x) for x in positive_pvalue_array]) -
                    np.mean([scipy.stats.norm.ppf(x) for x in negative_pvalue_array]))
    elif (positive_pvalue_median <= alpha) and (alpha < negative_pvalue_median <= 1-alpha): # less pos but not more or less neg
        edge_weight = (
            (np.mean([scipy.stats.norm.ppf(x) for x in positive_pvalue_array]) + 
            np.mean([scipy.stats.norm.ppf(x) for x in negative_pvalue_array]))
            / 2.0)
    elif (positive_pvalue_median <= alpha) and (negative_pvalue_median <= alpha): # less pos and less neg
        edge_weight = (np.mean([scipy.stats.norm.ppf(x) for x in positive_pvalue_array]) - 
                    np.mean([scipy.stats.norm.ppf(x) for x in negative_pvalue_array]))
    else:
        raise ValueError(positive_pvalue_array, negative_pvalue_array)
    return edge_weight


def calculate_edge_weight_both_sign_quantile(positive_pvalue_array: np.ndarray[float], 
                               negative_pvalue_array: np.ndarray[float],
                               alpha=0.05) -> float:
    positive_pvalue_median = np.median(positive_pvalue_array)
    negative_pvalue_median = np.median(negative_pvalue_array)
    if (positive_pvalue_median > 1-alpha) and (negative_pvalue_median > 1-alpha): 
        # more pos and more neg
        edge_weight = (np.mean(positive_pvalue_array) + np.mean(negative_pvalue_array))
    elif (positive_pvalue_median > 1-alpha) and (alpha < negative_pvalue_median <= 1-alpha): 
        # more pos but not more or less neg
        edge_weight = np.mean(positive_pvalue_array)
    elif (positive_pvalue_median > 1-alpha) and (negative_pvalue_median <= alpha): 
        # more pos but less neg
        edge_weight = (np.mean(positive_pvalue_array) - np.mean(negative_pvalue_array))
    elif (alpha < positive_pvalue_median <= 1-alpha) and (negative_pvalue_median > 1-alpha): 
        # not more or less pos but more neg
        edge_weight = np.negative(np.mean([scipy.stats.norm.ppf(x) for x in negative_pvalue_array]))
    elif (alpha < positive_pvalue_median <= 1-alpha) and (alpha < negative_pvalue_median <= 1-alpha): # NEUTRAL
        edge_weight = (
            (np.mean(positive_pvalue_array) + np.mean(negative_pvalue_array)) / 2.0)
    elif (alpha < positive_pvalue_median <= 1-alpha) and (negative_pvalue_median <= alpha): # not more or less pos but less neg
        edge_weight = (
            (np.mean(positive_pvalue_array) - np.mean(negative_pvalue_array)) / 2.0)
    elif (positive_pvalue_median <= alpha) and (negative_pvalue_median > 1-alpha): # less pos but more neg
        edge_weight = (np.mean(positive_pvalue_array) - np.mean(negative_pvalue_array))
    elif (positive_pvalue_median <= alpha) and (alpha < negative_pvalue_median <= 1-alpha): # less pos but not more or less neg
        edge_weight = (
            (np.mean(positive_pvalue_array) + np.mean(negative_pvalue_array)) / 2.0)
    elif (positive_pvalue_median <= alpha) and (negative_pvalue_median <= alpha): # less pos and less neg
        edge_weight = (np.mean(positive_pvalue_array) - np.mean(negative_pvalue_array))
    else:
        raise ValueError(positive_pvalue_array, negative_pvalue_array)
    return edge_weight


def make_transcriptome_adjacency(pvalue_dataframe: pd.DataFrame, 
                                 args_dict_ : dict[str, str], 
                                 with_self_loop = False) -> pd.DataFrame:
    gene_set_id = args_dict_['gene_set']; 
    this_gene_set_result = pvalue_dataframe.groupby('gene_set').get_group(gene_set_id)
    REGIONS_reordered = args_dict_['region_list'];
    base_dataframe = pd.DataFrame(np.nan, index=REGIONS_reordered, columns=REGIONS_reordered)
    iterator_ = itertools.permutations(REGIONS_reordered, 2)
    # ! 이터레이터 손 안볼거임. 이유? 나중에 알아서 걸러지게 됨. 일단 이 단계에서는 거르지 않는 것을 원칙으로 함.
    for perm_ in iterator_:
        try: # 전체를 넣어야 하는 이유는 위의 사람 데이터에서 없는 게 나올 것이기 때문임. 예를 들자면 AC_PFC -> BS_PFC 같은거.
            if (((args_dict_['sign_handler'] != 'naive') and (args_dict_['condition'] == 'difference')) or
                (args_dict_['sign_handler'] == 'naive')): #  negative corr 있는 경우
                pvalue_array_pos = this_gene_set_result.groupby('source').get_group(perm_[0]).groupby(
                    'target').get_group(perm_[1]).groupby('sign').get_group('pos').groupby(
                        'attribute').get_group(args_dict_['attribute_to_plot'])['pvalue'].to_numpy() # array of float
                pvalue_array_neg = this_gene_set_result.groupby('source').get_group(perm_[0]).groupby(
                    'target').get_group(perm_[1]).groupby('sign').get_group('neg').groupby(
                        'attribute').get_group(args_dict_['attribute_to_plot'])['pvalue'].to_numpy() # array of float
                if args_dict_['connectivity_measure'] == 'ppf':
                    calculated_connectivity = calculate_edge_weight_both_sign(
                        pvalue_array_pos, pvalue_array_neg,
                        alpha=float(args_dict_['significance_threshold']), )
                elif args_dict_['connectivity_measure'] == 'quantile':
                    calculated_connectivity = calculate_edge_weight_both_sign_quantile(
                        pvalue_array_pos, pvalue_array_neg,
                        alpha=float(args_dict_['significance_threshold']), )
                base_dataframe.loc[perm_[0], perm_[1]] = calculated_connectivity
            elif (args_dict_['sign_handler'] == 'negative_only'): #  negative corr 있는 경우
                pvalue_array_ = this_gene_set_result.groupby('source').get_group(perm_[0]).groupby(
                    'target').get_group(perm_[1]).groupby('sign').get_group('neg').groupby(
                        'attribute').get_group(args_dict_['attribute_to_plot'])['pvalue'].to_numpy() # array of float
                calculated_connectivity = calculate_edge_weight(pvalue_array_, 
                                                                float(args_dict_['significance_threshold']), 
                                                                method = args_dict_['connectivity_measure'])
                base_dataframe.loc[perm_[0], perm_[1]] = calculated_connectivity
            else:
                # print(gene_set_id, perm_)
                pvalue_array_ = this_gene_set_result.groupby('source').get_group(perm_[0]).groupby(
                    'target').get_group(perm_[1]).groupby('sign').get_group('pos').groupby(
                        'attribute').get_group(args_dict_['attribute_to_plot'])['pvalue'].to_numpy() # array of float
                # print(pvalue_array_)
                calculated_connectivity = calculate_edge_weight(pvalue_array_, 
                                                                float(args_dict_['significance_threshold']), 
                                                                method = args_dict_['connectivity_measure'])
                base_dataframe.loc[perm_[0], perm_[1]] = calculated_connectivity
        except KeyError: # AC_PFC -> BS_PFC 같은 경우의 핸들링.
            base_dataframe.loc[perm_[0], perm_[1]] = np.nan
    # for edges between same nodes. (self-loop)
    if with_self_loop: # ! redundant
        for region_ in REGIONS_reordered:
            pvalue_array_ = this_gene_set_result.groupby('source').get_group(region_).groupby(
                'target').get_group(region_).groupby('sign').get_group('pos').groupby(
                    'attribute').get_group(args_dict_['attribute_to_plot'])['pvalue'].to_numpy() # array of float
            calculated_connectivity = calculate_edge_weight(pvalue_array_, 
                                                         float(args_dict_['significance_threshold']), )
            base_dataframe.loc[region_, region_] = calculated_connectivity
    return base_dataframe


def return_predicted_adjacency_dict(
    quantile_files_saved_dirs: list[str],
    gene_set_dictionary: dict[str, list[str]],
    region_list: list[str],
    attribute_to_plot_: str, 
    connectivity_measure_: str, 
    condition_: str, 
    significance_threshold: str,
    minmax_normalization: bool,
    ) -> dict[str, pd.DataFrame]:
    args_dict = {
        'attribute_to_plot': attribute_to_plot_,            # 'counts' | 'average' | 'multiply'
        'connectivity_measure': connectivity_measure_,      # 'ppf' | 'quantile'
        'normalize_by_maximum': minmax_normalization,       # True | False
        'condition': condition_,                            # 'control' | 'schizo' | 'difference'
        'region_list': region_list,
        'sign_handler': 'ignore',                           # 'absolute' | 'naive' | 'ignore' | 'preserve' | 'negative_only'   
        'significance_threshold': float(significance_threshold),
    }
    data_arr = [pd.read_csv(dir_, index_col=0) for dir_ in quantile_files_saved_dirs]
    data_final = pd.concat(data_arr, axis=0)
    # print(data_final.columns)
    sign_handler_ = 'ignore'
    geneset_wise_heatmap_collection = {}
    for i, gene_set_id in enumerate(gene_set_dictionary.keys()): # glutamate and dopamine #FF0000
        args_dict['gene_set'] = gene_set_id
        data_to_calculate = data_final.groupby('condition').get_group(condition_)
        # print(gene_set_id, data_to_calculate)
        this_gene_set_adjacency = make_transcriptome_adjacency(data_to_calculate, args_dict)
        # 원숭이라면 이거로 됨.
        # 근데 사람은 컨디션이 나뉘어 있으니깐 이 경우를 생각을 해 주어야 함.
        # 즉 이거는 하나의 컨디션에 대해서만 내뱉는 함수라고 생각하면 될 것 같음.
        if minmax_normalization:
            if connectivity_measure_ == 'ppf':
                if sign_handler_ != 'naive':
                    normalized_adjacency = (
                        (this_gene_set_adjacency + np.abs(scipy.stats.norm.ppf(1/1002))) / 
                                            (np.abs(scipy.stats.norm.ppf(1/1002)) * 2)
                                            )
                else:
                    normalized_adjacency = (
                        (this_gene_set_adjacency + (np.abs(scipy.stats.norm.ppf(1/1002)) * 2) ) / 
                                            (np.abs(scipy.stats.norm.ppf(1/1002)) * 4)
                                            )
            elif connectivity_measure_ == 'quantile':
                normalized_adjacency = this_gene_set_adjacency
        else:
            normalized_adjacency = this_gene_set_adjacency
        geneset_wise_heatmap_collection[gene_set_id] = normalized_adjacency
    # ! 이제 큐레이트, 뉴런챗, 머지드에 대해 평균과 합 을 정의해 줘야 함.
    return geneset_wise_heatmap_collection


def main(task_name: str, result_save_dir: str, 
         gene_set_dir: str, expression_files: str, quantile_files: str,
         significance_threshold: str,
         ):
    '''기본 뼈대만 가져옴. 나머지는 전부 작업 다시 해야 할 것으로 보임.'''
    with open(file=quantile_files, mode='r', encoding='utf-8') as f:
        quantile_files_list = f.read().splitlines()
    expression_files = pd.read_csv(expression_files, sep=',')

    # 잘 불러와진거 확인함.
    # print(quantile_files_list)
    # print(expression_files)

    mean_abundance_control, median_abundance_control, trimean_abundance_control = \
        abund.calculate_neurotransmitter_abundance(expression_files, 'control')
    mean_abundance_schizo, median_abundance_schizo, trimean_abundance_schizo = \
        abund.calculate_neurotransmitter_abundance(expression_files, 'schizo')

    # 계산 잘 된거 확인함.
    # print(mean_abundance_control)
    # print(mean_abundance_schizo)

    region_list = np.unique(expression_files['region_name'].to_numpy())
    gene_set_dictionary = pickle.load(open(file=gene_set_dir, mode='rb'))
    network_args_control = [quantile_files_list, gene_set_dictionary, region_list,
                            'multiply', 'ppf', 'control', significance_threshold, True]
    network_args_schizo = [quantile_files_list, gene_set_dictionary, region_list,
                            'multiply', 'ppf', 'schizo', significance_threshold, True]
    predicted_adj_dict_control = return_predicted_adjacency_dict(*network_args_control)
    predicted_adj_dict_schizo = return_predicted_adjacency_dict(*network_args_schizo)


    # 계산 잘 된거 확인함.
    # print(predicted_adj_dict_control)
    # print(predicted_adj_dict_schizo)
    # for gene_set_id in ['curated_DA', 'curated_Glu', 'curated_GABA']:
    #     print(gene_set_id)
    #     print(predicted_adj_dict_control[gene_set_id])
    #     print(predicted_adj_dict_schizo[gene_set_id])

    # 곱하기 실행
    mean_abundance_control_essential: 'pd.DataFrame' = \
        mean_abundance_control.loc[['DA', '5-HT', 'Glu', 'GABA', 'NE', 'ACh', 'epi', 'Gly', 'NP', 'Syn',]]
    mean_abundance_schizo_essential: 'pd.DataFrame'  = \
        mean_abundance_schizo.loc[['DA', '5-HT', 'Glu', 'GABA', 'NE', 'ACh', 'epi', 'Gly', 'NP', 'Syn',]]
    
    control_multiplied = abund.multiply_weight_with_adjacency(predicted_adj_dict_control, 
                                   gene_set_dictionary.keys(),
                                   mean_abundance_control_essential)

    schizo_multiplied = abund.multiply_weight_with_adjacency(predicted_adj_dict_schizo, 
                                   gene_set_dictionary.keys(),
                                   mean_abundance_schizo_essential)

    # 결과 저장
    to_save_dir = os.path.join(result_save_dir, task_name)
    with open(file=os.path.join(to_save_dir, 'CONTROL_unweighted.pickle'), mode='wb') as f:
        pickle.dump(predicted_adj_dict_control, f)
    with open(file=os.path.join(to_save_dir, 'SCHIZO_unweighted.pickle'), mode='wb') as f:
        pickle.dump(predicted_adj_dict_schizo, f)
    mean_abundance_control_essential.to_csv(os.path.join(to_save_dir, 'CONTROL_ABUNDANCE_MEAN.csv'), sep=',')
    mean_abundance_schizo_essential.to_csv(os.path.join(to_save_dir, 'SCHIZO_ABUNDANCE_MEAN.csv'), sep=',')
    with open(file=os.path.join(to_save_dir, 'CONTROL_weighted.pickle'), mode='wb') as f:
        pickle.dump(control_multiplied, f)
    with open(file=os.path.join(to_save_dir, 'SCHIZO_weighted.pickle'), mode='wb') as f:
        pickle.dump(schizo_multiplied, f)



if __name__ == '__main__':
    import argparse
    print("Initializing...")
    parser = argparse.ArgumentParser(description="Calculate correlation between regions")
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--result-save-dir", type=str, required=True)
    parser.add_argument("--gene-set-dir", type=str, required=True)
    parser.add_argument("--expression-files", type=str, required=True)
    parser.add_argument("--quantile-files", type=str, required=True)
    parser.add_argument("--significance-threshold", type=str, required=True)

    parsed_args = parser.parse_args()

    task_name = parsed_args.task_name
    resut_save_dir = parsed_args.result_save_dir
    gene_set_dir = parsed_args.gene_set_dir
    expression_files = parsed_args.expression_files
    quantile_files = parsed_args.quantile_files
    significance_threshold = parsed_args.significance_threshold

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print('|'.join([task_name, quantile_files]), 'start...')

    if not os.path.exists(os.path.join(resut_save_dir, task_name)):
        os.makedirs(os.path.join(resut_save_dir, task_name))
    
    if not os.path.isfile(
        os.path.join(resut_save_dir, task_name, 'CONTROL_ABUNDANCE_MEAN.csv')
    ): # if there are no result file - especially abundance file
        main(task_name, 
             resut_save_dir, gene_set_dir, 
             expression_files, quantile_files,
             significance_threshold)

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print('|'.join([task_name, quantile_files]), 'done!') 
