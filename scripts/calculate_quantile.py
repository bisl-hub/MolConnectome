import os
import datetime
import warnings
import pickle
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np

from typing import Any

CORR_ATTRS = ['control_correlation', 'control_correlation_pvalue_perm_test', 'control_correlation_pvalue_scipy',
              'schizo_correlation', 'schizo_correlation_pvalue_perm_test', 'schizo_correlation_pvalue_scipy',
              'correlation_difference', 'correlation_difference_pvalue_perm_test', 'correlation_difference_pvalue_scipy',] 
warnings.filterwarnings('ignore')


def load_corr_result(corr_result_dir:str) -> 'dict[str, pd.DataFrame]':  
    corr_result_dict = {}
    for attr_ in CORR_ATTRS:
        this_result_path = os.path.join(corr_result_dir, attr_ + '.csv') 
        try:
            data_df = pd.read_csv(this_result_path, sep=',',)
        except FileNotFoundError:
            raise ValueError(corr_result_dir, 'seems to not exist.')
        if 'Unnamed: 0' in data_df.columns:
            data_df.set_index('Unnamed: 0', inplace=True)
            data_df.index.name = None
        data_df.astype(float)
        corr_result_dict[attr_] = data_df
    return corr_result_dict


def check_significance_permutation(pvalue:float, sign='none', threshold=0.05):
    eps = 1e-8
    if sign == 'naive': # ! 코릴레이션 값이 [-1, 1] 사이에 위치하니까 
        if pvalue < threshold + eps:
            sgn = 'pos'
        elif pvalue > 1 - threshold + eps:
            sgn = 'neg'
        else:
            sgn = 'NS'
    elif sign == 'absolute': # ! 시그니피컨트한 코릴레이션들 값에 절댓값을 씌울 생각이기 때문에
        if pvalue < threshold + eps:
            sgn = 'pos'
        elif pvalue > 1 - threshold + eps:
            sgn = 'pos'
        else:
            sgn = 'NS'
    elif sign == 'ignore' or sign == 'preserve':
        if pvalue < threshold + eps:
            sgn = 'pos'
        else:
            sgn = 'NS'
    return sgn


def sign_wrapper(corrcoef: float):
    eps = 1e-8
    if corrcoef < np.negative(eps):
        sign = -1.0
    elif corrcoef > eps:
        sign = 1.0
    elif np.negative(eps) <= corrcoef <= eps:
        sign = 0.0
    else:
        sign = 0.0
    return sign
    

def check_significance_scipy(pvalue:float, sign='none', threshold=0.05):
    eps = 1e-8
    if sign == 'naive': # ! 코릴레이션 값이 [-1, 1] 사이에 위치하니까 
        if eps <= pvalue <= threshold + eps: # 시그니피컨트 한 경우우    
            sgn = 'pos'
        elif np.negative(threshold + eps) <= pvalue <= eps:
            sgn = 'neg'
        else: # 피밸류가 0.05보다 큰 경우들
            sgn = 'NS'
    elif sign == 'absolute' or sign == 'ignore' or sign == 'preserve':
        if eps <= pvalue <= threshold + eps:
            sgn = 'pos'
        else:
            sgn = 'NS'
    return sgn


def aggregate_significant_correlation_each(   
    corr_values : 'pd.DataFrame[float]',
    corr_pvalue : 'pd.DataFrame[float]',
    selected_genes_cat1: list[str] | np.ndarray[str],
    selected_genes_cat2: list[str] | np.ndarray[str],
    args_dict: dict[str, str],) -> dict[str, float]:
    
    alpha = float(args_dict['significance_threshold'])
    sign_handler = args_dict['sign_handler']
    pvalue_method = args_dict['pvalue_method']
    
    # ! cat2 가 생산, 수송 관련이고 cat1 이 리셉터들임!!!!
    # ! 여기서부터 완전 손대기. 각 값들이 밸류로 리턴되었기 때문에 이를 처리해주어야 함.
    YtoX_corr_values: pd.DataFrame = corr_values.loc[selected_genes_cat2][selected_genes_cat1].copy()
    YtoX_corr_pvalue: pd.DataFrame = corr_pvalue.loc[selected_genes_cat2][selected_genes_cat1].copy()
    if pvalue_method == 'permutation':
        YtoX_corr_significance = YtoX_corr_pvalue.map(check_significance_permutation, 
                                                      sign=sign_handler, threshold=alpha)
    elif pvalue_method == 'scipy': # beta-dist or wald test t statistic
        YtoX_corr_sign = YtoX_corr_values.map(sign_wrapper)
        YtoX_corr_pvalue_temp = YtoX_corr_pvalue.multiply(YtoX_corr_sign) # 피밸류에 마이너스와 플러스를 씌움
        YtoX_corr_significance = YtoX_corr_pvalue_temp.map(check_significance_scipy, 
                                                           sign=sign_handler, threshold=alpha)
    YtoX_pos_counts = YtoX_corr_significance[(YtoX_corr_significance == 'pos')].notna().sum(axis=1).sum(axis=0) # counts
    YtoX_pos_values_avg = np.nanmean(YtoX_corr_values[(YtoX_corr_significance == 'pos')].to_numpy().flatten())
    
    # ! 다음으로 두번째 리전에서 첫번째 리전으로 가는 값 계산산  
    XtoY_corr_values: pd.DataFrame = corr_values.loc[selected_genes_cat1][selected_genes_cat2].copy()
    XtoY_corr_pvalue: pd.DataFrame = corr_pvalue.loc[selected_genes_cat1][selected_genes_cat2].copy()
    if pvalue_method == 'permutation':
        XtoY_corr_significance = XtoY_corr_pvalue.map(check_significance_permutation, 
                                                      sign=sign_handler, threshold=alpha)
    elif pvalue_method == 'scipy': # beta-dist or wald test t statistic
        XtoY_corr_sign = XtoY_corr_values.map(sign_wrapper)
        XtoY_corr_pvalue_temp = XtoY_corr_pvalue.multiply(XtoY_corr_sign) # 피밸류에 마이너스와 플러스를 씌움
        XtoY_corr_significance = XtoY_corr_pvalue_temp.map(check_significance_scipy, 
                                                           sign=sign_handler, threshold=alpha)
    XtoY_pos_counts = XtoY_corr_significance[(XtoY_corr_significance == 'pos')].notna().sum(axis=1).sum(axis=0) # counts
    XtoY_pos_values_avg = np.nanmean(XtoY_corr_values[(XtoY_corr_significance == 'pos')].to_numpy().flatten())
    
    # ! 이후 추가 케이스 핸들링
    if (((args_dict['sign_handler'] != 'naive') and (args_dict['condition'] == 'difference')) or
        (args_dict['sign_handler'] == 'naive')): #  negative corr 있는 경우
        YtoX_neg_counts = YtoX_corr_significance[(YtoX_corr_significance == 'neg')].notna().sum(axis=1).sum(axis=0) # counts
        YtoX_neg_values_avg = np.nanmean(YtoX_corr_values[(YtoX_corr_significance == 'neg')].to_numpy().flatten())
        XtoY_neg_counts = XtoY_corr_significance[(XtoY_corr_significance == 'neg')].notna().sum(axis=1).sum(axis=0) # counts
        XtoY_neg_values_avg = np.nanmean(XtoY_corr_values[(XtoY_corr_significance == 'neg')].to_numpy().flatten())
    else: # negative corr 없는 경우
        YtoX_neg_counts = 0
        YtoX_neg_values_avg = 0 
        XtoY_neg_counts = 0 
        XtoY_neg_values_avg = 0
    return_dict = {
        'YtoX_pos_average': YtoX_pos_values_avg, 'YtoX_pos_count': YtoX_pos_counts,
        'YtoX_pos_multiply': (YtoX_pos_counts * YtoX_pos_values_avg),
        'YtoX_neg_average': YtoX_neg_values_avg, 'YtoX_neg_count': YtoX_neg_counts,
        'YtoX_neg_multiply': (YtoX_neg_counts * YtoX_neg_values_avg),
        'XtoY_pos_average': XtoY_pos_values_avg, 'XtoY_pos_count': XtoY_pos_counts, 
        'XtoY_pos_multiply': (XtoY_pos_counts * XtoY_pos_values_avg),
        'XtoY_neg_average': XtoY_neg_values_avg, 'XtoY_neg_count': XtoY_neg_counts,
        'XtoY_neg_multiply': (XtoY_neg_counts * XtoY_neg_values_avg),
    }
    return return_dict


def report_significance(
    original_corr_values : 'pd.DataFrame[float]',
    original_corr_pvalue : 'pd.DataFrame[float]',
    null_corr_values : 'pd.DataFrame[float]',
    null_corr_pvalue : 'pd.DataFrame[float]',
    gene_set_ID: str,
    gene_set_data: dict[Any],
    args_dict: dict[str, str],
) -> None:
    n_permutations = int(args_dict['n_permutations'])
    y_axis_name = args_dict['region_pair'].split('#')[0] # first, 세로축, 보내는쪽
    x_axis_name = args_dict['region_pair'].split('#')[1] # second, 가로축, 받는쪽
    cat1_participants = original_corr_values.index.intersection(gene_set_data['cat1'])
    cat2_participants = original_corr_values.index.intersection(gene_set_data['cat2'])
    original_corr_significant_aggregated = aggregate_significant_correlation_each(
        original_corr_values, original_corr_pvalue,
        cat1_participants, cat2_participants, args_dict)
    total_calculated_cells = cat1_participants.shape[0] * cat2_participants.shape[0]
    
    rng = np.random.default_rng(seed=20240624)
    random_seeds = rng.integers(20240624, 20240624*2, size=int(n_permutations))
    perm_aggregated_collection = []
    for e in range(int(n_permutations)):
        this_perm_seed = random_seeds[e]
        this_perm_rng = np.random.default_rng(seed=this_perm_seed)
        this_perm_selector = this_perm_rng.choice(null_corr_values.shape[0], 
                                                        cat1_participants.shape[0] + cat2_participants.shape[0], 
                                                        replace=False)
        this_perm_cat1_genes = null_corr_values.index.to_numpy()[this_perm_selector[:cat1_participants.shape[0]]]
        this_perm_cat2_genes = null_corr_values.index.to_numpy()[this_perm_selector[cat1_participants.shape[0]:]]
        this_perm_corr_significant_aggregated = aggregate_significant_correlation_each(
            null_corr_values, null_corr_pvalue,
            this_perm_cat1_genes, this_perm_cat2_genes, args_dict)
        this_perm_data = pd.DataFrame.from_dict(this_perm_corr_significant_aggregated, orient='index')
        this_perm_data.columns = ['perm_' + str(e)]
        perm_aggregated_collection.append(this_perm_data)
    total_perm_values_data = pd.concat(perm_aggregated_collection, axis=1) # 12, 1000 사이즈의 데이터프레임이 됨.
    total_perm_values_data = total_perm_values_data.T # 1000, 12 사이즈의 데이터프레임이 됨.
    to_report_ = {}
    for value_key_to_compare in original_corr_significant_aggregated:
        Q = ( ( np.count_nonzero(
            total_perm_values_data[value_key_to_compare].to_numpy() <= \
                original_corr_significant_aggregated[value_key_to_compare]) + 1 ) 
             / ( int(n_permutations) + 2 ) )
        to_report_[value_key_to_compare] = Q
    # 퀀타일 값을 얻었음. 이를 리포트하기.
    final_return_dict_ = {}
    if (
        (args_dict['sign_handler'] == 'naive') or 
        ((args_dict['sign_handler'] != 'naive') and (args_dict['condition'] == 'difference'))
        ): #  negative corr 있는 경우
            for e, value_key_to_compare in enumerate(to_report_): # 0 ~ 11, YtoX_pos_average
                final_return_dict_[e] = {
                    'gene_set' :  gene_set_ID,
                    'compare_with': 'background_rep' + str(args_dict['replication_number']),
                    'condition': args_dict['condition'],
                    'attribute' : value_key_to_compare.split('_')[2],
                    'sign': value_key_to_compare.split('_')[1],
                    'pvalue': str(to_report_[value_key_to_compare]),
                    'n_cells': str(total_calculated_cells),
                    'n_permutations': str(n_permutations),
                }
                if 'YtoX' in value_key_to_compare:
                    final_return_dict_[e]['source'] = y_axis_name
                    final_return_dict_[e]['target'] = x_axis_name
                else:
                    final_return_dict_[e]['source'] = x_axis_name
                    final_return_dict_[e]['target'] = y_axis_name
    else: #! [0 ~ 1]
        for e, value_key_to_compare in enumerate(to_report_): # 0 ~ 11, YtoX_pos_average
            if 'pos' in value_key_to_compare:
                final_return_dict_[e] = {
                    'gene_set' :  gene_set_ID,
                    'compare_with': 'background_rep' + str(args_dict['replication_number']),
                    'condition': args_dict['condition'],
                    'attribute' : value_key_to_compare.split('_')[2],
                    'sign': value_key_to_compare.split('_')[1],
                    'pvalue': str(to_report_[value_key_to_compare]),
                    'n_cells': str(total_calculated_cells),
                    'n_permutations': str(n_permutations),
                }
                if 'YtoX' in value_key_to_compare:
                    final_return_dict_[e]['source'] = y_axis_name
                    final_return_dict_[e]['target'] = x_axis_name
                else:
                    final_return_dict_[e]['source'] = x_axis_name
                    final_return_dict_[e]['target'] = y_axis_name
            else:
                pass
    return final_return_dict_


def main(task_name, resut_save_dir, gene_set_dir, 
         n_permutations, pvalue_method, significance_threshold,
         region_1_name, region_2_name, 
         original_correlation_dir, null_correlation_dir
         ) -> None:
    args_dict = {
        'pvalue_method': pvalue_method,
        'n_permutations': n_permutations,
        'sign_handler': 'naive',
        'replication_number': int(null_correlation_dir.split('_')[-1]),
        'significance_threshold': float(significance_threshold),
    }
    attr_dict = {
            'control': ('control_correlation',
                        'control_correlation_pvalue_perm_test',
                        'control_correlation_pvalue_scipy',),
            'schizo': ('schizo_correlation',
                        'schizo_correlation_pvalue_perm_test',
                        'schizo_correlation_pvalue_scipy',),
            'difference': ('correlation_difference',
                        'correlation_difference_pvalue_perm_test',
                        'correlation_difference_pvalue_scipy',),
    }
    gene_set_dictionary = pickle.load(open(file=gene_set_dir, mode='rb'))
    """
        attributes for following variables are 
        ('control_correlation', 'control_correlation_pvalue_perm_test','control_correlation_pvalue_scipy',
        'schizo_correlation', 'schizo_correlation_pvalue_perm_test', 'schizo_correlation_pvalue_scipy',
        'correlation_difference', 'correlation_difference_pvalue_perm_test', 'correlation_difference_pvalue_scipy',),
    """
    region_pairs = [
        '#'.join([region_1_name, region_2_name]),
        # '#'.join([region_2_name, region_1_name]),
    ] # a to b and b to A
    full_list_of_df = []
    for region_pair_name in region_pairs:
        original_correlations = load_corr_result(original_correlation_dir) #! real correlation results
        background_correlations = load_corr_result(null_correlation_dir) #! background correlation results
        for gene_set_id in gene_set_dictionary:
            for condition_ in attr_dict: # to handle differences and cases
                args_dict['region_pair'] = region_pair_name
                args_dict['condition'] = condition_
                args_dict['gene_set'] = gene_set_id
                # 데이터프레임을 확실하게 어사인하기.
                corr_values_real = original_correlations[attr_dict[condition_][0]] # control_correlation
                if pvalue_method == 'permutation':
                    corr_significance_real = original_correlations[attr_dict[condition_][1]] # control_correlation_pvalue_perm_test
                elif pvalue_method == 'scipy':
                    corr_significance_real = original_correlations[attr_dict[condition_][2]] # control_correlation_pvalue_scipy
                corr_values_null_this_rep = background_correlations[attr_dict[condition_][0]]
                if pvalue_method == 'permutation':
                    corr_significance_null_this_rep = background_correlations[attr_dict[condition_][1]]
                    # control_correlation_pvalue_perm_test
                elif pvalue_method == 'scipy':
                    corr_significance_null_this_rep = background_correlations[attr_dict[condition_][2]]
                    # control_correlation_pvalue_scipy
                result_this_ = report_significance(
                    corr_values_real, corr_significance_real, 
                    corr_values_null_this_rep, corr_significance_null_this_rep,
                    gene_set_id, gene_set_dictionary[gene_set_id],
                    args_dict, 
                )
                this_result_dataframe = pd.DataFrame.from_dict(result_this_, orient='index')
                this_result_dataframe = this_result_dataframe[[
                    'gene_set', 'compare_with', 'source', 'target',
                    'condition', 'sign', 'attribute', 'pvalue', 
                    'n_cells', 'n_permutations', ]]
                print(this_result_dataframe)
                full_list_of_df.append(this_result_dataframe)
    final_result = pd.concat(full_list_of_df, axis=0)
    final_result.to_csv(os.path.join(resut_save_dir, task_name, 'connections.csv'))
    return None


if __name__ == '__main__':
    import argparse
    print("Initializing...")
    parser = argparse.ArgumentParser(description="Calculate correlation between regions")
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--result-save-dir", type=str, required=True)
    parser.add_argument("--gene-set-dir", type=str, required=True)
    parser.add_argument("--n-permutations", type=str, required=True)
    parser.add_argument("--pvalue-method", type=str, required=True)
    parser.add_argument("--significance-threshold", type=str, required=True)
    parser.add_argument("--region-1-name", type=str, required=True)
    parser.add_argument("--region-2-name", type=str, required=True)
    parser.add_argument("--original-correlation-dir", type=str, required=True)
    parser.add_argument("--null-correlation-dir", type=str, required=True)
    parsed_args = parser.parse_args()

    task_name = parsed_args.task_name
    resut_save_dir = parsed_args.result_save_dir
    gene_set_dir = parsed_args.gene_set_dir
    n_permutations = parsed_args.n_permutations
    pvalue_method = parsed_args.pvalue_method
    significance_threshold = parsed_args.significance_threshold
    region_1_name = parsed_args.region_1_name
    region_2_name = parsed_args.region_2_name
    original_correlation_dir = parsed_args.original_correlation_dir
    null_correlation_dir = parsed_args.null_correlation_dir

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print('|'.join([task_name, n_permutations, significance_threshold, region_1_name, region_2_name]), 'start...')

    if not os.path.exists(os.path.join(resut_save_dir, task_name)):
        os.makedirs(os.path.join(resut_save_dir, task_name))
    
    if not os.path.isfile(
        os.path.join(resut_save_dir, task_name, 'connections.csv')
    ): # if there are no result file
        main(task_name, resut_save_dir, gene_set_dir, 
             n_permutations, pvalue_method, significance_threshold,
             region_1_name, region_2_name, 
             original_correlation_dir, null_correlation_dir)

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print('|'.join([task_name, n_permutations, significance_threshold, region_1_name, region_2_name]), 'done!') 