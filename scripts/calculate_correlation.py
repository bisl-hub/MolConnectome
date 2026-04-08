import os
import datetime
import sys
import json
import pickle

from functools import reduce
from typing import Any
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import scipy.stats

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))


def linregress_wrapper(var_1: list[float] | np.ndarray[float], 
                       var_2: list[float] | np.ndarray[float], ) -> float:
    linregress_result = scipy.stats.linregress(var_1, var_2, 
                                               alternative='two-sided',
                                               nan_policy='omit')
    slope = linregress_result.slope
    return slope


def pcc_pvalue_beta_dist_wrapper(var_1: list[float] | np.ndarray[float], 
                                 var_2: list[float] | np.ndarray[float], ) -> float:
    _corr, _pval = scipy.stats.pearsonr(var_1, var_2)
    return _pval


def spearman_pvalue_beta_dist_wrapper(var_1: list[float] | np.ndarray[float], 
                       var_2: list[float] | np.ndarray[float], ) -> float:
    _corr, _pval = scipy.stats.spearmanr(var_1, var_2)
    return _pval


def linregress_pvalue_t_test_wrapper(var_1: list[float] | np.ndarray[float], 
                                      var_2: list[float] | np.ndarray[float], ) -> float:
    linregress_result = scipy.stats.linregress(var_1, var_2, 
                                               alternative='two-sided',
                                               nan_policy='omit')
    _pval = linregress_result.pvalue
    return _pval


def load_expression_matrix(expression_data_dir: str):
    expression_matrix = pd.read_csv(expression_data_dir, index_col=0)
    return expression_matrix


def load_gene_list(genes_to_analyze_dir: str) -> list[str]:
    if genes_to_analyze_dir.__contains__('npy'):
        with open(file=genes_to_analyze_dir, mode='rb') as f:
            genes_to_analyze = np.load(f, allow_pickle=True) # numpy array
        genes_to_analyze = genes_to_analyze.tolist()
    elif genes_to_analyze_dir.__contains__('txt'):
        with open(file=genes_to_analyze_dir, mode='r') as f:
            genes_to_analyze = f.read().splitlines()
    elif genes_to_analyze_dir.__contains__('pickle'):
        with open(file=genes_to_analyze_dir, mode='rb') as f:
            genes_to_analyze = pickle.load(f)
    else:
        raise ValueError(f'genes_to_analyze_dir is not in supported format, or is malformed: {genes_to_analyze_dir}')
    return np.array(genes_to_analyze)


def corr_calculator(real_data_R1: 'pd.DataFrame', real_data_R2: 'pd.DataFrame', 
                    args_dict: dict[str, Any], ) -> 'dict[str, pd.DataFrame | Any]':
    correlation_metric = args_dict['correlation_metric']
    n_permutations: int = args_dict['n_permutations']
    merged = pd.concat([real_data_R1, real_data_R2], axis=0)
    transposed_ = merged.T 
    if correlation_metric == 'pcc':
        real_corr_mat = transposed_.corr(method='pearson')
        scipy_pvalue_mat = transposed_.corr(method=pcc_pvalue_beta_dist_wrapper)
    elif correlation_metric == 'spearman':
        real_corr_mat = transposed_.corr(method='spearman')
        scipy_pvalue_mat = transposed_.corr(method=spearman_pvalue_beta_dist_wrapper)
    elif correlation_metric == 'linear_regression':
        real_corr_mat = transposed_.corr(method=linregress_wrapper)
        scipy_pvalue_mat = transposed_.corr(method=linregress_pvalue_t_test_wrapper)
    else:
        raise ValueError (args_dict)

    initial_seed = 20250716; rng = np.random.default_rng(seed=initial_seed)
    selected_seeds: np.ndarray[int] = rng.integers(initial_seed, initial_seed*2, n_permutations*2) # 시드 콜렉션
    permutated_corr_collection = []
    for e in range(n_permutations):
        this_perm_seed_1 = selected_seeds[e*2]; this_perm_seed_2 = selected_seeds[(e*2)+1]
        this_perm_rng_1 = np.random.default_rng(seed=this_perm_seed_1)
        this_perm_rng_2 = np.random.default_rng(seed=this_perm_seed_2)
        shuffled_sample_order_R1 = this_perm_rng_1.permutation(real_data_R1.columns)
        shuffled_sample_order_R2 = this_perm_rng_2.permutation(real_data_R2.columns)
        perm_data_R1 = real_data_R1[shuffled_sample_order_R1]; perm_data_R2 = real_data_R2[shuffled_sample_order_R2]
        perm_data_R1.columns = range(len(perm_data_R1.columns));  
        perm_data_R2.columns = range(len(perm_data_R2.columns));  # pseudo-sampleids
        perm_merged = pd.concat([perm_data_R1, perm_data_R2], axis=0)
        perm_transposed_ = perm_merged.T # ! DataFrame.corr 이 칼럼와이즈 하게 계산하기 때문임. 
        if correlation_metric == 'pcc':
            perm_corr_mat = perm_transposed_.corr(method='pearson')
        elif correlation_metric == 'spearman':
            perm_corr_mat = perm_transposed_.corr(method='spearman')
        elif correlation_metric == 'linear_regression':
            perm_corr_mat = perm_transposed_.corr(method=linregress_wrapper)
        else:
            raise ValueError (args_dict)
        permutated_corr_collection.append(perm_corr_mat.to_numpy())
    permutated_corrs = np.array(permutated_corr_collection)
    larger_than = permutated_corrs < real_corr_mat.to_numpy()
    larger_really : np.ndarray = np.sum(larger_than, axis=0)
    significance_array = np.ones(larger_really.shape) - (larger_really / n_permutations)
    significance_dataframe = pd.DataFrame(data = significance_array, 
                                          index = real_corr_mat.index, 
                                          columns = real_corr_mat.columns)
    return {
        'real_correlation_matrix': real_corr_mat,
        'permutated_correlation_array': permutated_corrs,
        'scipy_correlation_pvalue_matrix': scipy_pvalue_mat,
        'permutation_test_pvalue_matrix': significance_dataframe,
    }


def reindex_and_allocate_values(real_corr_mat: 'pd.DataFrame', 
                                scipy_pvalue_mat: 'pd.DataFrame', 
                                significance_dataframe: 'pd.DataFrame', 
                                correlation_result_dict: 'dict[str, dict[str, pd.DataFrame]]',
                                real_common_genes: 'list[str] | np.ndarray[str]', 
                                R1_name: str, R2_name: str, 
                                disease_indicator: str,
                                args_dict: dict[str, Any], ) -> 'dict[str, dict[str, pd.DataFrame]]':
    to_use_index = [x for x in real_corr_mat.index if x.endswith(R1_name)]
    to_use_columns = [x for x in real_corr_mat.columns if x.endswith(R2_name)]
    # # 인덱스랑 칼럼 이름 다시 바꾸기
    essential_corr = real_corr_mat.loc[to_use_index][to_use_columns]
    try:
        essential_corr.index = real_common_genes; essential_corr.columns = real_common_genes; 
    except ValueError:
        raise ValueError(R1_control_name, R2_control_name, args_dict)
    essential_scipy_pval = scipy_pvalue_mat.loc[to_use_index][to_use_columns]
    essential_scipy_pval.index = real_common_genes; essential_scipy_pval.columns = real_common_genes; 
    essential_significance = significance_dataframe.loc[to_use_index][to_use_columns]
    essential_significance.index = real_common_genes; essential_significance.columns = real_common_genes; 
    
    # ! 밸류 얼로케이션 진행.
    '''
    'control_correlation', 'control_correlation_pvalue_perm_test', 'control_correlation_pvalue_scipy',
    'schizo_correlation', 'schizo_correlation_pvalue_perm_test', 'schizo_correlation_pvalue_scipy',
    'correlation_difference', 'correlation_difference_pvalue_perm_test', 'correlation_difference_pvalue_scipy',
    '''
    if disease_indicator == 'control':
        attributes_to_allocate = ['control_correlation', 
                                  'control_correlation_pvalue_perm_test', 
                                  'control_correlation_pvalue_scipy',]
    elif disease_indicator == 'schizo':
        attributes_to_allocate = ['schizo_correlation', 
                                  'schizo_correlation_pvalue_perm_test', 
                                  'schizo_correlation_pvalue_scipy',]
    elif disease_indicator == 'difference':
        attributes_to_allocate = ['correlation_difference',
                                  'correlation_difference_pvalue_perm_test',
                                  'correlation_difference_pvalue_scipy',]
    else:
        raise ValueError

    # ! absolute - use the absoluted value of correlation coefficients which were significant
    # correlation_result_dict['absolute'][attributes_to_allocate[0]].update(np.abs(essential_corr))
    # correlation_result_dict['absolute'][attributes_to_allocate[1]].update(essential_significance)
    # correlation_result_dict['absolute'][attributes_to_allocate[2]].update(essential_scipy_pval)
    # # significance_as_str = essential_significance.map(check_significance, sign = 'absolute', alpha=alpha)
    # # correlation_result_dict['absolute']['control_significance_perm_test'].update(significance_as_str)
    
    # # ! preserve - normalize the correlation coefficients into interval [0, 1]
    # correlation_result_dict['preserve'][attributes_to_allocate[0]].update(((essential_corr + 1.0) / 2.0))
    # correlation_result_dict['preserve'][attributes_to_allocate[1]].update(essential_significance)
    # correlation_result_dict['preserve'][attributes_to_allocate[2]].update(essential_scipy_pval)
    # # significance_as_str = essential_significance.map(check_significance, sign = 'preserve', alpha=alpha)
    # # correlation_result_dict['preserve']['control_significance_perm_test'].update(significance_as_str)
    
    # # ! naive - use original correlation values
    # correlation_result_dict['naive'][attributes_to_allocate[0]].update(essential_corr)
    # correlation_result_dict['naive'][attributes_to_allocate[1]].update(essential_significance)
    # correlation_result_dict['naive'][attributes_to_allocate[2]].update(essential_scipy_pval)
    # significance_as_str = essential_significance.map(check_significance, sign = 'naive', alpha=alpha)
    # correlation_result_dict['naive']['control_significance_perm_test'].update(significance_as_str)

    # ! ignore - do not use any negative correlations regardless of their significance
    correlation_result_dict['ignore'][attributes_to_allocate[0]].update(essential_corr[essential_corr > 1e-8])
    correlation_result_dict['ignore'][attributes_to_allocate[1]].update(essential_significance)
    correlation_result_dict['ignore'][attributes_to_allocate[2]].update(essential_scipy_pval)
    # significance_as_str = essential_significance.map(check_significance, sign = 'ignore', alpha=alpha)
    # correlation_result_dict['ignore']['control_significance_perm_test'].update(significance_as_str)
    return correlation_result_dict


def calculate_correlation(R1_control_expression_matrix: pd.DataFrame, R2_control_expression_matrix: pd.DataFrame, 
                          R1_schizo_expression_matrix: pd.DataFrame, R2_schizo_expression_matrix: pd.DataFrame, 
                          R1_control_name: str, R2_control_name: str, 
                          R1_schizo_name: str, R2_schizo_name: str, 
                          args_dict: dict[str, Any], log_object: dict[str, Any]) -> \
                              'tuple[dict[str, pd.DataFrame], dict[str, Any]]':
    # ! 레거시 호환 - 변수들 할당하기
    R1_name = R1_control_name.split('.')[0]
    R2_name = R2_control_name.split('.')[0]
    expression_values_data_R1_control = R1_control_expression_matrix
    expression_values_data_R2_control = R2_control_expression_matrix
    expression_values_data_R1_schizo = R1_schizo_expression_matrix
    expression_values_data_R2_schizo = R2_schizo_expression_matrix
    correlation_metric = args_dict['correlation_metric']
    n_permutations: int = args_dict['n_permutations']
    alpha: float = args_dict['significance_threshold']
    
    # ! 쓰는 유전자만 슬라이스하기
    expression_values_list = [expression_values_data_R1_control, expression_values_data_R2_control,
                              expression_values_data_R1_schizo, expression_values_data_R2_schizo, ]
    genes__ = [exprs_.index for exprs_ in expression_values_list]
    common_genes = reduce(lambda left, right: left.intersection(right), genes__)
    union_genes = reduce(lambda left, right: left.union(right), genes__)
    
    # ! 변동 사항 - 뉴런챗 유전자들도 고려한 실행.
    to_calculate_genes = load_gene_list(args_dict['genes_to_analyze_dir'])
    real_common_genes = np.intersect1d(to_calculate_genes, common_genes) #FF0000

    
    # ! 샘플 매칭하고 매칭되는 샘플들만 남기기
    samples_control_both = expression_values_data_R1_control.columns.intersection(expression_values_data_R2_control.columns)
    samples_schizo_both = expression_values_data_R1_schizo.columns.intersection(expression_values_data_R2_schizo.columns)
    
    samples_control_R1 = samples_control_both
    samples_control_R2 = samples_control_both
    samples_schizo_R1 = samples_schizo_both
    samples_schizo_R2 = samples_schizo_both

    # ! 로그에 쓸 변수 저장
    log_object['genes'] = list(real_common_genes)
    log_object['R1 control samples'] = list(samples_control_R1)
    log_object['R2 control samples'] = list(samples_control_R2)
    log_object['R1 schizo samples'] = list(samples_schizo_R1)
    log_object['R2 schizo samples'] = list(samples_schizo_R2)

    # ! 리턴 데이터 기본 구조 어사인하기
    sign_handlers_ = ['absolute', 'preserve', 'naive', 'ignore']
    attributes_ = ['control_correlation',
                   'control_correlation_pvalue_perm_test',
                   'control_correlation_pvalue_scipy',
                   'schizo_correlation',
                   'schizo_correlation_pvalue_perm_test',
                   'schizo_correlation_pvalue_scipy',
                   'correlation_difference',
                   'correlation_difference_pvalue_perm_test',
                   'correlation_difference_pvalue_scipy',
                   ]
    correlation_result_dict = {
        x_: {attr_: pd.DataFrame(index=real_common_genes, columns=real_common_genes)
         for attr_ in attributes_ } for x_ in sign_handlers_
    }

    # ! 여기서부터 바꿔야 됨 -> 벌크 단위로 돌아가야 함.
    # ! 원숭이에 적용한 부분에는 조현병이 없었기 때문에 이 부분도 추가 계산을 해주어야 함.
    real_data_R1_control: 'pd.DataFrame' = expression_values_data_R1_control.loc[real_common_genes][samples_control_R1].copy()
    real_data_R2_control: 'pd.DataFrame' = expression_values_data_R2_control.loc[real_common_genes][samples_control_R2].copy()
    real_data_R1_schizo: 'pd.DataFrame' = expression_values_data_R1_schizo.loc[real_common_genes][samples_schizo_R1].copy()
    real_data_R2_schizo: 'pd.DataFrame' = expression_values_data_R2_schizo.loc[real_common_genes][samples_schizo_R2].copy() 
    # ! both should have same dimensions
    real_data_R1_control.columns = samples_control_R1; real_data_R2_control.columns = samples_control_R1
    real_data_R1_schizo.columns = samples_schizo_R1; real_data_R2_schizo.columns = samples_schizo_R1

    if R1_name == R2_name:
        R1_name = R1_name + '_1'
        R2_name = R2_name + '_2'
    real_data_R1_control = real_data_R1_control.add_suffix('_' + R1_name, axis=0)
    real_data_R2_control = real_data_R2_control.add_suffix('_' + R2_name, axis=0)
    real_data_R1_schizo = real_data_R1_schizo.add_suffix('_' + R1_name, axis=0)
    real_data_R2_schizo = real_data_R2_schizo.add_suffix('_' + R2_name, axis=0)

    # ! 코릴레이션 계산 진행.
    control_correlation_result_object = corr_calculator(real_data_R1_control, real_data_R2_control, args_dict)
    schizo_correlation_result_object = corr_calculator(real_data_R1_schizo, real_data_R2_schizo, args_dict)
    '''
    return {
        'real_correlation_matrix': real_corr_mat,
        'permutated_correlation_array': permutated_corrs,
        'scipy_correlation_pvalue_matrix': scipy_pvalue_mat,
        'permutation_test_pvalue_matrix': significance_dataframe,
    }
    '''

    real_correlation_difference: 'pd.DataFrame' = (schizo_correlation_result_object['real_correlation_matrix'] - 
                                                   control_correlation_result_object['real_correlation_matrix'] )
    correlation_difference_permutated_array: 'np.ndarray' = (schizo_correlation_result_object['permutated_correlation_array'] - 
                                                             control_correlation_result_object['permutated_correlation_array'] )
    
    difference_larger_than = correlation_difference_permutated_array < real_correlation_difference.to_numpy()
    difference_larger_really : np.ndarray = np.sum(difference_larger_than, axis=0)
    difference_significance_array = np.ones(difference_larger_really.shape) - (difference_larger_really / n_permutations)
    difference_significance_dataframe = pd.DataFrame(data = difference_significance_array, 
                                          index = real_correlation_difference.index, 
                                          columns = real_correlation_difference.columns)
    
    correlation_result_dict = reindex_and_allocate_values(control_correlation_result_object['real_correlation_matrix'], 
                                control_correlation_result_object['scipy_correlation_pvalue_matrix'], 
                                control_correlation_result_object['permutation_test_pvalue_matrix'], 
                                correlation_result_dict, 
                                real_common_genes, R1_name, R2_name, 
                                'control', args_dict)
    correlation_result_dict = reindex_and_allocate_values(schizo_correlation_result_object['real_correlation_matrix'], 
                                schizo_correlation_result_object['scipy_correlation_pvalue_matrix'], 
                                schizo_correlation_result_object['permutation_test_pvalue_matrix'], 
                                correlation_result_dict, 
                                real_common_genes, R1_name, R2_name, 
                                'schizo', args_dict)
    correlation_result_dict = reindex_and_allocate_values(real_correlation_difference, 
                                difference_significance_dataframe, 
                                difference_significance_dataframe, 
                                correlation_result_dict, 
                                real_common_genes, R1_name, R2_name, 
                                'difference', args_dict)
    return correlation_result_dict, log_object


def main(task_name_: str, result_saving_directory_: str, gene_list_dir_: str, 
         correlation_metric_: str, significance_threshold_: str, 
         n_permutations_corr_calc: str|int,
         region_1_control_data_dir: str, region_2_control_data_dir: str, 
         region_1_schizo_data_dir: str, region_2_schizo_data_dir: str) -> None:
    args_dict = {
        'genes_to_analyze_dir': gene_list_dir_,
        'correlation_metric': correlation_metric_,
        'significance_threshold' : float(significance_threshold_),
        'n_permutations': int(n_permutations_corr_calc),
    }
    log_json_obj = {}
    R1_control_name = os.path.basename(region_1_control_data_dir)
    R2_control_name = os.path.basename(region_2_control_data_dir)
    R1_schizo_name = os.path.basename(region_1_schizo_data_dir)
    R2_schizo_name = os.path.basename(region_2_schizo_data_dir)
    log_json_obj['task name'] = task_name_
    log_json_obj['file names'] = [R1_control_name, R2_control_name, R1_schizo_name, R2_schizo_name]
    log_json_obj['start time'] = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    R1_control_data = load_expression_matrix(region_1_control_data_dir); 
    R2_control_data = load_expression_matrix(region_2_control_data_dir) 
    R1_schizo_data = load_expression_matrix(region_1_schizo_data_dir); 
    R2_schizo_data = load_expression_matrix(region_2_schizo_data_dir)

    this_combination_result, log_json_obj = calculate_correlation(R1_control_data, R2_control_data, 
                                                                  R1_schizo_data, R2_schizo_data, 
                                                                  R1_control_name, R2_control_name, 
                                                                  R1_schizo_name, R2_schizo_name, 
                                                                  args_dict, log_json_obj )

    log_json_obj['finish time'] = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    log_json_obj = {**args_dict, **log_json_obj}

    # for sign_handler_ in this_combination_result:
    sign_handler_ = 'ignore'
    for attr_ in this_combination_result[sign_handler_]:
        this_result_save_dir = os.path.join(result_saving_directory_, task_name_, attr_ + '.csv')
        this_combination_result[sign_handler_][attr_].to_csv(this_result_save_dir, sep=',')
    log_json_save_dir = os.path.join(result_saving_directory_, task_name_, 'logs.json')
    with open(log_json_save_dir, 'w', encoding='utf-8') as jf_:
        json.dump(log_json_obj, jf_, ensure_ascii=False, indent=4)

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print(task_name_, 'saved!')
    return None

    
if __name__ == '__main__':
    import argparse
    print("Initializing...")
    parser = argparse.ArgumentParser(description="Calculate correlation between regions")
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--result-save-dir", type=str, required=True)
    parser.add_argument("--gene-list-dir", type=str, required=True)
    parser.add_argument("--correlation-metric", type=str, required=True)
    parser.add_argument("--correlation-alpha", type=str, required=True)
    parser.add_argument("--n-perm-correlation", type=str, required=True)
    parser.add_argument("--region-1-control-data-dir", type=str, required=True)
    parser.add_argument("--region-2-control-data-dir", type=str, required=True)
    parser.add_argument("--region-1-schizo-data-dir", type=str, required=True)
    parser.add_argument("--region-2-schizo-data-dir", type=str, required=True)
    parsed_args = parser.parse_args()

    task_name = parsed_args.task_name
    resut_save_dir = parsed_args.result_save_dir
    gene_list_dir = parsed_args.gene_list_dir
    corr_metric = parsed_args.correlation_metric    
    corr_alpha = parsed_args.correlation_alpha
    n_perm_corr_calc = parsed_args.n_perm_correlation
    region_1_control_data_dir = parsed_args.region_1_control_data_dir
    region_2_control_data_dir = parsed_args.region_2_control_data_dir
    region_1_schizo_data_dir = parsed_args.region_1_schizo_data_dir
    region_2_schizo_data_dir = parsed_args.region_2_schizo_data_dir

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print('|'.join([task_name, corr_metric, corr_alpha, ]), 'start...')

    if not os.path.exists(os.path.join(resut_save_dir, task_name)):
        os.makedirs(os.path.join(resut_save_dir, task_name))
        # for sign_handler_ in ['absolute', 'preserve', 'naive', 'ignore']:
        #     os.makedirs(os.path.join(resut_save_dir, task_name, sign_handler_))
            

    if not os.path.isfile(
        os.path.join(resut_save_dir, task_name, 'logs.json')
    ): # if the log file does not exists
        main(task_name, resut_save_dir, gene_list_dir, 
             corr_metric, corr_alpha, n_perm_corr_calc, 
             region_1_control_data_dir, region_2_control_data_dir, 
             region_1_schizo_data_dir, region_2_schizo_data_dir)

    print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
    print('|'.join([task_name, corr_metric, corr_alpha, ]), 'done!') 

