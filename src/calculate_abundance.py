import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
import sys
import warnings

import numpy as np
import pandas as pd
from typing import Any

os.environ['OPENBLAS_NUM_THREADS'] = '1'


neurotransmitter_genes = {
    'DA':    {'class_1': ['TH',], 'class_2': ['DDC', ], },
    '5-HT':   {'class_1': ['TPH1', 'TPH2', ], 'class_2': ['DDC', ], },
    'Glu_1':  {'class_1': ['CTPS2', 'GLS', 'GLS2',], },
    'Glu_2':  {'class_1': ['GOT1', 'GOT2', 'GLUD1', 'GLUD2', ], },
    'GABA':   {'class_1': ['GAD1', 'GAD2', ], },
    'NE_1':   {'class_1': ['DBH',], },
    'NE_2':   {'class_1': ['TH', ], 'class_2': ['DDC', ], 'class_3': ['DBH', ], },
    'ACh':    {'class_1': ['CHAT',], },
    'Gly':    {'class_1': ['SHMT1', 'SHMT2',], },
    'epi_1':  {'class_1': ['PNMT',], },
    'epi_2':  {'class_1': ['TH', ], 'class_2': ['DDC', ], 'class_3': ['DBH', ], 'class_4': ['PNMT', ], },
    'Gas_CO': {'class_1': ['HMOX1', 'HMOX2', ], },
    'Gas_NO': {'class_1': ['NOS1', 'NOS2', 'NOS3', ], },
    'Tyr':    {'class_1': ['PAH',], },
    'NP':     {'class_1': ['ADCYAP1', 'CCK', 'CORT', 'CRH', 'GRP', 'NMB', 'NPY', 'NTS',
                           'PDYN', 'PENK', 'PNOC', 'PTHLH', 'RLN1', 'SST', 'TAC1', 'TAC3',
                           'TRH', 'VIP']},
    'Syn':     {'class_1': ['NRXN1', 'NRXN2', 'NRXN3',], },
}
total_synthesis_genes = [neurotransmitter_genes[neurotransmitter_name][class_id]
                         for neurotransmitter_name in neurotransmitter_genes
                         for class_id in neurotransmitter_genes[neurotransmitter_name]]
essential_synthesis_genes = [
    x for sublist in total_synthesis_genes for x in sublist]
essential_synthesis_genes = np.unique(essential_synthesis_genes)
# ! 기능 밎 구역 순
MONKEY_REGIONS = ['dlPFC',  'dmPFC', 'vlPFC', 'vmPFC',
                  'M1', 'STS', 'ACCg', 'V1',
                  'CN', 'Pu', 'LGN',
                  'VMH', 'AMY', 'CA3', 'DG',]
HUMAN_REGIONS = ['PFC', 'HP', 'CC', 'CN', 'NAc']
HUMAN_REGIONS_TO_DRAW = ['PFC', 'CC', 'HP', 'CN', 'NAc']
N_SAMPLE_SHUFFLES = 100
INIT_SEED = 20250924
SHUFFLED_ABUNDANCE_SAVE = '/home/ohkwon/Documents/FinalSchizoProject/analysis_results/9.network_human_shuffle'


def calculate_trimean(expression_array: np.ndarray, axis_: int):
    Q2 = np.nanquantile(expression_array, 0.5, axis=axis_)
    Q1 = np.nanquantile(expression_array, 0.75, axis=axis_)
    Q3 = np.nanquantile(expression_array, 0.25, axis=axis_)
    trimean_ = (0.5 * Q2) + (0.25 * (Q1 + Q3))
    if np.all(expression_array < 1e-8):
        trimean_ = 0.0
    return trimean_


def calculate_ligand_abundance(aggreagated_gene_expressions: pd.DataFrame) -> pd.DataFrame:
    '''결과적으로, 이 리간드에 관련된 유전자들의 값을 활용해
    딕셔너리로 리턴을 해주어야 함. 그 딕셔너리는 키가 리전이고
    밸류는 하나의 어뷴던스 값이 될 것임.
    ligand_participants_and_membership_dict는
    {'class_1': array(['GAD1', 'GAD2'], dtype='<U7'), 
     'class_2': array(['SLC32A1'], dtype='<U7')} 같이 생겼음.
    '''
    final_dict = {}
    for neurotransmitter_name in neurotransmitter_genes:  # DA, Glu, ...
        this_NT_genes_dict = neurotransmitter_genes[neurotransmitter_name]
        # {'class_1' : ['TH', ], 'class_2': ['DDC', ], 'class_3': ['DBH', ],},
        this_neurotransmitter_whole = []
        for each_class in this_NT_genes_dict:  # class_1, class_2, class_3
            try:
                this_class_genes = this_NT_genes_dict[each_class]  # ['DBH', ]
                this_class_expression = aggreagated_gene_expressions.loc[this_class_genes]
            except KeyError:
                this_class_genes = this_NT_genes_dict[each_class]  # ['DBH', ]
                common_genes = np.intersect1d(
                    this_class_genes, aggreagated_gene_expressions.index)
                not_common_genes = [
                    x for x in this_class_genes if x not in common_genes]
                this_class_expression = aggreagated_gene_expressions.loc[common_genes].copy(
                )
                not_common_genes_expr = pd.DataFrame(
                    data=np.full(
                        (len(not_common_genes), this_class_expression.shape[1]), np.nan),
                    index=not_common_genes,
                    columns=this_class_expression.columns
                )
                this_class_expression = pd.concat(
                    [this_class_expression, not_common_genes_expr], axis=0)
            this_class_expression = this_class_expression.fillna(0.0)
            this_class_aggregated = np.average(
                this_class_expression.to_numpy(), axis=0)
            this_neurotransmitter_whole.append(this_class_aggregated)
        prod_ = np.cumprod(this_neurotransmitter_whole, axis=0)
        n_total_classes = len(list(this_NT_genes_dict.keys()))
        abundance_ = np.power(prod_[-1], (1/n_total_classes))
        final_dict[neurotransmitter_name] = abundance_
    final_ = pd.DataFrame.from_dict(final_dict, orient='index')
    final_.columns = aggreagated_gene_expressions.columns
    glu_final = np.average(
        final_.loc[['Glu_1', 'Glu_2',]].to_numpy(), axis=0).reshape(1, -1)
    glu_final = pd.DataFrame(data=glu_final, index=[
                             'Glu', ], columns=final_.columns)
    NE_final = final_.loc[['NE_2',]].copy()
    epi_final = final_.loc[['epi_2',]].copy()
    NE_final.index = ['NE',]
    epi_final.index = ['epi',]
    gas_final = np.average(
        final_.loc[['Gas_NO', 'Gas_CO',]].to_numpy(), axis=0).reshape(1, -1)
    gas_final = pd.DataFrame(data=gas_final, index=[
                             'Gas', ], columns=final_.columns)
    final_ = pd.concat([final_, glu_final, NE_final, epi_final, gas_final])
    return final_

# 결과적으로 만들어야 할 데이터는 두 종류임.
# 하나는 저 위에 나온 각각의 합성 효소들의 개별적인 리전 별 평균값 (미디언, 민, 트라이민 셋 다 저장)
# 두번째로는 이 값을 활용해 유추된 각 리전 별 뉴로트랜스미터의 어번던스 값 (미디언, 민, 트라이민 셋 다 사용해 계산)
# 이 값을 활용하여 웨이트로 사용해 보는 것을 고려.


def calculate_synthesis_gene_expressions(gene_expression_data: 'pd.DataFrame[float]',
                                         list_of_samples: 'np.ndarray[str]',) -> 'pd.DataFrame[float]':
    ''' 하나의 리전에 대해 값 세개가 리턴되어야 함.'''
    genes_to_extract = np.intersect1d(
        essential_synthesis_genes, gene_expression_data.index)
    essential_dataframe = gene_expression_data.loc[genes_to_extract][list_of_samples]
    essential_dataframe = essential_dataframe + 2.0
    mean_values = np.nanmean(essential_dataframe.to_numpy(), axis=1)
    median_values = np.nanmedian(essential_dataframe.to_numpy(), axis=1)
    trimean_values = calculate_trimean(essential_dataframe.to_numpy(), axis_=1)
    to_return = {
        'mean': mean_values, 'median': median_values, 'trimean': trimean_values,
    }
    to_return_df = pd.DataFrame.from_dict(to_return, orient='columns')
    to_return_df.index = genes_to_extract
    return to_return_df


def calculate_neurotransmitter_abundance(expression_data: 'pd.DataFrame[float]',
                                         condition: str):

    dataset_dict_essential = {}
    for each_line_index in expression_data.index:
        if expression_data.loc[each_line_index, 'condition'] == condition:
            region_and_condition = expression_data.loc[each_line_index,
                                                       'region_name'] + '.' + condition
            expression_directory = expression_data.loc[each_line_index,
                                                       'expression_dir']
            dataset_dict_essential[region_and_condition] = pd.read_csv(
                expression_directory, index_col=0)
    # dataset_dict_essential = {x: dataset_dict_renamed[x] for x in dataset_dict_renamed
    #                           if condition in x}
    # 각각의 데이터셋은 Array_Collection_Prefrontal_Cortex.control 같은 키로 매핑되어 있음.
    # 이를 우선 각각의 질병 별로 구별한 다음, 데이터셋 이름 축약하고 리어사인 해서
    # 동일한 방식으로 어번던스 계산하면 됨.
    samples_dict = {
        x: np.array(dataset_dict_essential[x].columns) for x in dataset_dict_essential
    }
    # 우선 필요한 유전자들의 발현량부터 먼저 계산하기
    mean_collection_ = []
    median_collection_ = []
    trimean_collection_ = []
    for region_name in samples_dict:
        values_ = calculate_synthesis_gene_expressions(dataset_dict_essential[region_name],
                                                       samples_dict[region_name])
        mean_collection_.append(values_['mean'])
        median_collection_.append(values_['median'])
        trimean_collection_.append(values_['trimean'])
    mean_dataframe = pd.concat(mean_collection_, axis=1, join='outer')
    median_dataframe = pd.concat(median_collection_, axis=1, join='outer')
    trimean_dataframe = pd.concat(trimean_collection_, axis=1, join='outer')
    mean_dataframe.columns = samples_dict.keys()
    median_dataframe.columns = samples_dict.keys()
    trimean_dataframe.columns = samples_dict.keys()
    # print(essential_synthesis_genes) # GLUD2가 없음
    mean_abundance = calculate_ligand_abundance(mean_dataframe)
    median_abundance = calculate_ligand_abundance(median_dataframe)
    trimean_abundance = calculate_ligand_abundance(trimean_dataframe)
    return mean_abundance, median_abundance, trimean_abundance


def multiply_weight_with_adjacency(each_gs_adj_dict: 'dict[str, pd.DataFrame[float]]',
                                   gene_set_names_to_merge: list[str],
                                   merge_weights: 'pd.DataFrame[float]',
                                   filter_regions=False):
    # 우선 웨이트들을 노말라이즈 진행, 가장 큰 값으로 나누기.
    normalized_weights = merge_weights / merge_weights.max(axis=None)
    weighted_collection = {}
    REGIONS = [x.split('.')[0] for x in merge_weights.columns]
    named_neurotransmitters = ['DA', '5-HT', 'Glu', 'GABA',
                               'NE', 'ACh', 'Gly', 'epi',
                               'NP', 'Syn', 'Gas', ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for neurotransmitter_name in named_neurotransmitters:  # 'DA', 'Glu', ...
            try:
                this_NT_id = [x for x in gene_set_names_to_merge if x.__contains__(
                    neurotransmitter_name)][0]  # 하나의 값
                weights_per_region: 'np.ndarray' = normalized_weights.loc[neurotransmitter_name].to_numpy(
                )
                this_NT_multiplied = np.multiply(each_gs_adj_dict[this_NT_id].to_numpy(),
                                                 weights_per_region.reshape(-1, 1))
                NT_id_new = 'weighted_' + neurotransmitter_name
                weighted_collection[NT_id_new] = this_NT_multiplied
            except IndexError:  # 'Reactome_Gly'
                pass
    '''NaN 어사인 해주어야 함.곱한 다음에 0일 것이기 때문에 일부러 굳이 NaN으로 바꿔 줘야 함.'''

    final_nt_collection_ = {}
    # index: NT, columns: regions. if abundance = 0: True.
    nan_regions_identifier = (merge_weights < 1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for neurotransmitter_name in named_neurotransmitters:  # 'DA', 'Glu', ...
            # new_nt_id = [x for x in gene_set_names_to_merge if x.__contains__(neurotransmitter_name)][0] # 하나의 값
            try:
                weighted_NT_id = 'weighted_' + neurotransmitter_name
                weighted_adj = pd.DataFrame(data=weighted_collection[weighted_NT_id],
                                            index=REGIONS,
                                            columns=REGIONS)
                if filter_regions:
                    for region_ in REGIONS:
                        if nan_regions_identifier.loc[neurotransmitter_name][region_]:
                            weighted_adj.loc[region_] = np.nan
            except KeyError:  # 'weighted_Gap'
                pass
            except IndexError:  # 'Reactome_Gly'
                pass
            final_nt_collection_[weighted_NT_id] = weighted_adj
    return final_nt_collection_


def weight_sum_adjacencies(each_gs_adj_dict: 'dict[str, pd.DataFrame[float]]',
                           gene_set_names_to_merge: list[str],
                           merge_weights: 'pd.DataFrame[float]',
                           organism: str,
                           weight_other_groups=False):
    # 우선 웨이트들을 노말라이즈 진행, 가장 큰 값으로 나누기.
    normalized_weights = merge_weights / merge_weights.max(axis=None)
    weighted_collection = []
    named_neurotransmitters = ['DA', '5-HT',
                               'Glu', 'GABA', 'NE', 'ACh', 'Gly', 'epi',]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for neurotransmitter_name in named_neurotransmitters:  # 'DA', 'Glu', ...
            try:
                this_NT_id = [x for x in gene_set_names_to_merge if x.__contains__(
                    neurotransmitter_name)][0]  # 하나의 값
                weights_per_region: 'np.ndarray' = normalized_weights.loc[neurotransmitter_name].to_numpy(
                )
                this_NT_multiplied = np.multiply(each_gs_adj_dict[this_NT_id].to_numpy(),
                                                 weights_per_region.reshape(-1, 1))
                weighted_collection.append(this_NT_multiplied)
            except IndexError:  # 'Reactome_Gly'
                pass
        if weight_other_groups:
            others_ = ['NP', 'Syn', 'Gap', 'Gas', ]
            for other_name in others_:
                try:
                    this_NT_id = [x for x in gene_set_names_to_merge if x.__contains__(
                        other_name)][0]  # 하나의 값
                    weights_per_region: 'np.ndarray' = normalized_weights.loc[other_name].to_numpy(
                    )
                    this_NT_multiplied = np.multiply(each_gs_adj_dict[this_NT_id].to_numpy(),
                                                     weights_per_region.reshape(-1, 1))
                    weighted_collection.append(this_NT_multiplied)
                except IndexError:  # 'Reactome_Gly'
                    pass
                except KeyError:  # 'Gap'
                    weighted_collection.append(
                        each_gs_adj_dict[this_NT_id].to_numpy())
        else:  # 웨이트 주지 말기
            not_considered_gene_sets = [z for z in gene_set_names_to_merge
                                        if z.split('_')[1] not in ['DA', '5-HT', 'Glu', 'GABA', 'NE', 'ACh', 'Gly', 'epi',]
                                        ]  # 'Other', 'Gas', ...
            for name_ in not_considered_gene_sets:
                weighted_collection.append(each_gs_adj_dict[name_].to_numpy())
    weighted_sum = np.array(weighted_collection).sum(axis=0)
    if (organism == 'human') or (organism == 'hsa') or (organism == 'HSA'):
        regions = HUMAN_REGIONS_TO_DRAW
    elif (organism == 'monkey') or (organism == 'macaque') or (organism == 'MMul') or (organism == 'mmul'):
        regions = MONKEY_REGIONS
    to_return = pd.DataFrame(data=weighted_sum, index=regions, columns=regions)
    return to_return


if __name__ == '__main__':
    pass