"""Ligand abundance from synthesis-enzyme expression, and abundance-weighted connectivity.

Used by: macaque_abundance.py, macaque_weighted_connectivity.py, human_abundance.py,
         human_weighted_connectivity.py, human_group_abundance_compare.py.
"""

import os
import sys
import warnings

from pathlib import Path
import yaml
with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)
PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from typing import Any


neurotransmitter_genes = {
    'DA':    {'class_1': ['TH',], 'class_2': ['DDC', ], },
    '5-HT':   {'class_1': ['TPH1', 'TPH2', ], 'class_2': ['DDC', ], },
    'Glu_1':  {'class_1': ['CTPS2', 'GLS', 'GLS2',], },
    'Glu_2':  {'class_1': ['GOT1', 'GOT2', 'GLUD1', 'GLUD2', ], },
    'GABA':   {'class_1': ['GAD1', 'GAD2', ], },
    'NE_2':   {'class_1': ['TH', ], 'class_2': ['DDC', ], 'class_3': ['DBH', ], },
    'ACh':    {'class_1': ['CHAT',], },
    'Gly':    {'class_1': ['SHMT1', 'SHMT2',], },
    'epi_2':  {'class_1': ['TH', ], 'class_2': ['DDC', ], 'class_3': ['DBH', ], 'class_4': ['PNMT', ], },
    'Syn':    {'class_1': ['NRXN1', 'NRXN2', 'NRXN3',], },
}
total_synthesis_genes = [neurotransmitter_genes[neurotransmitter_name][class_id]
                         for neurotransmitter_name in neurotransmitter_genes
                         for class_id in neurotransmitter_genes[neurotransmitter_name]]
essential_synthesis_genes = [
    x for sublist in total_synthesis_genes for x in sublist]
essential_synthesis_genes = np.unique(essential_synthesis_genes)
expression_log_base = SETTINGS['analysis_global_settings']['log_expression_base']


from project_variables import MACAQUE_REGIONS, HUMAN_REGIONS, NAMED_NEUROTRANSMITTERS


def calculate_trimean(expression_array: np.ndarray, axis_: int):
    Q2 = np.nanquantile(expression_array, 0.5, axis=axis_)
    Q1 = np.nanquantile(expression_array, 0.25, axis=axis_)
    Q3 = np.nanquantile(expression_array, 0.75, axis=axis_)
    trimean_ = (0.5 * Q2) + (0.25 * (Q1 + Q3))
    if np.all(expression_array < 1e-8):
        trimean_ = 0.0
    return trimean_


def calculate_ligand_abundance(aggreagated_gene_expressions: pd.DataFrame) -> pd.DataFrame:
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
    final_ = pd.concat([final_, glu_final, NE_final, epi_final])
    return final_


def calculate_synthesis_gene_expressions(gene_expression_data: 'pd.DataFrame[float]',
                                         list_of_samples: 'np.ndarray[str]',) -> 'pd.DataFrame[float]':
    genes_to_extract = np.intersect1d(
        essential_synthesis_genes, gene_expression_data.index)
    essential_dataframe = gene_expression_data.loc[genes_to_extract][list_of_samples]
    essential_dataframe = essential_dataframe + expression_log_base
    mean_values = np.nanmean(essential_dataframe.to_numpy(), axis=1)
    median_values = np.nanmedian(essential_dataframe.to_numpy(), axis=1)
    trimean_values = calculate_trimean(essential_dataframe.to_numpy(), axis_=1)
    to_return = {
        'mean': mean_values, 'median': median_values, 'trimean': trimean_values,
    }
    to_return_df = pd.DataFrame.from_dict(to_return, orient='columns')
    to_return_df.index = genes_to_extract
    return to_return_df


def multiply_weight_with_adjacency(each_gs_adj_dict: 'dict[str, pd.DataFrame[float]]',
                                   gene_set_names_to_merge: list[str],
                                   merge_weights: 'pd.DataFrame[float]',
                                   normalize_weights=True):
    if normalize_weights:
        normalized_weights = merge_weights / merge_weights.max(axis=None)
    else:
        normalized_weights = merge_weights

    weighted_collection = {}
    REGIONS = [x.split('.')[0] for x in merge_weights.columns]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for neurotransmitter_name in NAMED_NEUROTRANSMITTERS:  # 'DA', 'Glu', ...
            try:
                this_NT_id = [x for x in gene_set_names_to_merge if x.__contains__(
                    neurotransmitter_name)][0]
                weights_per_region: 'np.ndarray' = normalized_weights.loc[neurotransmitter_name].to_numpy(
                )
                this_NT_multiplied = np.multiply(each_gs_adj_dict[this_NT_id].to_numpy(),
                                                 weights_per_region.reshape(-1, 1))
                NT_id_new = 'weighted_' + neurotransmitter_name
                weighted_collection[NT_id_new] = this_NT_multiplied
            except IndexError:  # 'Reactome_Gly'
                pass

    final_nt_collection_ = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for neurotransmitter_name in NAMED_NEUROTRANSMITTERS:  # 'DA', 'Glu', ...
            weighted_NT_id = 'weighted_' + neurotransmitter_name
            try:
                weighted_adj = pd.DataFrame(data=weighted_collection[weighted_NT_id],
                                            index=REGIONS,
                                            columns=REGIONS)
            except KeyError:
                continue
            final_nt_collection_[weighted_NT_id] = weighted_adj
    return final_nt_collection_


def weight_sum_adjacencies(each_gs_adj_dict: 'dict[str, pd.DataFrame[float]]',
                           gene_set_names_to_merge: list[str],
                           merge_weights: 'pd.DataFrame[float]',
                           organism: str,
                           normalize_weights=True):
    if normalize_weights:
        normalized_weights = merge_weights / merge_weights.max(axis=None)
    else:
        normalized_weights = merge_weights

    weighted_collection = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for neurotransmitter_name in NAMED_NEUROTRANSMITTERS:  # 'DA', 'Glu', ...
            try:
                this_NT_id = [x for x in gene_set_names_to_merge if x.__contains__(
                    neurotransmitter_name)][0]
                weights_per_region: 'np.ndarray' = normalized_weights.loc[neurotransmitter_name].to_numpy(
                )
                this_NT_multiplied = np.multiply(each_gs_adj_dict[this_NT_id].to_numpy(),
                                                 weights_per_region.reshape(-1, 1))
                weighted_collection.append(this_NT_multiplied)
            except IndexError: 
                pass
            except KeyError: 
                pass
    weighted_sum = np.array(weighted_collection).sum(axis=0)
    if (organism == 'human') or (organism == 'hsa') or (organism == 'HSA'):
        regions = HUMAN_REGIONS
    elif (organism == 'monkey') or (organism == 'macaque') or (organism == 'MMul') or (organism == 'mmul'):
        regions = MACAQUE_REGIONS
    else:
        raise ValueError(f'unknown organism {organism!r}')
    to_return = pd.DataFrame(data=weighted_sum, index=regions, columns=regions)
    return to_return


if __name__ == '__main__':
    pass