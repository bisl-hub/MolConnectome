"""Macaque step 2: ligand abundance per neurotransmitter and region, from synthesis-enzyme expression.

Input:  macaque expression and sample-matching table.
Output: macaque_analysis_results/abundance.xlsx (sheets mean, median, trimean)
Usage:  see MACAQUE_PIPELINE.md, section 4.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import yaml

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

from src.calculate_abundance import (
    calculate_synthesis_gene_expressions,
    calculate_ligand_abundance,
)
import src.macaque_expression_load as mc
from src.project_variables import MACAQUE_REGIONS, NAMED_NEUROTRANSMITTERS


def compute_macaque_abundance():
    macaque_cpms = mc.load_macaque_log2_cpm()
    monkey_expressions = {x: macaque_cpms[x] for x in MACAQUE_REGIONS}  # ! reorder regions
    samples_dict = {
        x: np.array(monkey_expressions[x].columns) for x in monkey_expressions
    }
    mean_collection_ = []
    median_collection_ = []
    trimean_collection_ = []
    for region_name in samples_dict:
        values_ = calculate_synthesis_gene_expressions(monkey_expressions[region_name],
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

    mean_abundance = calculate_ligand_abundance(mean_dataframe)
    median_abundance = calculate_ligand_abundance(median_dataframe)
    trimean_abundance = calculate_ligand_abundance(trimean_dataframe)

    mean_abundance = mean_abundance.loc[NAMED_NEUROTRANSMITTERS, MACAQUE_REGIONS]
    median_abundance = median_abundance.loc[NAMED_NEUROTRANSMITTERS, MACAQUE_REGIONS]
    trimean_abundance = trimean_abundance.loc[NAMED_NEUROTRANSMITTERS, MACAQUE_REGIONS]
    return mean_abundance, median_abundance, trimean_abundance


def main():
    mean_abundance, median_abundance, trimean_abundance = compute_macaque_abundance()

    output_file_path = os.path.join(PROJECT_ROOT,
        SETTINGS['analysis_results']['mq_analysis_save_dir'],
        'abundance.xlsx')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with pd.ExcelWriter(output_file_path) as writer:
        mean_abundance.to_excel(writer, sheet_name='mean')
        median_abundance.to_excel(writer, sheet_name='median')
        trimean_abundance.to_excel(writer, sheet_name='trimean')

    return mean_abundance, median_abundance, trimean_abundance


if __name__ == '__main__':
    main()

