"""Human stage 1a: ligand abundance per diagnosis, for each cohort-region and merged region.

Input:  human expression and sample-matching table; an optional label-shuffle seed.
Output: <output_dir>/human_abundance_control.xlsx and human_abundance_schizo.xlsx
Usage:  see HUMAN_PIPELINE.md, section 4 (normally run by human_label_shuffle_wrapper.py).
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import pandas as pd
import sys
import numpy as np
import yaml
from pathlib import Path

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))
DEFAULT_ABUNDANCE_SAVE_PATH = (PROJECT_ROOT / SETTINGS['analysis_results']['hsa_analysis_save_dir']
                               / SETTINGS['human_label_shuffle_wrapper']['unshuffled_dir_name'])


import src.calculate_abundance as abund
from src.project_variables import NAMED_NEUROTRANSMITTERS
from src.human_expression_load import (load_human_expression_by_region_and_disease,
                                        load_human_expression_by_dataset_and_disease,
                                        resolve_sample_matching)


def calculate_human_abundance(seperated_expression_dict: dict[str, pd.DataFrame]):
    """input shape: 
        seperated_expression_dict.keys(): dataset_name | region_name
        seperated_expression_dict[key].shape: (gene_number, sample_number)
    """
    human_expressions = seperated_expression_dict
    samples_dict = {
        x: np.array(human_expressions[x].columns) for x in human_expressions
    }
    mean_collection_ = []
    median_collection_ = []
    trimean_collection_ = []
    for region_name in samples_dict:
        values_ = abund.calculate_synthesis_gene_expressions(human_expressions[region_name],
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

    mean_abundance = abund.calculate_ligand_abundance(mean_dataframe)
    median_abundance = abund.calculate_ligand_abundance(median_dataframe)
    trimean_abundance = abund.calculate_ligand_abundance(trimean_dataframe)

    mean_abundance = mean_abundance.loc[NAMED_NEUROTRANSMITTERS, list(seperated_expression_dict.keys())]
    median_abundance = median_abundance.loc[NAMED_NEUROTRANSMITTERS, list(seperated_expression_dict.keys())]
    trimean_abundance = trimean_abundance.loc[NAMED_NEUROTRANSMITTERS, list(seperated_expression_dict.keys())]
    return mean_abundance, median_abundance, trimean_abundance


def _abundance_output_paths(output_dir: str) -> tuple[str, str]:
    """(control_path, schizo_path), always directly under `output_dir` --
    abundance values don't depend on background-gene selection or
    p-value-method, so unlike human_oneshot.py/human_weighted_connectivity.py
    there's no per-background subfolder here. Callers (e.g.
    human_label_shuffle_wrapper.py) give each run -- unshuffled or a given
    label_shuffle_seed replicate -- its own `output_dir`, so no
    seed-dependent naming is needed here either."""
    return (
        os.path.join(output_dir, 'human_abundance_control.xlsx'),
        os.path.join(output_dir, 'human_abundance_schizo.xlsx'),
    )


def main(output_dir: str = str(DEFAULT_ABUNDANCE_SAVE_PATH),
         label_shuffle_seed: int = None):
    # same helper as human_oneshot.py's _init_worker, so a given
    # label_shuffle_seed gives both the identical disease-label permutation.
    sample_matching = resolve_sample_matching(label_shuffle_seed)

    # load expression and re-seperate them by disease to make single level dict
    seperated_expression_dict = load_human_expression_by_dataset_and_disease(sample_matching=sample_matching)
    merged_expression_dict = load_human_expression_by_region_and_disease(sample_matching=sample_matching)
    total_expression_dict = {**seperated_expression_dict, **merged_expression_dict}
    controls = {key_: total_expression_dict[key_]['CTL'] for key_ in total_expression_dict}
    schizos = {key_: total_expression_dict[key_]['SCZ'] for key_ in total_expression_dict}

    # now use them to make abundance
    mean_abundance_control, median_abundance_control, trimean_abundance_control = \
        calculate_human_abundance(controls)

    mean_abundance_schizo, median_abundance_schizo, trimean_abundance_schizo = \
        calculate_human_abundance(schizos)

    control_path, schizo_path = _abundance_output_paths(output_dir)

    os.makedirs(os.path.dirname(control_path), exist_ok=True)
    with pd.ExcelWriter(control_path) as writer:
        mean_abundance_control.to_excel(writer, sheet_name='mean')
        median_abundance_control.to_excel(writer, sheet_name='median')
        trimean_abundance_control.to_excel(writer, sheet_name='trimean')

    os.makedirs(os.path.dirname(schizo_path), exist_ok=True)
    with pd.ExcelWriter(schizo_path) as writer:
        mean_abundance_schizo.to_excel(writer, sheet_name='mean')
        median_abundance_schizo.to_excel(writer, sheet_name='median')
        trimean_abundance_schizo.to_excel(writer, sheet_name='trimean')

    print(f'label_shuffle_seed={label_shuffle_seed}: wrote {control_path} and {schizo_path}')
    return (
        (mean_abundance_control, median_abundance_control, trimean_abundance_control),
        (mean_abundance_schizo, median_abundance_schizo, trimean_abundance_schizo),
    )


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ABUNDANCE_SAVE_PATH,
        int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'None' else None,
    )
