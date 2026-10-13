"""Human prerequisite P2: GO over-representation of each gene set's full sender and receiver lists.

Input:  CURATED_GENE_SET.pickle; needs internet (PANTHER).
Output: human_analysis_results/go_enrichment/GO_baseline/<gene set>/cat{1,2}_baseline_<domain>.csv
Usage:  see HUMAN_PIPELINE.md, section 4.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys

import numpy as np
import pandas as pd

import yaml
from pathlib import Path

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f: 
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))


from src.quantile import load_gene_sets
from src import go_enrichment_analysis as gomodule
from src.project_variables import GO_TERMS

CATEGORIES = ['cat1', 'cat2']

GO_ENRICHMENT_DIR = os.path.join(PROJECT_ROOT,
    SETTINGS['analysis_results']['hsa_analysis_save_dir'],
    SETTINGS['human_over_enrichment']['report_save_dir'])
BASELINE_SAVE = os.path.join(GO_ENRICHMENT_DIR, SETTINGS['human_over_enrichment']['baseline_save_dir'])

GS_DICT = load_gene_sets()


def enrich_baseline_genes(gene_set_id: str, category_name: str, with_participants: bool = True) -> None:
    """GO enrichment (all 3 domains) on GS_DICT[gene_set_id][category_name]'s
    full gene list. An empty gene list writes the same one-row placeholder
    csv under every domain's own full-name filename, which is what
    human_over_enrichment.py's load_baseline_enrichment_dataframe looks up."""
    gene_symbols = np.asarray(GS_DICT[gene_set_id][category_name])
    save_dir = os.path.join(BASELINE_SAVE, gene_set_id)

    if gene_symbols.size == 0:
        print('there are no prioritized genes')
        placeholder = pd.DataFrame.from_dict({
            0: 'prioritized genes empty',
            1: 'conditions: ' + ';'.join([gene_set_id, category_name]),
        }, orient='index')
        for go_term in GO_TERMS:
            placeholder.to_csv(os.path.join(save_dir, f'{category_name}_baseline_{go_term}.csv'))
        return

    for go_term in GO_TERMS:
        enrichment_dataframe = gomodule.construct_enrichment_dataframe(gene_symbols, go_term)
        out_path = os.path.join(save_dir, f'{category_name}_baseline_{go_term}.csv')

        if enrichment_dataframe.empty:
            print(gene_set_id, category_name, go_term, 'enrichment result empty')
            pd.DataFrame.from_dict({
                0: 'GO enrichment result empty',
                1: 'conditions: ' + ';'.join([gene_set_id, category_name]),
                2: 'gene symbols: ',
                3: gene_symbols,
            }, orient='index').to_csv(out_path)
            continue

        print(gene_set_id, category_name, go_term)
        if with_participants:
            significant_terms = enrichment_dataframe['go_term_id'].to_numpy()
            enriched_participants = gomodule.construct_ontology_participant_dict(significant_terms)
            enrichment_dataframe = enrichment_dataframe.drop_duplicates('go_term_id')
            enrichment_dict = enrichment_dataframe.set_index('go_term_id').to_dict(orient='index')
            for term_id, row in enrichment_dict.items():
                row['common_genes'] = np.intersect1d(gene_symbols, enriched_participants[term_id])
                row['prioritized_genes'] = gene_symbols
        else:
            enrichment_dict = enrichment_dataframe.set_index('go_term_id').to_dict(orient='index')

        pd.DataFrame.from_dict(enrichment_dict, orient='index').to_csv(out_path)


def main() -> None:
    for gene_set_id in GS_DICT:
        os.makedirs(os.path.join(BASELINE_SAVE, gene_set_id), exist_ok=True)
    for gene_set_id in GS_DICT:
        for category_name in CATEGORIES:
            enrich_baseline_genes(gene_set_id, category_name, with_participants=False)


if __name__ == '__main__':
    main()
