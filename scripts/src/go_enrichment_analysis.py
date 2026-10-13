"""PANTHER GO over-representation requests and result tables.

Used by: baseline_GO_enrichment.py, human_over_enrichment.py.
"""

import numpy as np
import pandas as pd
import json
import requests
import time
from fake_useragent import UserAgent

HSA_ncbi_tax_id = '9606'

def get_over_representation_result(gene_list_: np.ndarray[str], ontology_type_):
    '''
    curl -X POST
    "https://pantherdb.org/services/oai/pantherdb/enrich/overrep?
    geneInputList=Q96PB1&organism=9606&annotDataSet=GO%3A0008150&enrichmentTestType=FISHER&correction=FDR" 
    -H  "accept: application/json"
    '''
    panther_url = 'https://pantherdb.org/services/oai/pantherdb/enrich/overrep'
    ua = UserAgent()
    header = {'Accept': 'application/json', 'User-Agent': ua.random, }
    input_gene_list = ','.join(gene_list_)
    onto_keys_dict = {
        'molecular_function':'GO:0003674',
        'biological_process':'GO:0008150',
        'cellular_component':'GO:0005575',
    }
    query_dict = {'geneInputList': input_gene_list,
                  'organism': [HSA_ncbi_tax_id],
                  'annotDataSet': [onto_keys_dict[ontology_type_]],
                  'enrichmentTestType': ['FISHER',],
                  'correction': ['FDR',],
    }
    r = requests.post(panther_url, headers=header, params=query_dict)
    results = r.json()
    return results


def return_over_representation_url(gene_list_: np.ndarray[str], ontology_type_):
    '''
    curl -X POST
    "https://pantherdb.org/services/oai/pantherdb/enrich/overrep?
    geneInputList=Q96PB1&organism=9606&annotDataSet=GO%3A0008150&enrichmentTestType=FISHER&correction=FDR" 
    -H  "accept: application/json"
    '''
    panther_url = 'https://pantherdb.org/services/oai/pantherdb/enrich/overrep'
    input_gene_list = ','.join(gene_list_)
    onto_keys_dict = {
        'molecular_function':'GO:0003674',
        'biological_process':'GO:0008150',
        'cellular_component':'GO:0005575',
    }
    return_url =( panther_url + 
                 '?geneInputList=' + input_gene_list + 
                 '&organism=' + HSA_ncbi_tax_id + '&annotDataSet=' + onto_keys_dict[ontology_type_] +
                 '&enrichmentTestType=FISHER&correction=FDR'
                 )
    return return_url


def get_GO_participants(GO_term: str):
    '''
    curl -X 'GET' 
    'https://api.geneontology.org/api/bioentity/function/GO%3A0140009/genes
    ?taxon=NCBITaxon%3A9606&relationship_type=involved_in&start=0&rows=100'
    -H 'accept: application/json'
    '''
    GO_url = ('https://api.geneontology.org/api/bioentity/function/GO%3A' +
              GO_term.split(':')[1] +
              '/genes?taxon=NCBITaxon%3A' + HSA_ncbi_tax_id + '&start=0&rows=20000')
    ua = UserAgent()
    header = {'Accept': 'application/json', 'User-Agent': ua.random, }
    r = requests.get(GO_url, headers=header,)
    try:
        results = r.json()
    except requests.exceptions.JSONDecodeError:
        print(r)
        print(GO_url)
    return results




def construct_enrichment_dataframe(gene_list: np.ndarray[str], ontology_type):
    results_json_primal = get_over_representation_result(gene_list, ontology_type)
    # print(results_json_primal)
    result_essential = results_json_primal['results']['result'] # is a list-of-dict
    for_enrichment_dataframe = {}
    for e, each_result in enumerate(result_essential):
        try:
            enriched_term_id = each_result['term']['id']
            enriched_term_name = each_result['term']['label']
            total_gene_count = int(each_result['number_in_reference'])
            included_gene_count = int(each_result['number_in_list'])
            expected_ = float(each_result['expected'])
            fold_enrichment_ = float(each_result['fold_enrichment'])
            raw_pval_ = float(each_result['pValue'])
            fdr_corrected_pval_ = float(each_result['fdr'])
        except KeyError: # 'UNCLASSIFIED'
            pass
        for_enrichment_dataframe[e] = {
            'go_term_id': enriched_term_id ,
            'go_term_name': enriched_term_name ,
            'total_gene_count': total_gene_count ,
            'included_gene_count': included_gene_count ,
            'expected': expected_ ,
            'fold_enrichment': fold_enrichment_ ,
            'raw_pval': raw_pval_ ,
            'fdr_corrected_pval': fdr_corrected_pval_ ,
        }
    result_dataframe = pd.DataFrame.from_dict(for_enrichment_dataframe, orient='index')
    result_dataframe = result_dataframe[result_dataframe['fdr_corrected_pval'] < 0.1]
    return result_dataframe


def construct_ontology_participant_dict(GO_term_list_: list[str] | np.ndarray[str])-> 'dict[str, list[str] | np.ndarray[str]]':
    to_return_dict = {}
    for GO_id in GO_term_list_:
        associations_list_primal = get_GO_participants(GO_id)
        try:
            associations_essential = associations_list_primal['associations']
        except KeyError: # maybe timeout error
            print(GO_id)
            raise ValueError(associations_list_primal)
        related_genes_list = []
        for assoc_ in associations_essential:
            associated_gene_symbol_ = assoc_['subject']['label']
            related_genes_list.append(associated_gene_symbol_)
        to_return_dict[GO_id] = np.unique(related_genes_list)
        time.sleep(2)
    return to_return_dict
