"""Constants: regions, neurotransmitters, gene sets, cohort folders and GO domains.

Used by: most scripts.
"""

# gene set level settings

N_CONTROL_PANELS = 10
PANEL_SIZE = 585

# human settings

HUMAN_REGIONS = ['PFC', 'CC', 'HP', 'CN', 'NAc']

MACAQUE_REGIONS = ['dlPFC', 'dmPFC', 'vlPFC', 'vmPFC',
            'M1', 'STS', 'ACCg', 'V1',
            'CN', 'Pu', 'LGN',
            'VMH', 'AMY', 'CA3', 'DG',]

MACAQUE_RESTRICTED_REGIONS = ['dlPFC', 'vmPFC', 'ACCg',
            'CN', 'Pu', 'LGN',
            'VMH', 'AMY', 'CA3', 'DG',]

NAMED_NEUROTRANSMITTERS = ['DA', '5-HT', 'Glu', 'GABA', 'NE', 'ACh', 'Gly',
                           'epi', 'Syn']

HUMAN_DATASET_ABBR = {
    'BrainSeq': 'BS',
    'dbGaP_HBCC_MA': 'HBCC',
    'GSE53987': 'GSE53987',
    'GSE80655': 'GSE80655',
    'Array_Collection': 'AC',
    'Neuropathology_Consortium': 'NPC',
    'New_Collection': 'NC',
}

DISEASE_ABBR = {'Control': 'CTL', 'Schizophrenia': 'SCZ'}
DISEASES = sorted(DISEASE_ABBR.values())  # ['CTL', 'SCZ'], fixed canonical order

# canonical "DATASET#REGION" key -> folder name under human_expression/
HUMAN_DATASET_REGION_TO_RESIDUALS_FOLDER = {
    'BS#PFC': 'BrainSeq_DLPFC',
    'BS#HP': 'BrainSeq_HP',
    'BS#CN': 'BrainSeq_caudate',

    'HBCC#PFC': 'dbGaP_HBCC_MA_DLPFC',
    'HBCC#HP': 'dbGaP_HBCC_MA_HP',

    'GSE53987#PFC': 'GSE53987_DLPFC',
    'GSE53987#HP': 'GSE53987_HP',
    'GSE53987#CN': 'GSE53987_caudate',

    'GSE80655#PFC': 'GSE80655_DLPFC',
    'GSE80655#CC': 'GSE80655_cingulate',
    'GSE80655#NAc': 'GSE80655_NAc',

    'AC#PFC': 'Array_Collection_Prefrontal_Cortex',
    'AC#CC': 'Array_Collection_Cingulate_Cortex',
    'AC#HP': 'Array_Collection_Hippocampus',
    'AC#NAc': 'Array_Collection_Nucleus_Accumbens',
    'AC#CN': 'Array_Collection_Caudate_Nucleus',

    'NPC#PFC': 'Neuropathology_Consortium_Prefrontal_Cortex',
    'NPC#CC': 'Neuropathology_Consortium_Cingulate_Cortex',
    'NPC#HP': 'Neuropathology_Consortium_Hippocampus',

    'NC#PFC': 'New_Collection_Prefrontal_Cortex',
    'NC#HP': 'New_Collection_Hippocampus',
}

# exactly the gene sets in metadata/gene_data/CURATED_GENE_SET.pickle
GENE_SET_TO_USE = ['Curated_DA', 'Curated_5-HT', 'Curated_Glu', 'Curated_GABA',
                   'Curated_NE', 'Curated_ACh', 'Curated_Gly', 'Curated_epi', 'Curated_Syn',]
# {'DA': 'Curated_DA', '5-HT': 'Curated_5-HT', ...} -- bridges a neurotransmitter name
# (weighted_<NT> sheets, delta-report rows) to its gene-set id.
NT_TO_GENE_SET = {gene_set.split('_', 1)[1]: gene_set for gene_set in GENE_SET_TO_USE}

GO_TERMS = ['molecular_function', 'cellular_component', 'biological_process']
