# Macaque analysis pipeline

Three scripts, run in order, turn the macaque expression residuals (15 regions) into a directed,
neurotransmitter-resolved interregional connectivity estimate:

| Step | Script (`scripts/macaque_analysis/`) | Produces                                                                                                                                     |
| ---- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `macaque_oneshot.py`                 | `directed_signaling_strength.csv`: per gene set × directed region pair, a z-score of co-expression strength against background-gene nulls |
| 2    | `macaque_abundance.py`               | `abundance.xlsx`: per neurotransmitter × region, a ligand-abundance estimate from synthesis-enzyme expression                             |
| 3    | `macaque_weighted_connectivity.py`   | `weighted_connectivity.xlsx`: step 1's z matrices weighted by step 2's sender abundance and summed                                         |

Step 3 reads step 1's output, so step 1 must finish first. Step 2 is independent: step 3 recomputes
abundance in memory and does not read `abundance.xlsx`.

Nothing else has to run first. All gene lists and background panels are already in
`metadata/gene_data/`.

---

## 1. What each step computes

### Step 1: directed signaling strength (`macaque_oneshot.py`)

For each of the 105 region pairs (A, B):

1. **Match monkeys.** Use the monkeys with a sample in both A and B, taken from
   `metadata/macaque_sample_matching_table.csv`.
2. **Correlate.** Compute Pearson r between every gene in A and every gene in B across those
   monkeys, using the 585 `CURATED_GENES`. Everything stays in memory; no per-pair files are
   written. A p-value is also computed, but no step uses it (see section 8).
3. **Real value.** For each of the 9 curated gene sets and each direction, `real_value` is the sum
   of every positive r over (sender genes `cat2` in A) × (receiver genes `cat1` in B). There is no
   p-value filter.
4. **Null.** Each of the 10 background panels (585 random non-NT genes) is correlated the same
   way. 1000 random gene subsets of the same size are drawn from genes that have data for this
   pair, and the same sum is computed for each.
5. **z per panel.** The rank of `real_value` in the null (mid-rank ties) gives
   (rank + 0.5) / 1001, which gives z. That z is bounded at ±3.291.
6. **Across panels.** `mean_z` is the mean of the 10 panel z-scores. `z_std`, `z_min` and `z_max`
   are also reported.

Pearson r is symmetric, so A→B and B→A come from the same correlation matrix with the sender and
receiver roles swapped.

**Gene sets** (`CURATED_GENE_SET.pickle`, all inside `CURATED_GENES.npy`):

| Gene set     | sender genes (`n_cat2`) | receiver genes (`n_cat1`) |
| ------------ | ------------------------- | --------------------------- |
| Curated_DA   | 12                        | 7                           |
| Curated_5-HT | 12                        | 17                          |
| Curated_Glu  | 4                         | 24                          |
| Curated_GABA | 3                         | 14                          |
| Curated_NE   | 9                         | 9                           |
| Curated_ACh  | 10                        | 21                          |
| Curated_Gly  | 10                        | 5                           |
| Curated_epi  | 9                         | 9                           |
| Curated_Syn  | 3                         | 5                           |

### Step 2: abundance (`macaque_abundance.py` → `src/calculate_abundance.py`)

For each region, each synthesis-enzyme gene gets one value: the mean, median or trimean across
samples of (residual + `log_expression_base`). The shift is 2.0, set in `settings.yaml` and shared
with the human pipeline.

Each neurotransmitter's enzymes are grouped into classes, for example DA = {TH} × {DDC}:

- A class's value is the average of its genes. A gene missing from a region counts as 0.
- The abundance is the geometric mean across classes.
- Glu averages two pathways, and NE and epi use the full TH→DDC→DBH(→PNMT) chain.

The output has 9 neurotransmitters, one per gene set: DA, 5-HT, Glu, GABA, NE, ACh, Gly, epi and
Syn (`NAMED_NEUROTRANSMITTERS` in `src/project_variables.py`).

### Step 3: weighted connectivity (`macaque_weighted_connectivity.py`)

1. For each gene set, `mean_z` is pivoted into a 15 × 15 sender × receiver matrix. It is min-max
   normalised to [0, 1] with the theoretical bound: (z + 3.291) / (2 · 3.291).
2. Each row (sender) is multiplied by that sender region's `mean` abundance for the gene set's
   neurotransmitter. With `normalize_weights=True`, the abundance table is first divided by its
   global maximum.
3. `total` is the sum of the 9 weighted matrices, one per gene set.

---

## 2. Requirements

**Packages.** Tested with Python 3.13.9, numpy 2.3.5, pandas 2.3.3, scipy 1.17.0, PyYAML 6.0.3
and openpyxl 3.1.5. openpyxl is needed for the `.xlsx` output.

```
pip install numpy pandas scipy pyyaml openpyxl
```

**`settings.yaml`.** Before the first run, replace the placeholder `<YOUR_PROJECT_DIR>` in
`project_directory` with the absolute path of your copy of the repo.

- Each script finds `settings.yaml` relative to its own location.
- It then builds every input path, output path and `sys.path` entry from `project_directory`.
- The unreplaced placeholder or a stale value makes imports fail, or makes them point at another
  copy of the project.

The macaque pipeline doesn't use `python_interpreter` (placeholder `<YOUR_PYTHON_INTERPRETER>`);
only the human label-shuffle wrapper reads it. Every other key the macaque pipeline reads already
has a working default:

- `macaque_oneshot.*`: the seeds and the 1000 null samples
- `gene_set_settings.*`: the default background `exclude_NT/BACKGROUND_GENES_00.pickle`
- `analysis_results.mq_analysis_save_dir`: where every output goes, including `abundance.xlsx`
- `analysis_global_settings.log_expression_base`: the abundance shift (2.0)
- `macaque_all_gene_correlation.all_gene_result_save_dir`: the optional all-gene output folder

**Single-threaded BLAS.** Every entry script sets `OPENBLAS_NUM_THREADS=1` before it first imports
numpy, so parallelism comes only from worker processes. This line has to run before numpy is
imported to take effect.

---

## 3. Inputs

```
settings.yaml
metadata/
├── macaque_sample_matching_table.csv          monkey × region → sample id (blank = no sample)
└── gene_data/
    ├── CURATED_GENES.npy                      585 NT-related genes (the correlation universe)
    ├── CURATED_GENE_SET.pickle                9 gene sets {gene_set: {'cat1': receivers, 'cat2': senders}}
    ├── exclude_NT/BACKGROUND_GENES_00..03.pickle   4 background files; each = 10 panels × 585 genes, no NT genes  (default: 00)
    └── include_NT/BACKGROUND_GENES_00..03.pickle   same, NT genes allowed
macaque_expression/
└── <region>/expression.csv                    genes × samples, one folder per region (15)
```

The regions, in the output order of every matrix (`src/project_variables.py: MACAQUE_REGIONS`), are
dlPFC, dmPFC, vlPFC, vmPFC, M1, STS, ACCg, V1, CN, Pu, LGN, VMH, AMY, CA3 and DG.

---

## 4. Running the pipeline

The commands below pass every argument explicitly, using the values the reference run used. These
are also the script defaults.

### PowerShell (Windows)

```powershell
$PY = "<YOUR_PYTHON_INTERPRETER>"                   # your interpreter
$P  = "<YOUR_PROJECT_DIR>"                          # = project_directory in settings.yaml
$BG = "$P\metadata\gene_data\exclude_NT\BACKGROUND_GENES_00.pickle"
$OUT = "$P\macaque_analysis_results"

# Step 1: directed signaling strength (~2 min with 14 workers)
& $PY "$P\scripts\macaque_analysis\macaque_oneshot.py" $BG $OUT "transformation" "None" "14" "mid-rank"

# Step 2: abundance (seconds)
& $PY "$P\scripts\macaque_analysis\macaque_abundance.py"

# Step 3: weighted connectivity (seconds). Argument 5 = normalize_weights (default True); see section 7.
& $PY "$P\scripts\macaque_analysis\macaque_weighted_connectivity.py" $BG $OUT "transformation" "None" "True"
```

### bash (Linux server)

```bash
PY=python; P=/path/to/project; BG=$P/metadata/gene_data/exclude_NT/BACKGROUND_GENES_00.pickle; OUT=$P/macaque_analysis_results
$PY $P/scripts/macaque_analysis/macaque_oneshot.py "$BG" "$OUT" transformation None 14 mid-rank
$PY $P/scripts/macaque_analysis/macaque_abundance.py
$PY $P/scripts/macaque_analysis/macaque_weighted_connectivity.py "$BG" "$OUT" transformation None True
```

### Arguments

The arguments are positional. To set argument *n* you must also give arguments 1 to *n*−1.
`None` is the literal string `None`, and booleans are true only for the exact string `True`.

**`macaque_oneshot.py`**

| # | Name                  | Default                                                      | Options / meaning                                                                                                                                    |
| - | --------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `background_source` | `metadata/gene_data/exclude_NT/BACKGROUND_GENES_00.pickle` | any `BACKGROUND_GENES_*.pickle`                                                                                                                     |
| 2 | `output_dir`        | `macaque_analysis_results`                                 | results go to `<output_dir>/<bg file stem, lowercased>_<p method>/`. Set your own `output_dir` for each configuration (section 7)                 |
| 3 | `p_value_method`    | `transformation` (settings.yaml)                           | `transformation` or `permutation`. It currently only names the output folder; see section 8                                                      |
| 4 | `run_seed`          | `None`                                                     | `None` = baseline run → `directed_signaling_strength.csv`; an integer → `run_XXXXXX.csv.gz`. It currently only names the file; see section 8 |
| 5 | `n_workers`         | CPU count − 2                                               | number of worker processes (`None` = default)                                                                                                      |
| 6 | `tie_rank_handler`  | `mid-rank`                                                 | `strict`, `inclusive`, `mid-rank`, `discard`                                                                                                 |

**`macaque_abundance.py`** takes no arguments. It writes `abundance.xlsx` into
`<project_directory>/<mq_analysis_save_dir>/` (default `macaque_analysis_results/`) and creates the
folder if needed.

**`macaque_weighted_connectivity.py`**

| #    | Name                                                                    | Default      | Meaning                                                                                                                                            |
| ---- | ----------------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1–4 | `background_source`, `output_dir`, `p_value_method`, `run_seed` | as in step 1 | must match step 1, or it reads or writes the wrong folder                                                                                          |
| 5    | `normalize_weights`                                                   | `True`     | `True` = divide the abundance table by its global maximum before weighting; `False` = weight by raw abundance. The reference run used `True` |

---

## 5. Outputs

```
macaque_analysis_results/
├── abundance.xlsx                                   step 2
└── background_genes_00_transformation/              <bg stem>_<p method>
    ├── directed_signaling_strength.csv              step 1
    └── weighted_connectivity.xlsx                   step 3
```

### `directed_signaling_strength.csv` (1 row per gene set × directed pair)

| Column                                         | Meaning                                                                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `gene_set`, `source`, `neurotransmitter` | e.g. `Curated_DA`, `Curated`, `DA`                                                                             |
| `sender_region`, `receiver_region`         | direction of the edge (sender expresses `cat2`, receiver expresses `cat1`)                                       |
| `n_cat1`, `n_cat2`                         | gene-set sizes                                                                                                      |
| `n_monkeys`                                  | monkeys matched for this region pair                                                                                |
| `n_qualifying_real_pairs`                    | gene pairs with r > 0, i.e. the pairs summed into `real_value`                                                     |
| `real_value`                                 | sum of the positive r: the raw, unbounded strength                                                                  |
| `mean_z`                                     | **main statistic**: mean over the 10 background panels of the z against that panel's null; bounded at ±3.291 |
| `z_std`, `z_min`, `z_max`                | other summaries over the same 10 panel z-scores                                                                     |

### `abundance.xlsx`

It has three sheets, `mean`, `median` and `trimean` (how samples are summarised per region). Each
is 9 neurotransmitters (rows) × 15 regions (columns). Step 3 uses only `mean`.

### `weighted_connectivity.xlsx`

Every sheet is a 15 × 15 matrix with rows = sender and columns = receiver. The diagonal is NaN
because there are no self-pairs.

| Sheet                                                           | Content                                                               |
| --------------------------------------------------------------- | --------------------------------------------------------------------- |
| `total`                                                       | Σ over the 9 gene sets of (normalised `mean_z` × sender abundance) |
| `weighted_<NT>` (DA, 5-HT, Glu, GABA, NE, ACh, Gly, epi, Syn) | the per-neurotransmitter terms of that sum                            |

---

## 6. Reference run and expected results

The reference run used the toy data (TOY_DATA.md: 15 monkeys, 220 samples) on 2026-10-07. It
ran on Windows 11 with 16 cores, using the exact commands in section 4.

| Step                                  | Wall time | Result                                    |
| ------------------------------------- | --------- | ----------------------------------------- |
| 1 `macaque_oneshot.py`               | 123 s     | 105/105 pairs, 1890 rows × 14 columns    |
| 2 `macaque_abundance.py`             | 1.6 s     | `abundance.xlsx`, 3 sheets × (9 × 15) |
| 3 `macaque_weighted_connectivity.py` | 2.8 s     | `total` + 9 `weighted_<NT>` sheets    |

Use these checks to confirm a run is healthy:

**`directed_signaling_strength.csv`**

- 1890 rows: 105 pairs × 2 directions × 9 gene sets, giving 210 directed pairs.
- `n_monkeys` is between 11 and 15.
- No NaN in any column.
- `mean_z` lies within ±3.290808. The bound is reached: 31 rows sit exactly on it, meaning all 10
  panels hit the floor or the ceiling, so checks must not use a tighter cut-off.
- In the reference run, 291 rows have `mean_z` > 1.645 and 225 have `mean_z` < −1.645.

**`abundance.xlsx`:** no NaN; values from 0 to 10.42 (the maximum is GABA in Pu).

**`weighted_connectivity.xlsx`**

- Every sheet is 15 × 15, with NaN only on the diagonal.
- In the `total` sheet, the off-diagonal values range from 0.856 to 3.126.

**Reproducibility:** step 1 is deterministic, because every seed is fixed in `settings.yaml`.
Compare outputs with the numbers above by value, not by file hash. CSV line endings depend on the
operating system, and xlsx files embed timestamps.

`real_value` depends on the BLAS thread count in its last bits, because the thread count changes
the order of the matrix-product sums. With multi-threaded BLAS, 36 of 1890 `real_value` cells
differed by at most 3.6e-15. Every z column and count was identical. Compare `real_value` with a
tolerance rather than by hash if your BLAS setup differs.

Quick check:

```python
import pandas as pd, scipy.stats
d = pd.read_csv('macaque_analysis_results/background_genes_00_transformation/directed_signaling_strength.csv')
z_bound = abs(scipy.stats.norm.ppf(0.5 / 1001))   # 3.290808; 31 rows sit exactly on it
assert len(d) == 1890 and d.isna().sum().sum() == 0 and d['mean_z'].abs().max() <= z_bound + 1e-9
print(d['n_monkeys'].agg(['min', 'max']).tolist())
```

**Exact zeros in `abundance.xlsx` are expected.** In the reference run these are:

- DA, 5-HT, NE and epi in dmPFC, vlPFC, M1, STS and V1
- ACh everywhere except CN, Pu and VMH

The cause is that a required synthesis gene was filtered out of that region's expression file:
`DDC` (and `TH` in V1), or `CHAT`. A missing gene counts as 0, and the geometric mean of the
classes then becomes 0. Those senders therefore contribute 0 to that neurotransmitter's
`weighted_<NT>` row.

---

## 7. Usage rules and notes

- **`normalize_weights` is argument 5 of step 3 and defaults to `True`.** Append `False` to weight
  by raw abundance instead. The reference run used `True`.
- **Set your own `output_dir` (argument 2) for each configuration when you start a run.** The
  results subfolder is named only from the background file's stem. For example,
  `include_NT/BACKGROUND_GENES_00.pickle` and `exclude_NT/BACKGROUND_GENES_00.pickle` would both
  land in `background_genes_00_transformation/` under the same `output_dir`. For example, use
  `macaque_analysis_results/curated_gene_include_nt` and `.../curated_gene_exclude_nt`.
- **`abundance.xlsx` follows `mq_analysis_save_dir`, not `output_dir`.** It does not depend on the
  background panel, and step 3 never reads it.
- **Step 3 must get the same arguments 1–4 as step 1.** It finds `directed_signaling_strength.csv`
  by rebuilding step 1's path from them.
- **Don't change `quantile_bg_n_null_samples` between steps 1 and 3.** Step 3 recomputes the
  normalisation bound `Z_BOUND` from the current setting, not from the run.
- **`N_CONTROL_PANELS = 10` and `PANEL_SIZE = 585` in `src/project_variables.py` must match the
  data**: the number of panels in the background pickle and the length of `CURATED_GENES`. Nothing
  checks this.
- **PowerShell 5.1 strips embedded double quotes** from arguments passed to `python.exe`. Quote
  paths that contain spaces as shown above, and don't nest `"` inside `python -c "..."`.
- **These are toy-data results.** The reference numbers come from 15 of 36 monkeys. They are for
  checking that the code runs, not for interpretation.

---

## 8. Variants

- **Another background panel:** change argument 1 in steps 1 and 3, and give each configuration
  its own `output_dir` (see section 7). For example, all four `exclude_NT`
  files into `macaque_analysis_results/curated_gene_exclude_nt/`.
- **`run_seed` sweeps:** an integer `run_seed` writes `run_XXXXXX.csv.gz` next to the baseline
  file, and step 3 can read it if it gets the same seed. Two things to know:

  - **Every seed currently gives the same table.** The seed only feeds the permutation p-values,
    and nothing uses p (see the next bullet). The null draws are seeded per pair, gene set,
    direction and panel, independently of `run_seed`.
  - Step 3 writes `weighted_connectivity.xlsx` to the same folder for every seed, so each seeded
    run overwrites the baseline's step-3 output. Copy it elsewhere first.
- **p-values are currently inert.** `real_value` sums every positive r with no p-value filter, so:

  - `transformation` and `permutation` give identical results; `permutation` is only much
    slower.
  - `p_value_method` affects only the output folder name (`<bg stem>_<p method>/`).
  - `permutation` (a one-sided empirical p from `correlation_n_permutations` = 1000 sample
    shuffles, in chunks of `correlation_permutation_chunk_size`) is kept so a p-based rule can be
    added back later.
- **Tie options:** `tie_rank_handler` accepts `strict`, `inclusive`, `mid-rank` or `discard`.
- **`macaque_all_gene_correlation.py`** (optional, not part of the three steps):

  - It sums positive, negative and significant cross-region correlations over all genes shared by
    the 15 regions.
  - It writes `macaque_analysis_results/whole_transcriptome/regional_correlation_matrix_*.csv`.
    The folder name comes from `macaque_all_gene_correlation.all_gene_result_save_dir`.
  - It takes no arguments and runs with 2 worker processes. More than 2 runs out of RAM on a
    48 GB machine.
