# Human analysis pipeline

The human pipeline turns the postmortem expression residuals (5 regions, control vs
schizophrenia) into a directed, neurotransmitter-resolved interregional connectivity estimate for
each diagnosis. It then tests every SCZ − CTL difference against a disease-label-shuffle null and
traces the significant connections back to driver genes and GO terms.

| Stage | Script                                                                                                                                                     | Produces (under `human_analysis_results/`)                                               |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| P1    | `scripts/prerequisite/human_group_abundance_compare.py`                                                                                                  | `group_abundance_comparison/group_abundance_stats.csv` + boxplot                        |
| P2    | `scripts/prerequisite/baseline_GO_enrichment.py`                                                                                                         | `go_enrichment/GO_baseline/<gene set>/cat{1,2}_baseline_<domain>.csv` (54 files)        |
| 1     | `scripts/human_analysis/human_label_shuffle_wrapper.py`, which runs `human_abundance.py` → `human_oneshot.py` → `human_weighted_connectivity.py` | `unshuffled/` (the real run) and `<JOB>/run_NNNNNN/` (one per label shuffle)          |
| 2     | `scripts/human_analysis/downstream_analysis/human_corr_landscape.py`                                                                                     | `corr_landscape/`                                                                       |
| 3     | `scripts/human_analysis/human_delta_quantile_report.py`                                                                                                  | `label_shuffle_summary/<JOB>/delta_quantile_report.csv`, `delta_quantile_flagged.csv` |
| 4     | `scripts/human_analysis/downstream_analysis/human_corr_contrib.py`                                                                                       | `label_shuffle_summary/<JOB>/corr_contrib/`                                             |
| 5     | `scripts/human_analysis/downstream_analysis/human_prioritize_drivers.py`                                                                                 | `label_shuffle_summary/<JOB>/driver_gene_prioritization.csv`                            |
| 6     | `scripts/human_analysis/downstream_analysis/human_over_enrichment.py`                                                                                    | `go_enrichment/<JOB>/<NT>_<sender>_<receiver>.xlsx`                                     |

`<JOB>` is `human_label_shuffle_wrapper.initial_job_name` in `settings.yaml`
(`10_shuffle_20261006` in the reference run).

### What depends on what

```
wave 1 (parallel):  [1 wrapper]   [P1 group abundance]    [2 corr landscape]
                         └──────────┬───────┘                     │
wave 2:                  [3 delta-quantile report]                │
                         ┌──────────┴──────────┐                  │
wave 3 (parallel):  [4 corr contrib]   [5 prioritize drivers] ◄───┘
                                               │
wave 4:          [P2 GO baseline, once] ──► [6 over-enrichment]
```

- Stages 1, P1 and 2 need only the expression data, so they can run at the same time.
- Stage 3 needs stage 1 (the unshuffled run and every replicate) and P1's CSV.
- Stages 4 and 5 need stage 3's flagged file. Stage 5 also needs stage 2's
  `gene_contribution_diff_full.csv`.
- Stage 6 needs stage 5's CSV, P2's baseline files and internet access.
- P2 depends only on the gene sets, so run it once per gene-set version. It needs internet access
  (PANTHER) and takes a while because of rate limiting, which is why it is a prerequisite.
- The repository ships no results, so a fresh clone has no `go_enrichment/GO_baseline/`. Run P2
  once before stage 6. It can run at the same time as waves 1–3.

---

## 1. What each step computes

### Stage 1a: abundance (`human_abundance.py` → `src/calculate_abundance.py`)

This uses the same formula as the macaque pipeline (see `MACAQUE_PIPELINE.md`, step 2):

- Each synthesis gene's value is the mean, median or trimean across samples of (residual +
  `log_expression_base`, 2.0 in `settings.yaml`).
- The output has 9 neurotransmitters, one per gene set: DA, 5-HT, Glu, GABA, NE, ACh, Gly, epi and
  Syn.
- Classes are averaged within, with missing genes counted as 0, and combined by geometric mean.

It is computed separately for each diagnosis and for two kinds of column:

- **13 columns.** There are 8 `DATASET#REGION` columns (each cohort on its own, e.g. `BS#PFC`) and
  5 merged-region columns (`PFC`, `CC`, `HP`, `CN`, `NAc`, which pool every cohort). Stage 1c uses
  only the 5 merged-region columns.
- **Label shuffle.** A `label_shuffle_seed` reshuffles the diagnoses first, using the same call
  as stage 1b (see below).

### Stage 1b: directed signaling strength (`human_oneshot.py`)

This is the macaque step 1 algorithm (`MACAQUE_PIPELINE.md`, step 1), run **once per diagnosis**:

1. **Pairs.** For each of the 10 region pairs and each diagnosis (20 work items), use the donors
   of that diagnosis who have a sample in both regions. Donors are matched by `donor_id` across
   cohorts.
2. **Correlation and null.** Compute Pearson r over the 585 `CURATED_GENES`. A p-value is also
   computed, but no stage-1 step uses it (see section 7).
3. **Real value.** For each of the 9 curated gene sets and each direction, `real_value` is the
   sum of every positive r over sender genes (`cat2`) × receiver genes (`cat1`), with no p-value
   filter. The 10 background panels each give a null of 1000 resamples, restricted to genes with
   data for the pair.
4. **z.** The mid-rank z is bounded at ±3.291. `mean_z` is the mean over the 10 panels.

**Label shuffles.** `label_shuffle_seed = None` is the real run. An integer seed permutes the
`disease` column of the matching table with `np.random.default_rng(seed)`
(`src/human_expression_load.shuffle_disease_labels`, `stratified=True`):

- The permutation happens within each (dataset, region-coverage pattern) block.
- So every region's, region pair's and cohort's CTL/SCZ counts stay exactly as in the real data.
  Only who is labelled SCZ changes.
- Stage 1a uses the same seed and the same call, so a replicate's abundance and correlations
  come from the same shuffled labels.

### Stage 1c: weighted connectivity (`human_weighted_connectivity.py`)

This is computed for each diagnosis:

1. **Normalise.** For each gene set, `mean_z` is pivoted into a 5 × 5 sender × receiver matrix and
   normalised to [0, 1] with the bound: (z + 3.291) / (2 · 3.291).
2. **Weight.** Each row (sender) is multiplied by that diagnosis's `mean` abundance of the gene
   set's neurotransmitter in the sender region. These matrices are the `weighted_<NT>` sheets.
   - With `normalize_weights=False` (the reference run), the raw abundance is used.
   - With `True`, the abundance table is first divided by its own global maximum. That maximum
     is taken separately for CTL and SCZ.
3. **`total` sheet.** It is the sum of the weighted matrices and always uses max-normalised
   abundance, whatever `normalize_weights` is.

The bound 3.291 (`Z_BOUND` = |Φ⁻¹(0.5/1001)|) is computed from the **current**
`human_oneshot.quantile_bg_n_null_samples` each time a script runs. It is not stored with the
signaling files. Stage 1c and stage 3 both use it, so the setting is shared across analyses; see
section 7.

### Stage 1 driver: `human_label_shuffle_wrapper.py`

1. **Phase 0, the real run.** It runs abundance → oneshot → weighted connectivity once, with
   `label_shuffle_seed=None`, into `unshuffled/`. `human_oneshot.py` gets no `n_workers`, so it
   uses CPU count − 2 workers.
2. **Replicates.** For seeds `seed_start … seed_start + n − 1`, it runs all abundance jobs, then
   all oneshot jobs, then all weighted-connectivity jobs. Each phase is a pool of up to
   `max_concurrency` subprocesses. Each oneshot subprocess uses `n_workers=1`, so the parallelism
   is across replicates rather than within one.
   - The pools check for finished processes every `process_wait_interval` seconds
     (`settings.yaml`, 20).
   - The oneshot phase, the longest, checks every 3× that interval.
   - Replicate oneshot calls always use the default options (arguments 6–8 aren't passed).
3. **Outputs.** Replicate *n* writes to `<output_dir>/<JOB>/run_<n:06d>/`. The log is appended to
   `<JOB>/label_shuffle_wrapper_log.txt`.

### Stage 3: delta-quantile report (`human_delta_quantile_report.py`)

This covers each of the 9 neurotransmitters × 20 directed region pairs (180 rows).

**Statistics.**

- `delta_weighted` is SCZ − CTL of the `weighted_<NT>` cell.
- `delta_non_weighted` is SCZ − CTL of the normalised `mean_z`.
- The null is the same delta in each shuffled replicate.
- `quantile = (mid-rank + 0.5) / (n_replicates + 1)`.

**Flagged rows.** `delta_quantile_flagged.csv` keeps the rows with `quantile_weighted` < 0.05
(**hypoactive** in SCZ) or > 0.95 (**hyperactive**). Both thresholds are in `settings.yaml`
(`human_delta_quantile_report.hypo/hyperactive_quantile_threshold`). It also joins the sender region's
group-abundance stats from P1 on (sender_region, neurotransmitter). The joined columns are the
mean difference, p, Hedges' g, `sender_abundance_direction` and
`sender_abundance_significance`.

**p- and q-values.** These are two-sided p-values with BH q-values, computed for each metric. A
row whose real SCZ and CTL values are both 0 is left out of the q-value calculation. "0" means
below `structural_zero_epsilon` (1e-8, `settings.yaml`). The q-values are informational; the
flag uses the raw quantile.

### Stage 2: correlation landscape (`human_corr_landscape.py`)

This uses the real labels only and has no shuffle dependency.

- **Sweep.** For all 180 (gene set × directed pair) connections and both diagnoses, it recomputes
  the sender-gene × receiver-gene r block.
- **Contribution.** A pair's `contribution` is r if r > 0, else 0. Summed, the contributions
  reproduce `real_value`.
- **Per gene.** Contributions are rolled up per gene and role. `PCT_DIFF` is the SCZ − CTL change
  in a gene's share of its connection's total.
- **Concentration curves.** It writes curves of how much of the total the top x% of
  gene-connection instances explain.

Stage 5's global threshold comes from this full sweep.

### Stage 4: correlation contributions (`human_corr_contrib.py`)

This is the same decomposition as stage 2, restricted to the connections in
`delta_quantile_flagged.csv`.

### Stage 5: driver prioritization (`human_prioritize_drivers.py`)

For each flagged row, a sender or receiver gene is a driver when two conditions hold:

- **Size.** Its |`PCT_DIFF`| is in the top `top_quantile` (0.3) of a reference pool.
- **Sign.** The sign of `PCT_DIFF` matches the status: > 0 for hyperactive, < 0 for hypoactive.

Two pools are used:

- **global:** every instance in stage 2's full landscape.
- **connectionwise:** only that connection's own genes in that role.

`sender_prioritized` and `receiver_prioritized` are the union of the two.

**Direction consistency.** Each row also gets `is_direction_consistent`
(`src/connection_analyzer._is_direction_consistent`). It is True when the row's status agrees
with its sender's abundance change: either the abundance moved the same way, or it moved the other
way but not significantly (P1's t-test, p > 0.05). A row whose sender abundance has no direction
(identical, e.g. 0 in both groups) gets False.

### Stage 6: GO over-enrichment (`human_over_enrichment.py`)

1. **Direction filter.** It keeps only the rows where stage 5's `is_direction_consistent` is
   True. If the column is missing, which means a driver CSV made by an older version, it stops and
   asks you to re-run stage 5.
2. **Enrichment.** For each kept connection and each of `cat2` (sender drivers) and `cat1`
   (receiver drivers), it runs PANTHER over-representation (Fisher, FDR, human, 3 GO domains) and
   keeps terms with FDR < 0.1.
3. **Comparison.** It keeps the terms shared with that gene set's P2 baseline where the driver
   subset has a lower FDR **or** a higher fold enrichment than the baseline.
4. **Output.** One xlsx per connection, with 6 sheets: `<domain>#cat1|cat2`.

### Prerequisites

- **P1 `human_group_abundance_compare.py`.** It computes abundance per donor rather than per region mean,
  then a CTL vs SCZ Student t-test and Hedges' g for each (region, neurotransmitter).
- **P2 `baseline_GO_enrichment.py`.** It runs the same PANTHER enrichment on each gene set's full
  `cat1`/`cat2` lists.

---

## 2. Requirements

**Packages.** Tested with Python 3.13.9, numpy 2.3.5, pandas 2.3.3, scipy 1.17.0, PyYAML 6.0.3,
openpyxl 3.1.5, matplotlib 3.10.7, seaborn 0.13.2, statannotations 0.7.2, requests 2.32.5 and
fake-useragent 2.2.0.

```
pip install -r requirements.txt
```

- seaborn and statannotations are only needed by P1. See the compatibility notes in `README.md`.
- matplotlib is needed by the plots of P1 and stage 2. Stage 5 imports stage 2, so it needs
  matplotlib too.
- requests and fake_useragent are only needed by P2 and stage 6.

**Single-threaded BLAS.** Every entry script sets `OPENBLAS_NUM_THREADS=1` before it first imports
numpy, so parallelism comes only from worker processes and the wrapper's process pool.

**Internet.** P2 and stage 6 call the PANTHER REST API (`pantherdb.org`). Everything else runs
offline.

**`settings.yaml`.**

- **`project_directory`.** Replace the placeholder `<YOUR_PROJECT_DIR>` with the absolute path of
  your copy of the repo. Every script builds its paths and `sys.path` from it.
- **`python_interpreter`.** Replace the placeholder `<YOUR_PYTHON_INTERPRETER>` with the absolute
  path of your Python executable. The wrapper launches every stage-1 subprocess with this
  interpreter, so it must point to an environment that has the packages above.
- **`human_label_shuffle_wrapper.*`.** It sets:

  - `n_label_shuffles`;
  - `initial_job_name` (`<JOB>`);
  - `unshuffled_dir_name`, which is also the default output of the stage-1 scripts run by hand;
  - `log_filename`;
  - `process_wait_interval`.

  Stages 3–6 default to `initial_job_name` when no job is given, so keep it pointing at the job
  you are analysing.
- **`human_delta_quantile_report.*`.** It sets the flag thresholds (0.05 / 0.95) and
  `structural_zero_epsilon`. Write that one as `1.0e-8`, because PyYAML reads `1e-8` as a string.
- **`human_group_abundance_compare.*`.** It sets P1's folder and file names. Stage 3 reads the
  stats file from the same keys.
- **`analysis_global_settings.max_concurrency`.** It caps the wrapper's subprocess pool. `null`
  means 1.
- **`human_oneshot.*`.** Seeds and null sizes. The default background is `gene_set_settings.*`,
  which is `exclude_NT/BACKGROUND_GENES_00.pickle`.

---

## 3. Inputs

```
settings.yaml
metadata/
├── human_sample_matching_table.csv      donor_id × {PFC, CC, HP, NAc, CN} → raw sample id (blank = none), + disease
└── gene_data/                           CURATED_GENES.npy, CURATED_GENE_SET.pickle, {exclude,include}_NT/BACKGROUND_GENES_0[0-3].pickle
human_expression/
└── <cohort>_<region>/expression.csv     genes × raw sample ids, one folder per cohort-region
```

- **Folders.** `src/project_variables.py: HUMAN_DATASET_REGION_TO_RESIDUALS_FOLDER` lists all 21
  cohort-region folders.
- **Missing folders.** A missing folder is skipped with a notice (`load_human_residuals: 13 dataset-region folder(s) not on disk, skipped: …`). That is expected with the public toy data
  (TOY_DATA.md), which has 8 folders and 61 donors.
- **Region order.** Every matrix uses the order PFC, CC, HP, CN, NAc (`HUMAN_REGIONS`).

---

## 4. Running the pipeline

The commands pass every argument explicitly, using the reference-run values.

### PowerShell (Windows)

```powershell
$PY  = "<YOUR_PYTHON_INTERPRETER>"                     # = python_interpreter in settings.yaml
$P   = "<YOUR_PROJECT_DIR>"                            # = project_directory in settings.yaml
$BG  = "$P\metadata\gene_data\exclude_NT\BACKGROUND_GENES_00.pickle"
$OUT = "$P\human_analysis_results"
$JOBNAME = "10_shuffle_20261006"                       # = initial_job_name in settings.yaml
$JOB = "$OUT\$JOBNAME"
$SUM = "$OUT\label_shuffle_summary\$JOBNAME"
$H   = "$P\scripts\human_analysis"

# P2: once per gene-set version; needs internet and takes a while. A fresh clone has no
# go_enrichment/GO_baseline/, so run it before wave 4. It can also run alongside waves 1-3.
& $PY "$P\scripts\prerequisite\baseline_GO_enrichment.py"

# Wave 1: run these three at the same time (e.g. three terminals or Start-Job)
& $PY "$H\human_label_shuffle_wrapper.py" $BG $OUT transformation "$OUT\unshuffled" 10 1 14 "$JOB\label_shuffle_wrapper_log.txt" False
& $PY "$P\scripts\prerequisite\human_group_abundance_compare.py"
& $PY "$H\downstream_analysis\human_corr_landscape.py"

# Wave 2: after all of wave 1 has finished
& $PY "$H\human_delta_quantile_report.py" "$OUT\unshuffled" $JOB $BG transformation

# Wave 3: these two can run at the same time
& $PY "$H\downstream_analysis\human_corr_contrib.py" $SUM
& $PY "$H\downstream_analysis\human_prioritize_drivers.py" $SUM

# Wave 4: needs internet
& $PY "$H\downstream_analysis\human_over_enrichment.py" $JOBNAME
```

### bash (Linux server)

```bash
PY=python; P=/path/to/project; BG=$P/metadata/gene_data/exclude_NT/BACKGROUND_GENES_00.pickle
OUT=$P/human_analysis_results; JOBNAME=10_shuffle_20261006; JOB=$OUT/$JOBNAME
SUM=$OUT/label_shuffle_summary/$JOBNAME; H=$P/scripts/human_analysis

$PY $P/scripts/prerequisite/baseline_GO_enrichment.py      # P2: once per gene-set version; needs internet
$PY $H/human_label_shuffle_wrapper.py "$BG" "$OUT" transformation "$OUT/unshuffled" 10 1 14 "$JOB/label_shuffle_wrapper_log.txt" False &
$PY $P/scripts/prerequisite/human_group_abundance_compare.py &
$PY $H/downstream_analysis/human_corr_landscape.py &
wait
$PY $H/human_delta_quantile_report.py "$OUT/unshuffled" "$JOB" "$BG" transformation
$PY $H/downstream_analysis/human_corr_contrib.py "$SUM" &
$PY $H/downstream_analysis/human_prioritize_drivers.py "$SUM" &
wait
$PY $H/downstream_analysis/human_over_enrichment.py "$JOBNAME"
```

### Arguments

The arguments are positional: to set argument *n* you must also give arguments 1 to *n*−1.
`None` is the literal string `None`.

**`human_label_shuffle_wrapper.py`**

| # | Name                         | Default                                   | Meaning                                                                                                  |
| - | ---------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1 | `background_source`        | `exclude_NT/BACKGROUND_GENES_00.pickle` | background panel file                                                                                    |
| 2 | `output_dir`               | `human_analysis_results`                | replicates go to `<output_dir>/<JOB>/run_NNNNNN/`                                                       |
| 3 | `p_value_method`           | `transformation`                        | `transformation` or `permutation`. It currently only names the results folder; see section 7         |
| 4 | `unshuffled_dir`           | `human_analysis_results/unshuffled`     | where the real run goes                                                                                  |
| 5 | `n_replicates`             | `n_label_shuffles` (settings.yaml)      | number of label shuffles                                                                                 |
| 6 | `seed_start`               | `1`                                     | first shuffle seed. Seeds are `seed_start … seed_start + n − 1`                                       |
| 7 | `max_concurrent_processes` | `max_concurrency` (settings.yaml)       | pool size, must be ≥ 1                                                                                  |
| 8 | `log_file`                 | `<JOB>/label_shuffle_wrapper_log.txt`   | appended, never truncated                                                                                |
| 9 | `normalize_weights`        | `False`                                 | `True` only for the exact string `True`. See stage 1c. The reference run passed `False` explicitly |

The `<JOB>` folder name always comes from `settings.yaml`, not from an argument.

**`human_oneshot.py`** (normally called by the wrapper)

| # | Name                   | Default                                                         | Meaning                                                                                                                                                                            |
| - | ---------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `background_source`  | as above                                                        |                                                                                                                                                                                    |
| 2 | `output_dir`         | `human_analysis_results/unshuffled` (`unshuffled_dir_name`) | results go to `<output_dir>/<bg stem>_<p method>/`                                                                                                                                |
| 3 | `p_value_method`     | `transformation`                                              |                                                                                                                                                                                    |
| 4 | `label_shuffle_seed` | `None`                                                        | `None` = real labels, written as `unshuffled_directed_signaling_strength#{CTL,SCZ}.csv.gz`. An integer is a shuffle, written as `label_shuffled/run_NNNNNN#{CTL,SCZ}.csv.gz` |
| 5 | `n_workers`          | CPU count − 2                                                  | the wrapper passes `1` for replicates                                                                                                                                             |
| 6 | `tie_rank_handler`   | `mid-rank`                                                    | `strict`, `inclusive`, `mid-rank`, `discard`                                                                                                                               |

**`human_abundance.py`**: (1) `output_dir`, default `human_analysis_results/unshuffled`;
(2) `label_shuffle_seed` (`None` = real labels).

**`human_weighted_connectivity.py`**: (1–2) the CTL/SCZ signaling files, (3–4) the CTL/SCZ
abundance xlsx, (5) `output_dir`, (6) `background_source`, (7) `p_value_method`,
(8) `label_shuffle_seed` (used only for log text), (9) `normalize_weights`. The caller must pair
inputs from the same seed. The wrapper does this.

**`human_delta_quantile_report.py`**

| #    | Name                                      | Default                                                                      |
| ---- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| 1    | `unshuffled_dir`                        | `human_analysis_results/unshuffled`                                        |
| 2    | `job_dir`                               | `human_analysis_results/<JOB>`. Every `run_NNNNNN` folder in it is used  |
| 3–4 | `background_source`, `p_value_method` | must match stage 1                                                           |
| 5–6 | `output_path`, `flagged_output_path`  | `label_shuffle_summary/<job_dir name>/delta_quantile_{report,flagged}.csv` |
| 7    | `group_abundance_stats_path`            | `group_abundance_comparison/group_abundance_stats.csv` (P1)                |

**`human_corr_contrib.py`**: (1) a flagged CSV or its folder, default `<JOB>`'s;
(2) `output_dir`, default `corr_contrib/` next to the flagged file; (3) `p_value_method`.

**`human_prioritize_drivers.py`**: (1) a flagged CSV or its folder; (2) `output_dir`, default the
flagged file's folder; (3) `top_quantile`, default 0.3 (settings.yaml); (4) the landscape
`gene_contribution_diff_full.csv`; (5) `p_value_method`.

**`human_corr_landscape.py`**: (1) `output_dir`, default `corr_landscape/`; (2) `p_value_method`.

Stages 2, 4 and 5 count a gene pair as contributing when r > 0, the same rule stage 1 uses, so
their contributions add up to stage 1's `real_value`.

**`human_over_enrichment.py`**:

1. A job name (default `<JOB>`), a folder that contains `driver_gene_prioritization.csv`, or the
   CSV path itself. A job name is looked up under `label_shuffle_summary/<job>/`.
2. `output_dir`. The default is `go_enrichment/<name of the CSV's folder>/`.

**`human_group_abundance_compare.py`**: (1) `output_dir`, default `group_abundance_comparison/`.

---

## 5. Outputs

```
human_analysis_results/
├── unshuffled/                                       stage 1, real labels
│   ├── human_abundance_control.xlsx                  sheets mean/median/trimean, 9 NT × 13 columns
│   ├── human_abundance_schizo.xlsx
│   └── background_genes_00_transformation/
│       ├── unshuffled_directed_signaling_strength#CTL.csv.gz   180 rows
│       ├── unshuffled_directed_signaling_strength#SCZ.csv.gz
│       ├── weighted_connectivity#{CTL,SCZ}.xlsx      'total' + 9 'weighted_<NT>' sheets, 5 × 5
│       └── meanz_weighted#{CTL,SCZ}.csv              same values as the 'total' sheet
├── <JOB>/                                            stage 1, label shuffles
│   ├── label_shuffle_wrapper_log.txt
│   └── run_000001 … run_0000NN/                      same layout as unshuffled/, except the signaling files are
│       └── background_genes_00_transformation/label_shuffled/run_NNNNNN#{CTL,SCZ}.csv.gz
├── group_abundance_comparison/                       P1: group_abundance_stats.csv, individual_abundance_boxplots.png
├── corr_landscape/                                   stage 2: gene_pair_contribution_full.csv.gz, gene_contribution_rollup_full.csv,
│                                                     gene_pair_contribution_diff_full.csv.gz, gene_contribution_diff_full.csv,
│                                                     cumulative_{contribution,diff}_curve.csv, cumulative_contribution_plot.png
├── label_shuffle_summary/<JOB>/                      stage 3 + 5
│   ├── delta_quantile_report.csv                     180 rows
│   ├── delta_quantile_flagged.csv
│   ├── driver_gene_prioritization.csv
│   └── corr_contrib/                                 stage 4: gene_pair_contribution.csv.gz, gene_contribution_rollup.csv,
│                                                     gene_pair_contribution_diff.csv.gz, gene_contribution_diff.csv
└── go_enrichment/
    ├── GO_baseline/<gene set>/cat{1,2}_baseline_<domain>.csv   P2
    └── <JOB>/<NT>_<sender>_<receiver>.xlsx                     stage 6
```

### `*_directed_signaling_strength#<disease>.csv.gz`

These have the same columns as the macaque `directed_signaling_strength.csv`
(`MACAQUE_PIPELINE.md`, section 5), except that `n_donors` replaces `n_monkeys`. There is one
row per gene set × directed region pair (9 × 20 = 180).

### `weighted_connectivity#<disease>.xlsx` and `meanz_weighted#<disease>.csv`

All are 5 × 5 with rows = sender and columns = receiver. The diagonal is NaN because there are no
self-pairs.

| File / sheet                                                    | Content                                                                       |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `weighted_<NT>` (DA, 5-HT, Glu, GABA, NE, ACh, Gly, epi, Syn) | normalised `mean_z` × sender abundance (raw if `normalize_weights=False`) |
| `total` = `meanz_weighted#<disease>.csv`                    | Σ of the 9 terms, using max-normalised abundance                             |

### `delta_quantile_report.csv` / `delta_quantile_flagged.csv`

| Column                                                                                                                                               | Meaning                                                 |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `neurotransmitter`, `sender_region`, `receiver_region`                                                                                         | the connection                                          |
| `SCZ_weighted`, `CTL_weighted`, `SCZ_non_weighted`, `CTL_non_weighted`                                                                       | real-run values                                         |
| `delta_weighted`, `delta_non_weighted`                                                                                                           | SCZ − CTL                                              |
| `delta_*_mean`, `_std`, `_skewness`, `_kurtosis`                                                                                             | moments of the shuffled-null deltas                     |
| `delta_weighted_normalized`                                                                                                                        | (delta − null mean) / null std. NaN when std = 0       |
| `quantile_weighted`, `quantile_non_weighted`                                                                                                     | mid-rank quantile of the real delta in the null         |
| `p_value_*`, `q_value_*`                                                                                                                         | two-sided p from the quantile, and BH q (informational) |
| flagged only: `status`                                                                                                                              | `hypoactive` / `hyperactive`                        |
| flagged only: `mean_diff_schizophrenia_minus_control`, `p_value`, `hedges_g`, `sender_abundance_direction`, `sender_abundance_significance` | P1 stats for (sender region, NT)                        |

### `driver_gene_prioritization.csv`

This is the flagged table plus:

- `sender_prioritized` and `receiver_prioritized`, the union;
- the four method-specific lists `{sender,receiver}_prioritized_{global,connectionwise}`;
- `is_direction_consistent`, which marks the rows stage 6 enriches.

The lists are stored as Python list literals.

---

## 6. Reference run and expected results

The reference run used the toy data (TOY_DATA.md: 61 donors, 8 cohort-region folders). It ran on
Windows 11 with 16 cores, using the exact commands in section 4: 10 label shuffles,
`max_concurrency` 14, `normalize_weights=False` and the default
`exclude_NT/BACKGROUND_GENES_00.pickle`. P2 had been run once beforehand, so the timings below
don't include it. Ran on 2026-10-07.

| Wave | Script                                           | Wall time     | Result                                                                       |
| ---- | ------------------------------------------------ | ------------- | ---------------------------------------------------------------------------- |
| 1    | wrapper (total)                                  | 418 s         | `unshuffled/` + `run_000001…run_000010`, all complete                   |
|      | ↳ phase 0, real run                             | 37 s          | abundance 3 s, oneshot 32 s (14 workers), weighted connectivity 2 s          |
|      | ↳ 10 × abundance                               | 41 s          | about 4 s of work; the rest is pool polling (`process_wait_interval` 20 s) |
|      | ↳ 10 × oneshot (1 worker each, all 10 at once) | 5 min         | about 3.8 min of work, plus polling at 3 × 20 s                             |
|      | ↳ 10 × weighted connectivity                   | 40 s          | about 3 s of work                                                            |
| 1    | P1 `human_group_abundance_compare.py`           | 21 s          | 45 rows (5 regions × 9 NT)                                                  |
| 1    | 2 `human_corr_landscape.py`                     | 17 s          | 180 connections                                                              |
| 2    | 3 `human_delta_quantile_report.py`              | 3 s           | 180 rows, 19 flagged                                                         |
| 3    | 4 `human_corr_contrib.py`                       | 3 s           | 19 combos                                                                    |
| 3    | 5 `human_prioritize_drivers.py`                 | 4 s           | 19 rows, 18 with `is_direction_consistent`                                  |
| 4    | 6 `human_over_enrichment.py`                    | 143 s         | 18 xlsx. About 18 s per connection that has drivers (6 PANTHER calls)        |

Use these checks to confirm a run is healthy.

**Signaling files** (each of the 22: 11 runs × 2 diagnoses)

- 180 rows and no NaN.
- |`mean_z`| ≤ 3.290808. That bound is |Φ⁻¹(0.5/1001)|, and it is reachable: several shuffled
  rows sit exactly on it, so don't test with a tighter constant such as 3.2906.
- `n_donors` is 0 on 72 rows: the 4 no-data pairs × 2 directions × 9 gene sets. On pairs with
  data it is 8–17 for CTL and 7–11 for SCZ. `mean_z` is exactly 0 on the no-data rows.
- Every shuffled replicate has the **same `n_donors` per pair** as the real run, because the
  stratified shuffle keeps the counts.
- Every replicate's `mean_z` differs from the real run's: the largest |Δ| per replicate was
  1.9–6.1.
- In the real run, CTL has 12 rows with `mean_z` > 1.645 and 4 with `mean_z` < −1.645. SCZ has
  9 and 9.

**Abundance xlsx**

- Each file has 3 sheets of 9 × 13: 9 neurotransmitters.
- The merged-region columns have no NaN and range from 0 to 10.82 (Syn in PFC, CTL).
- The same 18 (NT, region) cells are exactly 0 in both diagnoses. The cause is synthesis genes
  missing from the toy data, which is the same mechanism as in `MACAQUE_PIPELINE.md`, section 6.
  The cells are:

  - DA in PFC and CC
  - 5-HT in PFC, CC and CN
  - ACh in PFC, CC and HP
  - NE and epi in all 5 regions

  They are also the 18 rows with NaN `p_value` in `group_abundance_stats.csv`.

**Weighted connectivity**

- Every sheet is 5 × 5, with NaN only on the diagonal.
- `meanz_weighted.csv` equals the `total` sheet.
- In the real run, the `total` sheet's off-diagonal values range from 1.309 to 2.897 for CTL and
  from 0.936 to 2.210 for SCZ.
- Hand check for `weighted_Glu[PFC→HP]`, CTL: (1.569939 + 3.290808) / 6.581616 × 7.353785
  = 5.431020, which equals the sheet value.

**Delta-quantile report**

- 180 rows, and every quantile lies on the grid (k + 0.5)/11.
- 86 rows have `quantile_weighted` = 0.5. These include the 72 structurally-zero rows, which have
  NaN `q_value_weighted`.
- 19 rows are flagged: 17 hypoactive and 2 hyperactive. 10 of them are on no-data pairs
  (section 7).
- The smallest `q_value_weighted` is 0.517.

**Landscape**

- Row counts: 34,520 gene-pair, 7,320 rollup, 17,260 pair-DIFF and 3,660 gene-DIFF rows.
- |`PCT_DIFF`| concentration: 50% / 80% / 90% of the total come from the top 8.4% / 20.4% / 27.5%
  of instances.

**Downstream**

- `corr_contrib/` has 3,720 / 844 / 1,860 / 422 rows. Summing `contribution` reproduces stage 1's
  `real_value` to 2.4 × 10⁻⁶, which is the `%.6g` rounding of the CSV.
- The global |`PCT_DIFF`| threshold is 0.08098.
- 9 flagged rows have sender drivers and 9 have receiver drivers, all on pairs with data.
- 18 of the 19 flagged rows are direction-consistent. Gly HP→PFC is dropped because it is
  hypoactive while Gly abundance is significantly elevated in HP. That gives 18 GO xlsx files:
  - 7 have at least one over-enriched term sheet.
  - Gly NAc→CC has drivers but no PANTHER term with FDR < 0.1.
  - The 10 no-data rows have "no prioritized genes" placeholders.

Other machines or BLAS builds can still differ in the last bits, so compare with a tolerance

rather than a hash.

Quick check:

```python
import pandas as pd, glob
R = 'human_analysis_results'
files = glob.glob(f'{R}/unshuffled/*/unshuffled_directed_signaling_strength#*.csv.gz') \
      + glob.glob(f'{R}/10_shuffle_20261006/run_*/*/label_shuffled/*.csv.gz')
assert len(files) == 22
for f in files:
    d = pd.read_csv(f)
    assert len(d) == 180 and d.isna().sum().sum() == 0 and d['mean_z'].abs().max() <= 3.29081, f
```

---

## 7. Usage rules and notes

- **Always pass argument 9 (`normalize_weights`) to the wrapper.** The reference run used
  `False`, which is also the default.
- **Replicate failures are silent.** `run_process_pool` sends each child's stdout and stderr to
  DEVNULL and ignores exit codes, so the wrapper exits 0 even when a replicate crashed. Only
  phase 0 (`subprocess.run(check=True)`) stops on error. Count the files after every wrapper run
  (section 6). To repair one seed, re-run its three commands by hand, as below. Rerunning the
  whole wrapper would also redo phase 0.
  ```
  human_abundance.py <JOB>/run_NNNNNN N
  human_oneshot.py <bg> <JOB>/run_NNNNNN transformation N 1
  human_weighted_connectivity.py <4 input paths> <JOB>/run_NNNNNN <bg> transformation N False
  ```
- **Stage 3 uses every `run_*` folder in `job_dir`.** A stale or half-written replicate silently
  becomes part of the null. Give each configuration its own `initial_job_name` (and
  `unshuffled_dir`) instead of reusing a folder.
- **The real run is the "+1" of every job, and it gets overwritten.** A wrapper call makes
  N label shuffles plus one deterministic real run (phase 0).
  - Phase 0 always re-runs and writes to argument 4, `unshuffled_dir`. Its default does **not**
    follow argument 2.
  - So every call, including one that only extends a job with a new `seed_start`, overwrites
    `human_analysis_results/unshuffled/`.
  - With the same background and settings that is harmless, because the real run is
    deterministic.
  - For a different configuration (another background panel, `include_NT`, …), pass arguments
    4 and 8 too. Otherwise the shared real run, which every job's stage 3 reads, changes under
    the other jobs.
- **Don't change `human_oneshot.quantile_bg_n_null_samples` between stage 1 and stage 3.**
  Stage 1c and stage 3 both recompute the normalisation bound `Z_BOUND` from the current setting.
  That includes stage 3 for old jobs: it re-normalises every replicate when it runs. A changed
  setting silently rescales the weighted cells and the `*_non_weighted` deltas. This is the same
  rule as in `MACAQUE_PIPELINE.md`.
- **p-values are currently inert.** Stage 1 sums every positive r with no p-value filter, so:
  - `transformation` and `permutation` give identical stage-1 statistics; `permutation` is only
    much slower.
  - `p_value_method` names the results folder (`<bg stem>_<p method>/`).
  - `permutation` is kept so a p-based rule can be added back later. Stages 2, 4 and 5 accept only
    `transformation` and raise an error for `permutation`, so use `transformation` for full runs.
- **10 shuffles is a smoke test, not an analysis.** The quantile resolution is 1/11, so a row is
  flagged only when the real delta is beyond all 10 null deltas. The two-sided p is then ≥ 0.091,
  so BH q never gets small. Run thousands of shuffles on a server; earlier jobs used 10k.
- **Ignore flags on the toy data's no-data pairs** (CC–HP, CC–CN, HP–NAc, CN–NAc; TOY_DATA.md).
  There `mean_z` = 0, so every `weighted_<NT>` cell is 0.5 × sender abundance. The delta then
  only reflects the sender's abundance difference. 10 of the 19 reference flags are of this kind.
  They get empty driver lists and placeholder GO sheets.
- **Structurally zero abundance** (18 NT × region cells in the toy data) makes a weighted row
  0 in every run. That row then has quantile 0.5, is never flagged and is left out of the q-values.
- **Running stage 1 by hand.** `human_abundance.py`, `human_oneshot.py` and
  `human_weighted_connectivity.py` default to the same `human_analysis_results/unshuffled/`
  (`unshuffled_dir_name`) that the wrapper and stage 3 use. Their defaults alone give stage 3 a
  real run it can find, and they overwrite the wrapper's real run in the same way.
- **`<JOB>` comes only from `settings.yaml`.** Set `initial_job_name` before each new job. Stages
  3–6 also default to it.
- **Expected warnings.** `RuntimeWarning: Mean of empty slice` / `Degrees of freedom <= 0` come
  from the 4 no-data pairs in the toy data, and so do the `load_human_residuals: 13 … skipped`
  notices.
- **Stage 6 and P2 send gene-symbol lists to `pantherdb.org`.** They have no retry logic, so a
  network error stops the script. Re-run it; it rewrites all the xlsx files.
- **PowerShell 5.1 strips embedded double quotes** from arguments passed to `python.exe`. Use a
  script file instead of `python -c "…"` with quoted strings inside.
- **These are toy-data results.** They check that the code runs and are not for interpretation.

---

## 8. Variants

- **More shuffles (server).**

  1. Set `n_label_shuffles`, `initial_job_name` and `max_concurrency` (about cores − 2) in
     `settings.yaml`.
  2. Run the wave-1 wrapper command.

  To extend an existing job, pass `seed_start` = (last seed + 1) with the same job name. Phase 0
  re-runs and overwrites `unshuffled/` with the same statistics.
- **Another background panel** (e.g. `include_NT`). Change argument 1 of the wrapper and of
  stage 3. Also give the configuration its own `unshuffled_dir` and job name, for example
  `unshuffled_include_nt` and `<n>_shuffle_<date>_include_nt`. The results subfolder is named
  only from the background file's stem, so `include_NT/00` and `exclude_NT/00` would otherwise
  collide.
- **`normalize_weights=True`.** Pass `True` as argument 9. It changes only the `weighted_<NT>`
  sheets, and therefore stage 3's `*_weighted` columns and flags.
- **Not part of this pipeline run:** `prerequisite/human_donor_split.py`, which writes stratified
  donor splits into `donor_splits/`. Its `__main__` ignores argv.
