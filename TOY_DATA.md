# Toy example data

A small, de-identified subset of the expression data, laid out exactly like the full project
(same folders and filenames), so the pipeline runs on it unchanged. It is for trying the code,
not for reproducing results: the sample sizes are far too small for meaningful statistics.

All values are real expression residuals, rounded to 6 significant digits. Every donor and
sample id is replaced by a meaningless one (`monkey_NN` / `BS_NNN`-style donors, `SNNNN`
samples); the metadata matching table is the only link between a sample and its donor.

## Macaque — 15 of 36 monkeys, 220 samples

- Source: GSE180092 (public), all 15 regions.
- Monkeys sampled to keep the real region-coverage mix (10 with all 15 regions, 5 missing some).
- Only the filtered expression file per region (`expression.csv`).
- Every region pair keeps 11–15 matched monkeys.

## Human — 61 donors, 151 samples, public cohorts only

| Dataset       | Regions included | Control | Schizophrenia |
| ------------- | ---------------- | ------- | ------------- |
| BrainSeq (BS) | PFC, HP          | 12      | 8             |
| GSE53987      | PFC, HP, CN      | 9       | 9             |
| GSE80655      | PFC, CC, NAc     | 12      | 11            |

- Donors sampled within each dataset × diagnosis × region-coverage stratum
  (5% of BrainSeq, 50% of each GSE dataset).
- **Not included (controlled access):** dbGaP HBCC, Stanley Array Collection,
  Neuropathology Consortium, New Collection, and BrainSeq caudate. The loader skips their
  folders and prints which were skipped.

### Region pairs with no data

Without the controlled-access cohorts, no donor has samples in both regions of these pairs:

**CC–HP, CC–CN, HP–NAc, CN–NAc** (both directions)

The pipeline still produces rows for them (`n_donors = 0`, `mean_z = 0`), but these are
placeholders. Their abundance-weighted SCZ − CTL deltas reflect only the sender region's
abundance difference, so **ignore any flagged connection on these pairs**.

Pairs with data keep 7–17 donors per diagnosis:
PFC–HP, PFC–CC, PFC–CN, PFC–NAc, CC–NAc, HP–CN.
