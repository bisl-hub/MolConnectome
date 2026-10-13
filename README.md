# MolConnectome

Code for reconstructing an **interregional molecular connectome** from postmortem brain
transcriptomes. For each neurotransmitter gene set and each directed region pair, it measures
how strongly the sender-side genes in one region co-vary with the receiver-side genes in another,
across individuals. It tests that strength against background-gene nulls and weights it by the
sender region's estimated ligand abundance.

- **Macaque** (15 regions) reconstructs the connectome. See [MACAQUE_PIPELINE.md](MACAQUE_PIPELINE.md).
- **Human** (5 regions, schizophrenia vs control) reconstructs it per diagnosis. It then:
  - tests every SCZ − CTL difference against a diagnosis-label-shuffle null;
  - traces the significant connections back to driver genes and GO terms.

  See [HUMAN_PIPELINE.md](HUMAN_PIPELINE.md).

This repository accompanies the preprint listed under [Citation](#citation).

## Repository layout

```
├── settings.yaml                  paths, seeds and output names (set two placeholders first)
├── requirements.txt
├── metadata/                      sample-matching tables, gene list, gene sets, background panels
├── macaque_expression/            toy data: 15 monkeys, 15 regions
├── human_expression/              toy data: 61 donors from 3 public cohorts
└── scripts/
    ├── src/                       shared code: loaders, correlation, null quantiles, abundance
    ├── macaque_analysis/          macaque steps 1-3, plus the optional all-gene correlation
    ├── human_analysis/            human stage 1 (label-shuffle wrapper and its 3 steps) and stage 3
    │   └── downstream_analysis/   human stages 2, 4, 5 and 6
    └── prerequisite/              human P1 (group abundance), P2 (GO baseline), donor splits
```

Results go to `macaque_analysis_results/` and `human_analysis_results/`. Both are created on the
first run and are not part of the repository.

## Installation

Python 3.10 or newer (tested with 3.13.9).

```
git clone https://github.com/bisl-hub/MolConnectome.git
cd MolConnectome
pip install -r requirements.txt
```

## Configuration

Open `settings.yaml` and replace the two placeholders at the top:

```yaml
project_directory: <YOUR_PROJECT_DIR>          # absolute path of this repository
python_interpreter: <YOUR_PYTHON_INTERPRETER>  # absolute path of your python executable
```

- **How paths are found:** every script finds `settings.yaml` at the repository root, relative to
  its own location. It then builds all input and output paths from `project_directory`.
- **Who needs `python_interpreter`:** only the human label-shuffle wrapper, which starts its
  subprocesses with it.

## Quick start (toy data)

The three macaque steps run without arguments, using the default background panel and output
folder. Step 1 takes about 2 minutes on a 16-core machine.

```
python scripts/macaque_analysis/macaque_oneshot.py                # step 1: directed signalling strength
python scripts/macaque_analysis/macaque_abundance.py              # step 2: ligand abundance
python scripts/macaque_analysis/macaque_weighted_connectivity.py  # step 3: abundance-weighted connectivity
```

- **Checking the output:** compare it with the reference numbers in
  [MACAQUE_PIPELINE.md](MACAQUE_PIPELINE.md), section 6.
- **Human pipeline:** its run order, commands and expected results are in
  [HUMAN_PIPELINE.md](HUMAN_PIPELINE.md). Its GO steps need internet access.

## Documentation

| Document | Content |
|---|---|
| [MACAQUE_PIPELINE.md](MACAQUE_PIPELINE.md) | macaque steps, arguments, outputs, reference run |
| [HUMAN_PIPELINE.md](HUMAN_PIPELINE.md) | human stages and their dependencies, arguments, outputs, reference run |
| [TOY_DATA.md](TOY_DATA.md) | what the bundled toy data contain, and their limits |

## Toy data

The bundled expression data are small, de-identified subsets of public datasets. They let you run
every step, but they are far too small to reproduce the preprint's results. The controlled-access
human cohorts used in the preprint are not included. See [TOY_DATA.md](TOY_DATA.md).

## Compatibility notes

- **seaborn and statannotations.** statannotations 0.6.x requires seaborn < 0.12. Installing it
  next to a newer seaborn makes pip downgrade seaborn. `requirements.txt` therefore asks for
  statannotations ≥ 0.7 (tested: statannotations 0.7.2 with seaborn 0.13.2). Only
  `scripts/prerequisite/human_group_abundance_compare.py` (human P1) imports either package.
- **NumPy 2.** `metadata/gene_data/CURATED_GENES.npy` and the gene-set pickles were written with
  NumPy 2. NumPy 1.x may fail to load them.
- **Newer releases.** These were also checked with pandas 3.0.6, numpy 2.5.4, scipy 1.18.1 and
  matplotlib 3.11.2:
  - macaque steps 1 and 3;
  - the human stage-1 real run;
  - human stage 2 and P1.

  All of them reproduced the toy-data reference outputs.

## Data sources

- **Macaque:** GEO [GSE180092](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE180092).
- **Human:**
  - BrainSeq Consortium (DLPFC and hippocampus);
  - GEO [GSE53987](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53987);
  - GEO [GSE80655](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE80655).

The toy data remain subject to the terms of these original sources.

## License

The code is released under the GNU General Public License v3.0 (see [LICENSE](LICENSE)). The
bundled data keep the terms of their original sources.

## Citation

If you use this code, please cite:

> Kwon O, Kim U, Webster MJ, Kim S, Lee D. Transcriptomic reconstruction of an interregional
> molecular connectome in schizophrenia. *bioRxiv* (2026).
> https://doi.org/10.64898/2026.10.05.756648

Citation metadata is also in [CITATION.cff](CITATION.cff).
