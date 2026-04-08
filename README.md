# Deciphering the dysregulated molecular connectome of schizophrenia using transcriptomics with LLM-augmented evidence synthesis.

This project provides utilities for data processing and metric calculation in "Deciphering the dysregulated molecular connectome of schizophrenia using transcriptomics with LLM-augmented evidence synthesis." This includes two different tasks: 1) reconstruction of molecular connectome and 2) benchmarking augmented BioNLI dataset using different LLM models.

## Project Structure

* **/scripts**: Main executing scripts to process the results and perform calculations.
  * `LLM_API_call.py`: call LLM to prepare the output for bioNLI data.
  * `LLM_calculate_metrics.py`: use the output of `LLM_API_call.py` to calculate benchmarking metrics.
  * `calculate_correlation.py`: Step 1 of the molecular connectome reconstruction - calculating co-expressions.
  * `calculate_quantile.py`: Step 2 of the molecular connectome reconstruction - calculating quantile of co-expressions compared to background correlations.
  * `calculate_strength.py`: Step 3 of the molecular connectome reconstruction - estimating abundances of neurotransmitters and multiply them with normalized quantile to generate final weighted connectivity.
* **/src**: Utility source files and common logic.
  * `acronyms.py`: Region acronyms for `calculate_strength.py`
  * `calculate_abundance.py`: Module for estimating neurotransmitter abundances used in `calculate_strength.py`
* **/test**: Data resources for testing the pipeline (data files are ignored from git by default due to size). Contains subdivisions for `inputs/`, `original_inputs/`, `correlation/`, `strength/`, etc.
* **/test_scripts**: Small test scripts focusing on verifying the algorithms and results.

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/USERNAME/REPO_NAME.git
   cd REPO_NAME
   ```
2. **Environment Setup:** It is recommended to use a virtual environment (`venv` or `conda`).

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install git+https://github.com/arkonger/python-oklch
   ```

> **Note**: `python-oklch` is required for some figure scripts, but it is not available on PyPI. The code above will install it directly from the Git repository (https://github.com/arkonger/python-oklch).


## Usage

Various scripts can be executed independently depending on the evaluation target.
For example, to run metric evaluations:

```bash
python scripts/LLM_calculate_metrics.py
```
