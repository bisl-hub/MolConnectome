# LLM benchmarking

Two scripts measure how well a large language model judges whether a PubMed abstract **supports**,
**rejects** or is **neutral** toward a stated conclusion:

| Script (`scripts/llm_benchmarking/`) | Produces |
| --- | --- |
| `run_experiment_oss.py` | `llm_benchmarking/results/<model>_<result_file_suffix>`: one verdict, confidence and rationale per abstract–conclusion pair |
| `check_metrics.py` | accuracy, macro precision / recall / F1 and the confusion matrix, printed. The model's row in `scripts/llm_benchmarking/llm_eval_metrics.csv` is added or updated |

The benchmark needs an OpenAI-compatible endpoint that serves the model. Its address and key go in
`settings.yaml` (section 3). Unlike the connectome pipelines, the inputs and the gpt-oss-120b
predictions are part of the repository.

---

## 1. Data

`llm_benchmarking/inputs/bionli_augmented_400.csv` has 400 pairs, one per PubMed ID (`pmid`):

| Column | Content |
| --- | --- |
| `supp_set` | the abstract (sent as the premise) |
| `conclusion` | the conclusion to judge (sent as the hypothesis); entity markers such as `<el> … <le>` and `<re> … <er>` are kept |
| `ori_conclusion` | the paper's original conclusion sentence, with the same markers |
| `label_cat` | how the pair was built (table below) |
| `gold_label` | `SUPPORT`, `REJECT` or `NEUTRAL` |

| `label_cat` | Gold label | Pairs |
| --- | --- | --- |
| `pos` | SUPPORT | 134 |
| `neutral_random` | NEUTRAL | 133 |
| `posToNeg` | REJECT | 78 |
| `LPR` | REJECT | 49 |
| `negToPos` | REJECT | 6 |

The pairs labelled `neutral_random` join an abstract to the conclusion of an unrelated paper.

## 2. What the model is asked

The instructions are in `llm_benchmarking/system_prompt.txt`. It defines the three verdicts:
- a null finding is NEUTRAL, not REJECT;
- REJECT needs an explicit opposite effect.

It also defines three confidence levels, and asks for a JSON object with `verdict`, `confidence`
and `rationale`.

Each pair is one chat request:

| Field | Value |
| --- | --- |
| model | `openai/gpt-oss-120b` (`model_id`) |
| developer message | `# Instructions` + `system_prompt.txt` |
| user message | `Premise: <supp_set>` + blank line + `Hypothesis: <conclusion>` |
| `reasoning_effort` | `high` |
| `max_tokens`, `temperature`, `top_p`, `presence_penalty` | 8192, 1.0, 0.95, 1.5 |
| `seed` | 20260310 (`random_seed`) |

**Endpoint behaviour (checked 2026-10).** Two things changed on the endpoint after the bundled
predictions were made:
- it rejects the `extra_body={"reasoning": {"enabled": true}}` parameter;
- if the client also sends a system message, it drops the developer message. The model then never
  sees the instructions and answers in prose, without JSON.

The request above therefore sends no system message, because the endpoint writes the gpt-oss
(Harmony) system header itself. The reasoning effort is set with `reasoning_effort`.

**Parsing the answer.** The first JSON object in the answer is parsed. A verdict outside
SUPPORT / REJECT / NEUTRAL, an answer without JSON, or a failed request is stored as `ERROR`.

## 3. Settings (`settings.yaml`, section `llm_benchmarking`)

| Key | Meaning |
| --- | --- |
| `api_key`, `base_url` | endpoint credentials and address |
| `model_id`, `reasoning_effort` | model and reasoning effort |
| `llm_benchmarking_dir`, `input_dirname`, `data_file_name` | input path: `llm_benchmarking/inputs/bionli_augmented_400.csv` |
| `system_prompt_file_name` | `llm_benchmarking/system_prompt.txt` |
| `results_dirname`, `result_file_suffix` | output path: `llm_benchmarking/results/<last part of model_id>_<suffix>` |
| `n_threads` | concurrent requests (512) |
| `random_seed` | sampling seed sent with every request |
| `max_trials` | how many passes a run makes over pairs without a successful verdict (5) |
| `eval_metrics_file_name`, `eval_model_label` | summary table in `scripts/llm_benchmarking/`, and this model's row label in it |

`project_directory` must be filled in, as for the other scripts.

## 4. Running

```
python scripts/llm_benchmarking/run_experiment_oss.py            # every pair without a successful verdict
python scripts/llm_benchmarking/run_experiment_oss.py 15258100   # only these keys (comma-separated)
python scripts/llm_benchmarking/check_metrics.py                 # metrics; updates llm_eval_metrics.csv
```

**How a run resumes.**
- Rows are appended to the results file, never overwritten.
- A run sends only the pairs that have no successful verdict in that file. It repeats until none
  are left or `max_trials` passes are done.
- With the bundled results file every pair already has a verdict, so a plain run sends nothing.
  To redo the whole benchmark, pass a new results path as the third argument:
  `run_experiment_oss.py None <input_csv> <new_results_csv>`.

**What check_metrics does.**
- It drops `ERROR` rows and, for a pair with several rows, keeps the last one.
- It reports gold-standard keys without a prediction, and predicted keys that are not in the gold
  standard.
- It computes the metrics on the pairs present in both.
- Its arguments are: `[predictions_csv] [gold_csv] [model_label] [update_table]`.

## 5. Results

`scripts/llm_benchmarking/llm_eval_metrics.csv` (macro averages over the three labels; 400 pairs):

| Model | Accuracy | Precision | Recall |
| --- | --- | --- | --- |
| gemini-3-pro-preview-11-2025-high | 0.945 | 0.9477 | 0.945 |
| **gpt-oss-120b-high-reasoning** | **0.8975** | **0.9037** | **0.8974** |
| deepseek-v3.2-thinking | 0.85 | 0.8731 | 0.8498 |

Only the gpt-oss-120b predictions are in this repository. The Gemini and DeepSeek rows come from
runs whose outputs are not included. gpt-oss-120b macro F1 is 0.8955.

**gpt-oss-120b confusion matrix** (rows: gold label; columns: predicted):

| | SUPPORT | REJECT | NEUTRAL | Recall |
| --- | --- | --- | --- | --- |
| **SUPPORT** | 124 | 4 | 6 | 0.925 |
| **REJECT** | 24 | 102 | 7 | 0.767 |
| **NEUTRAL** | 0 | 0 | 133 | 1.000 |
| **Precision** | 0.838 | 0.962 | 0.911 | |

**Correct verdicts by `label_cat`:**

| `label_cat` | Correct | Most common error |
| --- | --- | --- |
| `pos` | 124 / 134 (92.5%) | NEUTRAL (6) |
| `neutral_random` | 133 / 133 (100%) | — |
| `posToNeg` | 60 / 78 (76.9%) | SUPPORT (13) |
| `LPR` | 38 / 49 (77.6%) | SUPPORT (9) |
| `negToPos` | 4 / 6 (66.7%) | SUPPORT (2) |

All 133 unrelated pairs are recognised as NEUTRAL. Most errors are REJECT pairs read as SUPPORT
(24 of 41 errors).
