"""LLM benchmarking: ask an OpenAI-compatible endpoint (gpt-oss-120b by default) whether each
abstract SUPPORTs, REJECTs or is NEUTRAL toward its hypothesis.

Input:  llm_benchmarking/inputs/<data_file_name> (pmid / pair_id / data_index, supp_set, conclusion)
        and llm_benchmarking/<system_prompt_file_name>; endpoint, model and concurrency from
        settings.yaml `llm_benchmarking`.
Output: llm_benchmarking/results/<model>_<result_file_suffix>, one row per request, appended. A
        rerun only sends the pairs that have no successful verdict in that file yet.
Usage:  python scripts/llm_benchmarking/run_experiment_oss.py [key1,key2,...]  (optional keys
        restrict the run to those pairs; see LLM_BENCHMARKING.md).
"""

import os
import re
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
import yaml
from openai import OpenAI

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
LLM_SETTINGS = SETTINGS['llm_benchmarking']
LLM_DIR = os.path.join(PROJECT_ROOT, LLM_SETTINGS['llm_benchmarking_dir'])

# Configuration
API_KEY = LLM_SETTINGS['api_key']
BASE_URL = LLM_SETTINGS['base_url']
MODEL_ID = LLM_SETTINGS['model_id']

# Paths
DEFAULT_DATA_FILE = os.path.join(LLM_DIR, LLM_SETTINGS['input_dirname'], LLM_SETTINGS['data_file_name'])
SYSTEM_PROMPT_FILE = os.path.join(LLM_DIR, LLM_SETTINGS['system_prompt_file_name'])
DEFAULT_RESULTS_FILE = os.path.join(
    LLM_DIR, LLM_SETTINGS['results_dirname'],
    f"{MODEL_ID.split('/')[-1]}_{LLM_SETTINGS['result_file_suffix']}",
)

# Row keys, in output-column order. Only those present in the input are written, and the first
# present one identifies a pair (8400-pair set: data_index; 8200/8700: pair_id; 400: pmid).
KEY_CANDIDATES = ('data_index', 'pair_id', 'pmid')
RESULT_COLUMNS = ['model', 'real_model_id', 'verdict', 'confidence', 'rationale',
                  'cost', 'duration', 'status', 'error', 'raw_output']
VALID_VERDICTS = {"SUPPORT", "REJECT", "NEUTRAL"}

# The endpoint writes the Harmony system header itself: a client-sent system message makes it
# drop the developer (instruction) message, so the effort goes through `reasoning_effort`.
REASONING_EFFORT = LLM_SETTINGS['reasoning_effort']
# Concurrency and retries
REQUEST_THREADS = LLM_SETTINGS['n_threads']
RANDOM_SEED = LLM_SETTINGS['random_seed']
MAX_TRIALS = LLM_SETTINGS['max_trials']

# Global lock for file writing
file_lock = threading.Lock()


def get_client():
    timeout = httpx.Timeout(None)
    limits = httpx.Limits(max_connections=5000, max_keepalive_connections=2000)
    http_client = httpx.Client(timeout=timeout, limits=limits, verify=False)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client, max_retries=3)


def extract_json_object(text):
    if not isinstance(text, str):
        return None

    if "```" in text:
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)

    text_clean = re.sub(r",\s*([\}\]])", r"\1", text)

    for match in re.finditer(r"\{", text_clean):
        try:
            candidate = text_clean[match.start():]
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(candidate)
            return obj
        except json.JSONDecodeError:
            continue

    return None


def process_row(client, row, key_columns, system_prompt):
    keys = {k: row[k] for k in key_columns}
    user_content = f"Premise: {row['supp_set']}\n\nHypothesis: {row['conclusion']}"

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "developer", "content": f"# Instructions\n\n{system_prompt}"},
                {"role": "user", "content": user_content}
            ],
            max_tokens=8192,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            seed=RANDOM_SEED,
            reasoning_effort=REASONING_EFFORT,
        )
        duration = time.time() - start_time

        if not response.choices:
            raise ValueError(f"API Error: {getattr(response, 'error', 'No choices returned')}")

        content = response.choices[0].message.content
        result_json = extract_json_object(content)
        if result_json is None:
            result_json = {"verdict": "ERROR", "confidence": "LOW", "rationale": "JSON Decode Error"}

        def sanitize(val):
            if isinstance(val, list):
                if len(val) > 0:
                    val = val[0]
                else:
                    return "ERROR"
            return str(val).upper().strip()

        verdict = sanitize(result_json.get("verdict", "ERROR"))
        confidence = sanitize(result_json.get("confidence", "LOW"))
        if verdict not in VALID_VERDICTS:
            verdict = "ERROR"

        return {
            **keys,
            "model": MODEL_ID,
            "real_model_id": MODEL_ID,
            "verdict": verdict,
            "confidence": confidence,
            "rationale": result_json.get("rationale", ""),
            "cost": 0.0,
            "duration": duration,
            "status": "success",
            "error": "",
            "raw_output": content
        }

    except Exception as e:
        return {
            **keys,
            "model": MODEL_ID,
            "real_model_id": MODEL_ID,
            "verdict": "ERROR",
            "confidence": "LOW",
            "rationale": "",
            "cost": 0.0,
            "duration": time.time() - start_time,
            "status": "error",
            "error": str(e),
            "raw_output": ""
        }


def save_result(result, results_file):
    # LF line endings on every OS, matching the existing results files
    with file_lock:
        file_exists = os.path.exists(results_file)
        df = pd.DataFrame([result])
        df.to_csv(results_file, mode='a', header=not file_exists, index=False, lineterminator='\n')


def load_done_keys(results_file, key_column):
    """Keys that already have a successful (non-ERROR) verdict in `results_file`."""
    if not os.path.exists(results_file):
        return set()
    done_df = pd.read_csv(results_file, dtype=str)
    return set(done_df.loc[done_df['verdict'] != 'ERROR', key_column])


def main(only_keys=None, data_file=DEFAULT_DATA_FILE, results_file=DEFAULT_RESULTS_FILE):
    with open(SYSTEM_PROMPT_FILE, "r", encoding='utf-8') as f:
        system_prompt = f.read()

    print(f"Loading data from {data_file}...")
    original_df = pd.read_csv(data_file, dtype=str)
    key_columns = [c for c in KEY_CANDIDATES if c in original_df.columns]
    key_column = key_columns[0]
    if only_keys is not None:
        unknown = sorted(set(only_keys) - set(original_df[key_column]))
        if unknown:
            raise ValueError(f"{len(unknown)} requested keys are not in {data_file}: {unknown}")
        original_df = original_df[original_df[key_column].isin(only_keys)]
    print(f"Loaded {len(original_df)} samples (key column: {key_column}).")

    # appended rows must line up with the existing header
    expected_columns = key_columns + RESULT_COLUMNS
    if os.path.exists(results_file):
        existing_columns = list(pd.read_csv(results_file, nrows=0).columns)
        if existing_columns != expected_columns:
            raise ValueError(f"{results_file} has columns {existing_columns}, expected {expected_columns}")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    client = get_client()
    trial = 0
    while True:
        done_keys = load_done_keys(results_file, key_column)
        df = original_df[~original_df[key_column].isin(done_keys)]
        if len(df) == 0:
            print("No more samples to process. All samples have successful verdicts.")
            break
        if trial == MAX_TRIALS:
            print(f"Warning: {len(df)} samples still have no successful verdict after {MAX_TRIALS} trials.")
            break
        trial += 1

        print(f"Starting trial {trial} for {len(df)} remaining samples with {REQUEST_THREADS} concurrent requests.")
        if len(df) <= 20:
            print(f"Remaining {key_column}s: {df[key_column].tolist()}")

        with ThreadPoolExecutor(max_workers=REQUEST_THREADS) as executor:
            future_to_row = {executor.submit(process_row, client, row, key_columns, system_prompt): row
                             for _, row in df.iterrows()}
            for i, future in enumerate(as_completed(future_to_row)):
                res = future.result()
                save_result(res, results_file)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} / {len(df)} samples in trial {trial}...")
        print(f"Trial {trial} Completed.")


if __name__ == "__main__":
    main(
        sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] != 'None' else None,
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATA_FILE,
        sys.argv[3] if len(sys.argv) > 3 else DEFAULT_RESULTS_FILE,
    )
