import os
import time
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import httpx
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY") # your API key
BASE_URL = os.getenv("LLM_BASE_URL") # your API call base url
MODEL_ID = os.getenv("LLM_MODEL_ID", "openai/gpt-oss-120b") # your model id

if not API_KEY or not BASE_URL:
    raise ValueError("OPENAI_API_KEY and LLM_BASE_URL must be set in your .env file")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'test', 'inputs', "bionli_augmented_400.csv")
SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, "scripts", "system_prompt.txt")
RESULTS_DIR = os.path.join(BASE_DIR, "test", "BioNLI_output") # your workspace
ALREADY_DONE_RESULTS_FILE = os.path.join(RESULTS_DIR, "oss_all_predictions_400.csv") # initlal filename

# Harmony format constants
HARMONY_SYSTEM_PROMPT = """You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message."""
# Concurrency
REQUEST_THREADS = 512 # your concurrency
# N_SAMPLES = 10
RANDOM_SEED = 20260310 # your random seed

# Load System Prompt
with open(SYSTEM_PROMPT_FILE, "r", encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()

# Global lock for file writing
file_lock = threading.Lock()


def get_client():
    timeout = httpx.Timeout(None)
    limits = httpx.Limits(max_connections=5000, max_keepalive_connections=2000)
    http_client = httpx.Client(timeout=timeout, limits=limits, verify=False)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client, max_retries=3)


def extract_json_object(text):
    import re
    import json
    
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


def process_row(client, row):
    pmid = row['pmid']
    pair_id = row['pair_id']
    data_index = row['data_index']
    supp_set = row['supp_set']
    conclusion = row['conclusion']
    user_content = f"Premise: {supp_set}\n\nHypothesis: {conclusion}"

    start_time = time.time()
    try:
        if MODEL_ID == "openai/gpt-oss-120b":
            message_content = [
                {"role": "system", "content": HARMONY_SYSTEM_PROMPT},
                {"role": "developer", "content": f"# Instructions\n\n{SYSTEM_PROMPT}"},
                {"role": "user", "content": user_content}
            ]
        else:
            message_content = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        # openAi model needs HARMONY style system prompt.
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=message_content,
            max_tokens=8192,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            extra_body={
                "reasoning": {"enabled": True},
            }, 
        )
        duration = time.time() - start_time
        
        if not response.choices:
            raise ValueError(f"API Error: {getattr(response, 'error', 'No choices returned')}")
            
        content = response.choices[0].message.content
        try:
            result_json = extract_json_object(content)
            if result_json is None:
                 raise ValueError("Could not extract JSON")
        except Exception:
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
        
        VALID_VERDICTS = {"SUPPORT", "REJECT", "NEUTRAL"}
        if verdict not in VALID_VERDICTS:
            verdict = "ERROR"

        return {
                "data_index": data_index,
                "pair_id": pair_id,
                "pmid": pmid,
                "model": MODEL_ID,
                "real_model_id": MODEL_ID,
                "verdict": verdict,
                "confidence": confidence,
                "rationale": result_json.get("rationale", ""),
                "cost": 0.0 if MODEL_ID == 'openai/gpt-oss-120b' else response.usage.cost,
                "duration": duration,
                "status": "success",
                "error": "",
                "raw_output": content
            }

    except Exception as e:
        return {
            "data_index": data_index,
            "pair_id": pair_id,
            "pmid": pmid,
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


def save_result(result, RESULTS_FILE):
    with file_lock:
        file_exists = os.path.exists(RESULTS_FILE)
        df = pd.DataFrame([result])
        df.to_csv(RESULTS_FILE, mode='a', header=not file_exists, index=False)


def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    print(f"Loading data from {DATA_FILE}...")
    original_df = pd.read_csv(DATA_FILE, header=0)
    print(f"Loaded {len(original_df)} samples.")
    client = get_client()
    trial = 1
    while True:
        # Determine output file. Initial run vs retry.
        csv_files_in_dir = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
        if not csv_files_in_dir:
            print("No previous runs found. This is the initial run.")
            RESULTS_FILE = ALREADY_DONE_RESULTS_FILE
        else:
            RESULTS_FILE = os.path.join(RESULTS_DIR, f"oss_error_corrected_8400_trial_{trial}.csv")
            while os.path.exists(RESULTS_FILE) or RESULTS_FILE == ALREADY_DONE_RESULTS_FILE:
                # Need to check `== ALREADY_DONE_RESULTS_FILE` just in case a filename collision happens
                trial += 1
                RESULTS_FILE = os.path.join(RESULTS_DIR, f"oss_error_corrected_8400_trial_{trial}.csv")
            
        all_done_dfs = []
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith(".csv"):
                filepath = os.path.join(RESULTS_DIR, filename)
                try:
                    all_done_dfs.append(pd.read_csv(filepath))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

        df = original_df.copy()
        if all_done_dfs:
            print(f"Loading previous results from {RESULTS_DIR}...")
            done_df = pd.concat(all_done_dfs, ignore_index=True)
            if 'verdict' in done_df.columns and 'pair_id' in done_df.columns:
                successful_pair_ids = set(done_df[done_df['verdict'] != 'ERROR']['pair_id'].unique())
                df = df[~df['pair_id'].isin(successful_pair_ids)]
                print(f"Filtered down to {len(df)} samples that had ERROR verdicts or were unprocessed.")
            else:
                print("Warning: 'verdict' or 'pair_id' column not found in previous results.")
        else:
            print(f"No previous results found in {RESULTS_DIR}, processing all data...")

        if len(df) == 0:
            print("No more samples to process. All samples have successful verdicts.")
            break
            
        print(f"Starting trial {trial} for {len(df)} remaining samples with {REQUEST_THREADS} concurrent requests.")
        
        with ThreadPoolExecutor(max_workers=REQUEST_THREADS) as executor:
            future_to_row = {executor.submit(process_row, client, row): row for _, row in df.iterrows()}
            for i, future in enumerate(as_completed(future_to_row)):
                res = future.result()
                save_result(res, RESULTS_FILE)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} / {len(df)} samples in trial {trial}...")
        print(f"Trial {trial} Completed.")
        trial += 1


if __name__ == "__main__":
    main()