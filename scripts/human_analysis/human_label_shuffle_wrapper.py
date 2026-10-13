"""Human stage 1 driver: the real run plus N diagnosis-label shuffles of stages 1a-1c.

Input:  human expression and sample-matching table (through the three stage-1 scripts).
Output: human_analysis_results/unshuffled/ and human_analysis_results/<JOB>/run_NNNNNN/
Usage:  see HUMAN_PIPELINE.md, section 4.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # single-threaded BLAS; must precede the first numpy import
import sys
import subprocess
from pathlib import Path
import yaml

with open(Path(__file__).resolve().parent.parent.parent / 'settings.yaml', encoding='utf-8') as _f:
    SETTINGS = yaml.safe_load(_f)

PROJECT_ROOT = Path(SETTINGS['project_directory'])
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts', 'src'))

from src.concurrent_multiprocessing import run_process_pool
import human_oneshot
import human_abundance

QUANTILE_SAVE_PATH = PROJECT_ROOT / SETTINGS['analysis_results']['hsa_analysis_save_dir']
SCRIPTS_PATH = PROJECT_ROOT / 'scripts' / 'human_analysis'

JOB_NAME = SETTINGS['human_label_shuffle_wrapper']['initial_job_name']
# run_process_pool's LOG_DIR param is actually a single appended-to log FILE
# path, not a directory -- named to match that function's own parameter. It
# now lives inside the job's own directory (hsa_analysis_save_dir/JOB_NAME/),
# not prefixed onto a flat filename directly under hsa_analysis_save_dir/.
DEFAULT_LOG_FILE = os.path.join(QUANTILE_SAVE_PATH, JOB_NAME,
                                SETTINGS['human_label_shuffle_wrapper']['log_filename'])
# Holds the one real, non-shuffled baseline run -- separate from JOB_NAME,
# which is only for the label-shuffle replicates.
DEFAULT_UNSHUFFLED_DIR = os.path.join(QUANTILE_SAVE_PATH,
                                      SETTINGS['human_label_shuffle_wrapper']['unshuffled_dir_name'])
N_LABEL_SHUFFLES = SETTINGS['human_label_shuffle_wrapper']['n_label_shuffles']
# abundance / weighted connectivity pools poll at this interval; oneshot (the longest) at 3x
PROCESS_WAIT_INTERVAL = SETTINGS['human_label_shuffle_wrapper']['process_wait_interval']

PYTHON_INTERPRETER = SETTINGS['python_interpreter']
HUMAN_ONESHOT_PATH = os.path.join(SCRIPTS_PATH, 'human_oneshot.py')
HUMAN_ABUNDANCE_PATH = os.path.join(SCRIPTS_PATH, 'human_abundance.py')
HUMAN_WEIGHTED_CONNECTIVITY_PATH = os.path.join(SCRIPTS_PATH, 'human_weighted_connectivity.py')


def _run_initial_unshuffled(background_source: str, p_value_method: str, unshuffled_dir: str, log_file: str,
                            normalize_weights: bool = False) -> None:
    """The one real, non-shuffled baseline (label_shuffle_seed=None) -- a
    single job, so no run_process_pool/concurrency bookkeeping and no
    n_workers/thread arg passed to human_oneshot.py (it falls back to its
    own default internal region-pair parallelism, which is fine here since
    nothing else is competing for cores during this phase)."""
    with open(log_file, 'a', encoding='utf-8', buffering=1) as f:
        f.write(f'Starting initial (unshuffled) run.\n')
        f.write(f'Output directory: {unshuffled_dir}\n')
        f.write(f'P-value method: {p_value_method}\n')
        f.write(f'Background source: {background_source}\n')
        f.write(f'\n\n')
        f.write(f'------------------------------\n\n')

    subprocess.run([PYTHON_INTERPRETER, HUMAN_ABUNDANCE_PATH, unshuffled_dir, 'None'], check=True)
    subprocess.run(
        [PYTHON_INTERPRETER, HUMAN_ONESHOT_PATH, background_source, unshuffled_dir, p_value_method, 'None'],
        check=True,
    )
    signaling_ctl_path = human_oneshot.output_path(unshuffled_dir, background_source, p_value_method, None, 'CTL')
    signaling_scz_path = human_oneshot.output_path(unshuffled_dir, background_source, p_value_method, None, 'SCZ')
    abundance_ctl_path, abundance_scz_path = human_abundance._abundance_output_paths(unshuffled_dir)
    subprocess.run(
        [PYTHON_INTERPRETER, HUMAN_WEIGHTED_CONNECTIVITY_PATH,
         signaling_ctl_path, signaling_scz_path,
         abundance_ctl_path, abundance_scz_path,
         unshuffled_dir, background_source, p_value_method, 'None', str(normalize_weights)],
        check=True,
    )

    with open(log_file, 'a', encoding='utf-8', buffering=1) as f:
        f.write(f'Finished initial (unshuffled) run.\n\n')
        f.write(f'------------------------------\n\n')


def main(background_source: str = human_oneshot.DEFAULT_BACKGROUND_SOURCE,
         output_dir: str = str(QUANTILE_SAVE_PATH),
         p_value_method: str = human_oneshot.DEFAULT_P_VALUE_METHOD,
         unshuffled_dir: str = DEFAULT_UNSHUFFLED_DIR,
         n_replicates: int = N_LABEL_SHUFFLES,
         seed_start: int = 1,
         max_concurrent_processes: int = None,
         log_file: str = DEFAULT_LOG_FILE,
         normalize_weights: bool = False):
    """Drives the human pipeline (abundance -> oneshot -> weighted
    connectivity) twice over: once for the real, unshuffled baseline (under
    `unshuffled_dir`, no concurrency needed -- it's a single job), then
    `n_replicates` independent disease-label-shuffle repetitions, one OS
    subprocess per label_shuffle_seed in [seed_start, seed_start +
    n_replicates), each replicate getting its own directory under
    `output_dir`/JOB_NAME (JOB_NAME from settings.yaml
    human_label_shuffle_wrapper.initial_job_name).

    Each human_oneshot.py call only has 10 region pairs * 2 diseases = 20
    work items (vs macaque's 105), so -- unlike macaque_oneshot.py, which
    parallelizes region pairs *within* one call across many cores -- the
    parallelism for the shuffled replicates is inverted: many concurrent
    human_oneshot.py subprocesses (up to `max_concurrent_processes`), each
    internally single-threaded (n_workers=1), so cpu cores aren't
    oversubscribed by process-pool-inside-process-pool.
    """
    if max_concurrent_processes is None:
        # settings.yaml analysis_global_settings.max_concurrency; null -> 1
        configured_max_concurrency = SETTINGS['analysis_global_settings'].get('max_concurrency')
        max_concurrent_processes = 1 if configured_max_concurrency is None else int(configured_max_concurrency)
    if max_concurrent_processes < 1:
        raise ValueError(
            f'max_concurrent_processes must be >= 1, got {max_concurrent_processes} '
            f'(check settings.yaml analysis_global_settings.max_concurrency, or the '
            f'max_concurrent_processes argument) -- 0 or negative silently hangs '
            f'run_process_pool forever, since it never starts a process to drain the queue.'
        )

    job_dir = os.path.join(output_dir, JOB_NAME)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Phase 0: the real, unshuffled baseline -- single job, no pooling.
    _run_initial_unshuffled(background_source, p_value_method, unshuffled_dir, log_file, normalize_weights)

    def replicate_dir(seed: int) -> str:
        return os.path.join(job_dir, f'run_{seed:06d}')

    # First log for abundance pipeline.
    with open(log_file, 'a', encoding='utf-8', buffering=1) as f:
        f.write(f'Starting human abundance commands.\n')
        f.write(f'Total number of sample label shuffle: {n_replicates}\n')
        f.write(f'Seed start: {seed_start}\n')
        f.write(f'Max concurrent processes: {max_concurrent_processes}\n')
        f.write(f'Job directory: {job_dir}\n')
        f.write(f'\n\n')
        f.write(f'------------------------------\n\n')

    # Human abundance commands.
    abundance_commands = [
        [PYTHON_INTERPRETER, HUMAN_ABUNDANCE_PATH, replicate_dir(seed), str(seed)]
        for seed in range(seed_start, seed_start + n_replicates)
    ]
    run_process_pool(abundance_commands, max_concurrent_processes, log_file,
                     process_wait_interval=PROCESS_WAIT_INTERVAL)

    # Then log for one-shot pipeline.
    with open(log_file, 'a', encoding='utf-8', buffering=1) as f:
        f.write(f'------------------------------\n\n')
        f.write(f'Starting human one-shot commands.\n')
        f.write(f'Total number of sample label shuffle: {n_replicates}\n')
        f.write(f'Seed start: {seed_start}\n')
        f.write(f'Max concurrent processes: {max_concurrent_processes}\n')
        f.write(f'Job directory: {job_dir}\n')
        f.write(f'P-value method: {p_value_method}\n')
        f.write(f'Background source: {background_source}\n')
        f.write(f'\n\n')
        f.write(f'------------------------------\n\n')

    # Human oneshot commands.
    oneshot_commands = [
        [PYTHON_INTERPRETER, HUMAN_ONESHOT_PATH, background_source, replicate_dir(seed), p_value_method, str(seed), '1']
        for seed in range(seed_start, seed_start + n_replicates)
    ]
    run_process_pool(oneshot_commands, max_concurrent_processes, log_file,
                     process_wait_interval=3 * PROCESS_WAIT_INTERVAL)

    # Then log for weighted connectivity pipeline.
    with open(log_file, 'a', encoding='utf-8', buffering=1) as f:
        f.write(f'------------------------------\n\n')
        f.write(f'Starting human weighted connectivity commands.\n')
        f.write(f'Total number of sample label shuffle: {n_replicates}\n')
        f.write(f'Seed start: {seed_start}\n')
        f.write(f'Max concurrent processes: {max_concurrent_processes}\n')
        f.write(f'Job directory: {job_dir}\n')
        f.write(f'\n\n')
        f.write(f'------------------------------\n\n')

    # Human weighted connectivity commands. human_weighted_connectivity.py
    # takes its oneshot/abundance inputs as explicit file paths (not a bare
    # seed) -- by this point in the pipeline both prerequisite stages above
    # have already written this seed's worth of output to disk, so we just
    # build the matching paths here the same way human_oneshot.output_path()
    # / human_abundance._abundance_output_paths() do internally.
    weighted_connectivity_commands = []
    for seed in range(seed_start, seed_start + n_replicates):
        this_replicate_dir = replicate_dir(seed)
        signaling_ctl_path = human_oneshot.output_path(
            this_replicate_dir, background_source, p_value_method, seed, 'CTL')
        signaling_scz_path = human_oneshot.output_path(
            this_replicate_dir, background_source, p_value_method, seed, 'SCZ')
        abundance_ctl_path, abundance_scz_path = human_abundance._abundance_output_paths(this_replicate_dir)
        weighted_connectivity_commands.append([
            PYTHON_INTERPRETER, HUMAN_WEIGHTED_CONNECTIVITY_PATH,
            signaling_ctl_path, signaling_scz_path,
            abundance_ctl_path, abundance_scz_path,
            this_replicate_dir, background_source, p_value_method, str(seed), str(normalize_weights),
        ])
    run_process_pool(weighted_connectivity_commands, max_concurrent_processes, log_file,
                     process_wait_interval=PROCESS_WAIT_INTERVAL)

if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else human_oneshot.DEFAULT_BACKGROUND_SOURCE,
        sys.argv[2] if len(sys.argv) > 2 else str(QUANTILE_SAVE_PATH),
        sys.argv[3] if len(sys.argv) > 3 else human_oneshot.DEFAULT_P_VALUE_METHOD,
        sys.argv[4] if len(sys.argv) > 4 else DEFAULT_UNSHUFFLED_DIR,
        int(sys.argv[5]) if len(sys.argv) > 5 else N_LABEL_SHUFFLES, # total number of sample label shuffle
        int(sys.argv[6]) if len(sys.argv) > 6 else 1,  # seed start
        int(sys.argv[7]) if len(sys.argv) > 7 else None,  # max concurrent process
        sys.argv[8] if len(sys.argv) > 8 else DEFAULT_LOG_FILE,  # log path
        (sys.argv[9] == 'True') if len(sys.argv) > 9 else False,  # normalize weights by maximum
    )
