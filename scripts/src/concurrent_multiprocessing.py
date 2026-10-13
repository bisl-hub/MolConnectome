"""Runs a list of commands as a pool of concurrent subprocesses.

Used by: human_label_shuffle_wrapper.py.
"""

import os
import subprocess
import time
import datetime

DEFAULT_WAIT_INTERVAL = 60 # seconds

def run_process_pool(COMMANDS_TO_RUN: 'list[list[str]]',
                     MAX_CONCURRENT_PROCESSES: int,
                     LOG_DIR: str,
                     process_wait_interval: int = DEFAULT_WAIT_INTERVAL,
                     ):
    """
    Manages a pool of concurrent processes.

    Args:
        COMMANDS_TO_RUN: List of commands to run.
        MAX_CONCURRENT_PROCESSES: Maximum number of concurrent processes.
        LOG_DIR: Path of the log FILE to append progress to (not a directory).
        process_wait_interval: Time to wait between checking for finished processes.
    """
    active_processes = []
    command_queue = list(COMMANDS_TO_RUN)
    print(f"Starting process pool with a max of {MAX_CONCURRENT_PROCESSES} concurrent processes.")

    if not process_wait_interval:
        process_wait_interval = DEFAULT_WAIT_INTERVAL
    # Keep a single, line-buffered handle open for the whole run instead of
    # opening/closing the log file on every process start.
    if LOG_DIR:
        log_parent_dir = os.path.dirname(LOG_DIR)
        if log_parent_dir:
            os.makedirs(log_parent_dir, exist_ok=True)
    lfile = open(LOG_DIR, 'a', buffering=1) if LOG_DIR else None

    def log(message: str):
        if lfile:
            lfile.write(message + '\n')

    total_count = len(COMMANDS_TO_RUN)

    def log_progress():
        completed_count = total_count - len(command_queue) - len(active_processes)
        percent = (completed_count / total_count * 100) if total_count else 100.0
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        log(f"{completed_count} out of {total_count} ({percent:.1f}%) processed until {timestamp}")

    # The polling loop below refreshes every 60s regardless; only gate how
    # often that progress gets written to the log (once every 10 minutes).
    # LOG_INTERVAL_SECONDS = 10 * process_wait_interval
    LOG_INTERVAL_SECONDS = 10 * max(process_wait_interval, 60) # for convenient log tracking.

    last_log_time = time.time()

    try:
        # Loop as long as there are commands to run or processes still running.
        while command_queue or active_processes:
            # 1. Clean up finished processes
            # Iterate over a copy of the list because we modify it in the loop.
            for process_info in active_processes[:]:
                process = process_info["process"]
                if process.poll() is not None:  # The process has terminated
                    # print(f"'{" ".join(process_info['command'])}' (PID: {process.pid}) finished with exit code {process.returncode}.")
                    active_processes.remove(process_info)
            # 2. Start new processes if there is capacity
            while len(active_processes) < MAX_CONCURRENT_PROCESSES and command_queue:
                command_to_start = command_queue.pop(0)
                try:
                    # Start a new process.
                    # stdout and stderr are redirected to DEVNULL to keep the manager's output clean.
                    # You can change this to subprocess.PIPE to capture output if needed.
                    process = subprocess.Popen(command_to_start, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    process_info = {
                        "process": process,
                        "command": command_to_start
                    }
                    active_processes.append(process_info)
                    # print(f"Starting '{" ".join(command_to_start)}' (PID: {process.pid})...")
                except FileNotFoundError:
                    raise FileNotFoundError
                    # print(f"Error: Command not found for '{" ".join(command_to_start)}'. Make sure Python is in your PATH.")
                except Exception as e:
                    raise ValueError(e)
                    # print(f"Failed to start command '{" ".join(command_to_start)}': {e}")
            # Sleep for a short duration to prevent the loop from consuming too much CPU.
            time.sleep(process_wait_interval)
            if time.time() - last_log_time >= LOG_INTERVAL_SECONDS:
                log_progress()
                last_log_time = time.time()
            print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
            print(len(command_queue), 'out of', len(COMMANDS_TO_RUN), 'commands left...',
                   '(', len(active_processes), 'running)' )
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Terminating all active processes...")
        for process_info in active_processes:
            process_info["process"].terminate() # or .kill()
        # Wait for them to terminate
        for process_info in active_processes:
            process_info["process"].wait()
        print("All processes terminated.")
    finally:
        log_progress()
        if lfile:
            lfile.close()
        print("\nProcess pool has finished all tasks.")
