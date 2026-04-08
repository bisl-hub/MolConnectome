import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_task(k, python_executable, script_path, project_root):
    genes_to_analyze_dir = os.path.join(project_root, 'test', 'inputs', f'NULL_GENES_{k}.npy')
    task_name = f"test_AC_CN_NAc_background_{k}"
    
    args = [
        python_executable,
        script_path,
        "--task-name", task_name,
        "--result-save-dir", os.path.join(project_root, "test", "correlation"),
        "--gene-list-dir", genes_to_analyze_dir,
        "--correlation-metric", "pcc",
        "--correlation-alpha", "0.05",
        "--n-perm-correlation", "1000",
        "--region-1-control-data-dir", os.path.join(project_root, "test", "inputs", "AC_CN_control.csv"),
        "--region-2-control-data-dir", os.path.join(project_root, "test", "inputs", "AC_NAc_control.csv"),
        "--region-1-schizo-data-dir", os.path.join(project_root, "test", "inputs", "AC_CN_schizo.csv"),
        "--region-2-schizo-data-dir", os.path.join(project_root, "test", "inputs", "AC_NAc_schizo.csv")
    ]
    
    print(f"[{task_name}] Executing...")
    # Run the calculate_correlation.py script
    return subprocess.run(args, cwd=project_root, capture_output=True, text=True)

def main():
    python_executable = sys.executable
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    script_path = os.path.join(project_root, 'scripts', 'calculate_correlation.py')
    
    N_CONCURRENT_PROCESSES = 5
    total_k = 10
    
    print(f"Running {total_k} background tasks with {N_CONCURRENT_PROCESSES} concurrent processes...")
    
    with ProcessPoolExecutor(max_workers=N_CONCURRENT_PROCESSES) as executor:
        futures = {
            executor.submit(run_task, k, python_executable, script_path, project_root): k
            for k in range(1, total_k + 1)
        }
        
        for future in as_completed(futures):
            k = futures[future]
            try:
                result = future.result()
                if result.returncode == 0:
                    print(f"[test_AC_CN_NAc_background_{k}] Completed successfully.")
                else:
                    print(f"[test_AC_CN_NAc_background_{k}] FAILED with exit code {result.returncode}")
                    print(result.stderr)
            except Exception as exc:
                print(f"[test_AC_CN_NAc_background_{k}] Generated an exception: {exc}")

if __name__ == "__main__":
    main()
