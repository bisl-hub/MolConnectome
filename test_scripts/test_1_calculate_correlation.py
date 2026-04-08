import os
import subprocess
import sys

def main():
    python_executable = sys.executable
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    script_path = os.path.join(project_root, 'scripts', 'calculate_correlation.py')
    genes_to_analyze_dir = os.path.join(project_root, 'test', 'inputs', 'FINAL_GENES.npy')
    
    args = [
        python_executable,
        script_path,
        "--task-name", "test_AC_CN_NAc_corr",
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

    print("Running test for calculate_correlation.py with the following arguments:")
    for i, arg in enumerate(args[2:], 1):
        print(f"Arg {i}: {arg}")
    
    print("\nExecuting command:")
    print(" ".join(args))
    print("-" * 50)
    
    # Run the calculate_correlation.py script
    subprocess.run(args, cwd=project_root)

if __name__ == "__main__":
    main()
