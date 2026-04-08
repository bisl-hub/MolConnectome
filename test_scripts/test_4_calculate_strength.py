import os
import subprocess
import sys

def main():
    python_executable = sys.executable
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    script_path = os.path.join(project_root, 'scripts', 'calculate_strength.py')
    gene_set_path = os.path.join(project_root, "test", 'inputs', "CURATED_GENE_SET.pickle")
    quantile_files_path = os.path.join(project_root, "test_scripts", "test_4_dirs_list.txt")

    args = [
        python_executable,
        script_path,
        "--task-name", "test_AC_CN_NAc_strength",
        "--result-save-dir", os.path.join(project_root, "test", "strength"),
        "--gene-set-dir", gene_set_path,
        "--expression-files", os.path.join(project_root, "test_scripts", "test_4_expressions_list.csv"),
        "--quantile-files", quantile_files_path,
        "--significance-threshold", "0.05",
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
