import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "figure_dependency"))

# Setup paths relative to the script location
RESULTS_SAVED_DIR = os.path.join(PROJECT_ROOT, "test", "figure_inputs")
PLOT_SAVE_DIR = os.path.join(PROJECT_ROOT, "test", "plots")
input_file = os.path.join(RESULTS_SAVED_DIR, 'LLM_BioNLI_metrics.csv')


def main():
    # Check if the inputs file exists
    if not os.path.exists(input_file):
        print(f"Error: Could not find inputs file at {input_file}")
        return

    # Read data
    try:
        # Based on inspection, the file is tab-separated
        df = pd.read_csv(input_file, sep='\t')
        if len(df.columns) == 1:
            # Fallback to comma separation
            df = pd.read_csv(input_file, sep=',')
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return
        
    # Standardize column names by stripping leading/trailing whitespace
    df.columns = df.columns.str.strip()
    
    # Calculate F1 Score
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    if 'Precision' in df.columns and 'Recall' in df.columns:
        df['F1 Score'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'])
    else:
        print("Error: 'Precision' and/or 'Recall' columns missing for F1 Score calculation.")
        return

    # Set Model ID as index for easier plotting
    if 'Model ID' in df.columns:
        df['Model ID'] = df['Model ID'].replace({
            'gemini-3-pro-preview-11-2025-high': 'gemini-3-pro',
            'gpt-oss-120b-high-reasoning': 'gpt-oss-120b',
            'deepseek-v3.2-thinking': 'deepseek-v3.2'
        })
        df.set_index('Model ID', inplace=True)
    else:
        print("Warning: 'Model ID' column not found. Using the default index instead.")

    # Colors defined by user (RGB values mapped to 0-1 scale for matplotlib)
    # Accuracy: (16, 85, 154)
    # Precision: (64, 176, 166)
    # Recall: (81, 40, 136)
    # F1 Score: (219, 16, 72)
    colors = {
        'Accuracy': (16/255, 85/255, 154/255),
        'Precision': (64/255, 176/255, 166/255),
        'Recall': (81/255, 40/255, 136/255),
        'F1 Score': (219/255, 16/255, 72/255)
    }

    # Specify metrics to plot
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Check if all targeted metrics are in the dataframe
    missing_metrics = [m for m in metrics_to_plot if m not in df.columns]
    if missing_metrics:
        print(f"Warning: Discovered missing metrics to plot: {missing_metrics}")
        metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    # Map colors to the available metrics
    plot_colors = [colors[m] for m in metrics_to_plot]

    # Create the figure and axis (< 4 inches width)
    fig, ax = plt.subplots(figsize=(4, 5))

    # Draw grouped bar plot
    df[metrics_to_plot].plot(
        kind='bar',
        ax=ax,
        color=plot_colors,
        width=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    # Customize the plot
    ax.set_xlabel('Model ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
    
    # Configure x-ticks
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)
    
    # Configure y-axis limits to clarify differences and leave room for legend
    ax.set_ylim(0.8, 1.0)

    # Add values above each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=2, fontsize=7.5, rotation=60)

    # Configure legend
    ax.legend(title='Metrics', title_fontsize='9', fontsize='8', 
              loc='best')

    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent cutting off labels or legends
    plt.tight_layout()

    # Create outputs directory and save figure
    output_dir = PLOT_SAVE_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'LLM_metrics_bar_plot.svg')
    
    plt.savefig(output_path, dpi=500, bbox_inches='tight', transparent=True)
    print(f"Plot saved successfully to {output_path}")

    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()
