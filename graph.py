import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

# Redirect stderr to suppress macOS warnings when saving
def save_figure_without_warnings(fig, filename, **kwargs):
    """Save figure while suppressing macOS warnings"""
    # Store original stderr
    original_stderr = sys.stderr
    
    try:
        # Redirect stderr to null device
        with open(os.devnull, 'w') as null:
            sys.stderr = null
            fig.savefig(filename, **kwargs)
    finally:
        # Restore stderr
        sys.stderr = original_stderr

# Load the performance data
df = pd.read_csv('nsga_performance.csv')

# Set a nice style
plt.style.use('ggplot')
sns.set_palette("Set2")

# Create a single tall figure for the bar charts
fig = plt.figure(figsize=(10, 20))

# 1. Execution Time Comparison (log scale)
ax1 = fig.add_subplot(411)  # 4 rows, 1 column, position 1
bars = ax1.bar(df['Implementation'], df['Time (ms)'])
ax1.set_title('Execution Time (ms) - Log Scale', fontsize=14)
ax1.set_ylabel('Time (ms)')
ax1.set_yscale('log')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.2f}',
             ha='center', va='bottom', rotation=0, fontsize=10)

# 2. Memory Usage Comparison
ax2 = fig.add_subplot(412)  # 4 rows, 1 column, position 2
bars = ax2.bar(df['Implementation'], df['Memory Usage (MB)'])
ax2.set_title('Memory Usage (MB)', fontsize=14)
ax2.set_ylabel('Memory (MB)')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', rotation=0, fontsize=10)

# 3. Throughput Comparison
ax3 = fig.add_subplot(413)  # 4 rows, 1 column, position 3
bars = ax3.bar(df['Implementation'], df['Throughput (evals/s)'])
ax3.set_title('Throughput (evaluations/second)', fontsize=12)
ax3.set_ylabel('Throughput')
ax3.set_yscale('log')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.2f}',
             ha='center', va='bottom', rotation=0, fontsize=10)

# 4. Speedup Visualization
ax4 = fig.add_subplot(414)  # 4 rows, 1 column, position 4
speedups = df['Speedup'].str.replace('x', '').astype(float)
bars = ax4.bar(df['Implementation'], speedups)
ax4.set_title('Speedup Factor (higher is better)', fontsize=14)
ax4.set_ylabel('Speedup (x)')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}x',
             ha='center', va='bottom', rotation=0, fontsize=10)

# Save the main vertical figure using the custom function
plt.tight_layout(pad=3.0)  # Add padding between subplots
save_figure_without_warnings(fig, 'nsga_performance_bars.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a separate figure for GPU Time Breakdown
gpu_row = df[df['Implementation'] == 'GPU MPS'].iloc[0]
if not pd.isna(gpu_row['Memory Transfer (ms)']):
    # Create a new figure for the pie chart
    pie_fig = plt.figure(figsize=(10, 10))
    gpu_time = float(gpu_row['Time (ms)'])
    transfer_time = float(gpu_row['Memory Transfer (ms)'])
    compute_time = gpu_time - transfer_time
    
    # Create pie chart
    plt.pie([transfer_time, compute_time], 
            labels=['Memory Transfer', 'Computation'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.1, 0),
            shadow=True,
            textprops={'fontsize': 14})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('GPU Time Breakdown', fontsize=18)
    
    # Add details as text
    plt.figtext(0.5, 0.01, 
                f"Total GPU time: {gpu_time:.2f} ms\n"
                f"Memory transfer: {transfer_time:.2f} ms\n"
                f"Computation: {compute_time:.2f} ms",
                ha="center", fontsize=14)
    
    # Save the pie chart figure using the custom function
    save_figure_without_warnings(pie_fig, 'gpu_time_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()