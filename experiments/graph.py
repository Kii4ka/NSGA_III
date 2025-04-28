import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV
df = pd.read_csv('nsga_performance_multisizes.csv')

# For execution time bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Implementation', y='Time (ms)', data=df)
plt.title('NSGA-III Execution Time (Lower is Better)')
plt.yscale('log')  # Log scale recommended due to large differences
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('execution_time.png')