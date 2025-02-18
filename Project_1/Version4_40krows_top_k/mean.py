import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
file_path_k_25 = r'C:\Users\disch\Desktop\CiteSert\Project_1\Version4_40krows_top_k\input_data\k_percent_performance_results.csv'
df_k_25 = pd.read_csv(file_path_k_25)

# Calculate the new metric
df_k_25['combined_metric'] = (df_k_25['mean_of_mean_performance'] - 0.0004) * df_k_25['number_of_monthly_predictions']

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df_k_25['k_percent'], df_k_25['combined_metric'], marker='o')
plt.title('Combined Metric: (Mean Performance - 0.0004) * Avg Monthly Predictions')
plt.xlabel('K Percent')
plt.ylabel('Combined Metric')
plt.grid(True)
plt.tight_layout()
plt.savefig('input_data/combined_metric_plot.png')
plt.show()

# Save the figure
plt.close()

# Print the combined metric values
print(df_k_25[['k_percent', 'combined_metric']])