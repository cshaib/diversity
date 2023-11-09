import time
import pandas as pd
import numpy as np

from diversity import compression_ratio, homogenization_score, ngram_diversity_score

# Replace this with the actual path where your datasets are located
dataset_path = "synthetic_datasets/"

# The number of times to repeat each experiment
num_experiments = 10

# Initialize results dictionary
results = {
    'Dataset': [],
    'Function': [],
    'Mean Time': [],
    'Std Dev Time': [],
}

# Define the timing function
def time_function(func, *args):
    times = []
    for _ in range(num_experiments):
        start_time = time.time()
        func(*args)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times), np.std(times)

# Iterate over datasets and functions
for i in range(1, 6):
    dataset_filename = f"dataset_{i}.txt"
    dataset = [x.strip() for x in open(dataset_path + dataset_filename).read().split("\n")][:10]

    # Measure compression_ratio
    cr_mean, cr_std = time_function(compression_ratio, dataset, 'gzip')
    results['Dataset'].append(i)
    results['Function'].append('compression_ratio')
    results['Mean Time'].append(cr_mean)
    results['Std Dev Time'].append(cr_std)

    # Measure homogenization_score with rougel
    hs_mean, hs_std = time_function(homogenization_score, dataset, 'rougel')
    results['Dataset'].append(i)
    results['Function'].append('homogenization_score_rougel')
    results['Mean Time'].append(hs_mean)
    results['Std Dev Time'].append(hs_std)

    # Measure ngram_diversity_score
    nds_mean, nds_std = time_function(ngram_diversity_score, dataset, 4)
    results['Dataset'].append(i)
    results['Function'].append('ngram_diversity_score')
    results['Mean Time'].append(nds_mean)
    results['Std Dev Time'].append(nds_std)

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_csv_path = dataset_path + 'timing_experiments_results.csv'
results_df.to_csv(results_csv_path, index=False)

print(f"Results saved to {results_csv_path}")
