import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import numpy as np
import json
import matplotlib.pyplot as plt

# def plot_boxplots(CCPS_H, CCPS_B, HMP, Bonf, alpha):
#     fig, ax = plt.subplots()
#     CCPS_H = [len(item) for item in CCPS_H]
#     CCPS_B = [len(item) for item in CCPS_B]
#     HMP = [len(item) for item in HMP]
#     Bonf = [len(item) for item in Bonf]

#     # Create a list of data for each class
#     data = [CCPS_H, CCPS_B, HMP, Bonf]

#     # Plot the boxplots
#     ax.boxplot(data)

#     # Customize the plot
#     ax.set_xticklabels(['CCPS_H', 'CCPS_B', 'HMP', 'Bonf'], fontsize=14)
#     ax.set_ylabel('Unique Answers', fontsize=14)
#     ax.set_xlabel('Methods', fontsize=14)
#     ax.set_yticklabels(ax.get_yticks(), fontsize=13)
#     # ax.set_title('Boxplots for Two Classes')

#     # Show the plot
#     plt.tight_layout()
#     plt.savefig(f'QA_boxplot_{alpha}.png')

# alpha = 0.05
# with open(f'HMP_unique_answer_sets_baseline_{alpha}.json', 'r') as f:
#     HMP_baseline = json.load(f)
# with open(f'Bonf_unique_answer_sets_baseline_{alpha}.json', 'r') as f:
#     Bonf_baseline = json.load(f)
# with open(f'HMP_unique_answer_sets_weighted_{alpha}.json', 'r') as f:
#     HMP_weighted = json.load(f)
# with open(f'Bonf_unique_answer_sets_weighted_{alpha}.json', 'r') as f:
#     Bonf_weighted = json.load(f)

# plot_boxplots(HMP_weighted, Bonf_weighted, HMP_baseline, Bonf_baseline, alpha)



baseline_coverage = r"Baseline coverage\s(\d+\.\d+)"
baseline_average_request = r"Baseline average request\s(\d+\.\d+)"
baseline_unique_answer_sets = r"Baseline unique answer sets\s(\d+\.\d+)"
baseline_total_answer_sets = r"Baseline total answer sets\s(\d+\.\d+)"
baseline_semantic_clusters = r"Baseline semantic clusters\s(\d+\.\d+)"
best_parameters = r"Best parameters: \[([\d\.]+) ([\d\.]+)\]"

opt_coverage = r"Opt coverage\s(\d+\.\d+)"
opt_average_request = r"Opt average request\s(\d+\.\d+)"
opt_unique_answer_sets = r"Opt unique answer sets\s(\d+\.\d+)"
opt_total_answer_sets = r"Opt total answer sets\s(\d+\.\d+)"
opt_semantic_clusters = r"Opt semantic clusters\s(\d+\.\d+)"


with open('log_e2e_Bonf_rougeL', 'r') as file:
    content = file.read()
baseline_coverage_values = re.findall(baseline_coverage, content)
baseline_average_request_values = re.findall(baseline_average_request, content)
baseline_unique_answer_sets_values = re.findall(baseline_unique_answer_sets, content)
baseline_total_answer_sets_values = re.findall(baseline_total_answer_sets, content)
baseline_semantic_clusters_values = re.findall(baseline_semantic_clusters, content)

opt_coverage_values = re.findall(opt_coverage, content)
opt_average_request_values = re.findall(opt_average_request, content)
opt_unique_answer_sets_values = re.findall(opt_unique_answer_sets, content)
opt_total_answer_sets_values = re.findall(opt_total_answer_sets, content)
opt_semantic_clusters_values = re.findall(opt_semantic_clusters, content)
match = re.findall(best_parameters, content)

print("----Bonf----")
print("Baseline coverage", np.mean([float(val) for val in baseline_coverage_values]))
print("Baseline average request", np.mean([float(val) for val in baseline_average_request_values]))
print("Baseline unique answer sets", np.mean([float(val) for val in baseline_unique_answer_sets_values]))
print("Baseline total answer sets", np.mean([float(val) for val in baseline_total_answer_sets_values]))
print("Baseline semantic clusters", np.mean([float(val) for val in baseline_semantic_clusters_values]))

print("Opt coverage", np.mean([float(val) for val in opt_coverage_values]))
print("Opt average request", np.mean([float(val) for val in opt_average_request_values]))
print("Opt unique answer sets", np.mean([float(val) for val in opt_unique_answer_sets_values]))
print("Opt total answer sets", np.mean([float(val) for val in opt_total_answer_sets_values]))
print("Opt semantic clusters", np.mean([float(val) for val in opt_semantic_clusters_values]))

param1s = []
param2s = []
for (param1, param2) in match:
    param1s.append(float(param1))
    param2s.append(float(param2))
print("Best parameters", np.mean(param1s), np.mean(param2s))
breakpoint()

with open('log_e2e_HMP_02_0', 'r') as file:
    content = file.read()
baseline_coverage_values.extend(re.findall(baseline_coverage, content))
baseline_average_request_values.extend(re.findall(baseline_average_request, content))
baseline_unique_answer_sets_values.extend(re.findall(baseline_unique_answer_sets, content))
baseline_total_answer_sets_values.extend(re.findall(baseline_total_answer_sets, content))
baseline_semantic_clusters_values.extend(re.findall(baseline_semantic_clusters, content))

opt_coverage_values.extend(re.findall(opt_coverage, content))
opt_average_request_values.extend(re.findall(opt_average_request, content))
opt_unique_answer_sets_values.extend(re.findall(opt_unique_answer_sets, content))
opt_total_answer_sets_values.extend(re.findall(opt_total_answer_sets, content))
opt_semantic_clusters_values.extend(re.findall(opt_semantic_clusters, content))
match = re.findall(best_parameters, content)

print("----HMP----")
print("Baseline coverage", np.mean([float(val) for val in baseline_coverage_values]))
print("Baseline average request", np.mean([float(val) for val in baseline_average_request_values]))
print("Baseline unique answer sets", np.mean([float(val) for val in baseline_unique_answer_sets_values]))
print("Baseline total answer sets", np.mean([float(val) for val in baseline_total_answer_sets_values]))
print("Baseline semantic clusters", np.mean([float(val) for val in baseline_semantic_clusters_values]))

print("Opt coverage", np.mean([float(val) for val in opt_coverage_values]))
print("Opt average request", np.mean([float(val) for val in opt_average_request_values]))
print("Opt unique answer sets", np.mean([float(val) for val in opt_unique_answer_sets_values]))
print("Opt total answer sets", np.mean([float(val) for val in opt_total_answer_sets_values]))
print("Opt semantic clusters", np.mean([float(val) for val in opt_semantic_clusters_values]))

param1s = []
param2s = []
for (param1, param2) in match:
    param1s.append(float(param1))
    param2s.append(float(param2))
print("Best parameters", np.mean(param1s), np.mean(param2s))
