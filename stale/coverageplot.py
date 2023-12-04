import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import numpy as np

most_relevant_pattern = r"Most relevant coverage:\s(\d+\.\d+)"
qa_coverage_pattern = r"QA coverage:\s(\d+\.\d+)"


levels = ['02', '015', '01', '005']
ret_covers = []
chat_covers = []
for level in levels:
    with open(f'log_bonf_rouge1_05_{level}', 'r') as file:
        content = file.read()
    most_relevant_values = re.findall(most_relevant_pattern, content)
    ret_covers.extend([float(val) for val in most_relevant_values])
    qa_coverage_values = re.findall(qa_coverage_pattern, content)
    chat_covers.extend([float(val) for val in qa_coverage_values])
levels_plot = []
levels = [0.2, 0.15, 0.1, 0.05]
for level in levels:
    level_tmp = 1 - float(level)/2
    levels_plot.extend([level_tmp] * 10)
ret_df = pd.DataFrame({'Prediction Set': ['Ret']*40, 'Coverage': ret_covers,
                       'Coverage Level': levels_plot})
chat_df = pd.DataFrame({'Prediction Set': ['Chat']*40, 'Coverage': chat_covers,
                        'Coverage Level': levels_plot})
Baseline = pd.DataFrame({'Prediction Set': ['Baseline']*40, 'Coverage': levels_plot,
                         'Coverage Level': levels_plot})
df = pd.concat([ret_df, chat_df, Baseline])
sns.lineplot(data=df, x='Coverage Level', y='Coverage', hue='Prediction Set', style="Prediction Set",)
plt.savefig('individual.png')




levels = [0.8, 0.9, 0.95]
Bonf_005 = [0.95, 0.95, 0.95, 0.94, 0.94,
            0.95, 0.95, 0.95, 0.95, 0.95]
Bonf_01 = [0.93, 0.93, 0.93, 0.93, 0.93,
           0.93, 0.93, 0.93, 0.93, 0.94]
Bonf_015 = [0.90, 0.90, 0.88, 0.93, 0.88,
            0.90, 0.86, 0.86, 0.91, 0.91]
Bonf_02 = [0.84, 0.82, 0.86, 0.81, 0.85,
           0.88, 0.85, 0.85, 0.88, 0.85]
Bonf_025 = [0.79, 0.80, 0.84, 0.73, 0.80, 
            0.79, 0.84, 0.81, 0.84, 0.75]
levels = [0.8] * 10 + \
         [0.85] * 10 + \
         [0.9] * 10 + \
         [0.95] * 10

HMP_005 = [0.94, 0.95, 0.95, 0.95, 0.95,
           0.95, 0.94, 0.95, 0.95, 0.95]
HMP_01 = [0.91, 0.93, 0.93, 0.93, 0.93,
          0.93, 0.93, 0.92, 0.93, 0.93]
HMP_015 = [0.84, 0.91, 0.91, 0.86, 0.86,
           0.91, 0.85, 0.91, 0.84, 0.90]
HMP_02 = [0.77, 0.83, 0.81, 0.77, 0.77,
          0.86, 0.877, 0.83, 0.77, 0.85]
HMP_025 = [0.74, 0.75, 0.75, 0.74, 0.74,
           0.74, 0.78, 0.73, 0.75, 0.73]


Bonf_df = pd.DataFrame({'Method': ['Bonf']*40, 'Coverage': Bonf_02 + Bonf_015 + Bonf_01 + Bonf_005,
                        'Coverage Level': levels})
HMP_df = pd.DataFrame({'Method': ['HMP']*40, 'Coverage': HMP_02 + HMP_015+ HMP_01 + HMP_005,
                        'Coverage Level': levels})
Baseline = pd.DataFrame({'Method': ['Baseline']*40, 'Coverage': levels,
                         'Coverage Level': levels})
df = pd.concat([Bonf_df, HMP_df, Baseline])
sns.lineplot(data=df, x='Coverage Level', y='Coverage', hue='Method', style="Method",)
plt.savefig('coverageplot.png')
