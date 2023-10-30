import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='chatgpt')
    args = parser.parse_args()

    # read in json file line by line
    with open(f'{args.task}_results.txt', 'r') as f:
        lines = f.readlines()
    # convert lines into pandas dataframe as a list of dicts
    lines = [json.loads(line) for line in lines]
    df = pd.DataFrame.from_records(lines)

    for ds in ['nq', 'trivia', 'squad1']:
        print('Dataset:', ds)
        df_ds = df[df['task'] == ds]
        # check if df_df is empty
        if df_ds.empty:
            print('No results found for this dataset.')
            continue
        # exclude task from df_ds
        df_ds = df_ds.drop(columns=['task'])
        # group df by alpha
        df_grouped = df_ds.groupby('alpha')
        # get mean and std for each task
        df_mean = df_grouped.mean()
        df_std = df_grouped.std()
        # print df_mean as latex table using two decimal places
        print(tabulate(df_mean.round(2), headers='keys', tablefmt='psql'))
        df_mean.index = 1 - df_mean.index /2
        df_mean = df_mean.reindex(df_mean.index[::-1])


        df_component = df_mean[['retrieval_coverage', 'qa_coverage', 'retrieval_coverage_pac', 'qa_coverage_pac']]
        # df_component = df_mean[['retrieval_coverage', 'qa_coverage']]
        # print(tabulate(df_component.round(2), headers='keys', tablefmt='psql'))
        print(df_component.round(2).to_markdown())
        print()

        df_mean.index = 1 - (1-df_mean.index)*2
        
        df_end2end = df_mean[['Vanila_coverage', 'TRAC_coverage', 'Bonf_coverage', 'PAC_Bonf_coverage', 'PAC_TRAC_coverage']]
        # df_end2end = df_mean[['Vanila_coverage', 'TRAC_coverage', 'Bonf_coverage', 'PAC_Bonf_coverage']]
        # # print(tabulate(df_end2end.round(2), headers='keys', tablefmt='psql'))
        print(df_end2end.round(2).to_markdown())
        print()

        if "Vania_average_semantic" in df_mean.columns:
            df_efficiency = df_mean[['Vania_average_semantic', 'TRAC_average_semantic', 'Bonf_average_semantic', 'PAC_Bonf_average_semantic', 'PAC_TRAC_average_semantic']]
        else:
            df_efficiency = df_mean[['Vanila_average_semantic', 'TRAC_average_semantic', 'Bonf_average_semantic', 'PAC_Bonf_average_semantic', 'PAC_TRAC_average_semantic']]
        # # print(tabulate(df_efficiency.round(2), headers='keys', tablefmt='psql'))
        print(df_efficiency.round(2).to_markdown())
        print()
    
    df_nq = df[df['task'] == 'nq']
    # exclude task from df_ds
    df_nq = df_nq.drop(columns=['task'])
    df_mean = df_nq.groupby('alpha').mean()
    df_mean.index = 1 - df_mean.index
    df_mean = df_mean.reindex(df_mean.index[::-1])
    df_coverage = df_mean[['Vanila_coverage', 'TRAC_coverage', 'Bonf_coverage', 'PAC_Bonf_coverage', 'PAC_TRAC_coverage']]

    # plot coverage
    plt.figure()
    plt.plot(df_coverage)
    plt.plot([0.5, 0.9], [0.5, 0.9], 'k--')
    plt.legend(list(df_coverage.columns) + ['Baseline'])
    plt.xlabel('alpha')
    plt.ylabel('coverage')
    plt.title('Coverage vs. alpha')
    plt.savefig(f'{args.task}_coverage.png')
