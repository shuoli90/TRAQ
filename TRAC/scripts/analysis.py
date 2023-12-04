import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
plt.rcParams.update({'font.size': 16}) # must set in top

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

    efficiencies = []
    for ds in ['semantic', 'bio', 'fewshot', 'nq', 'trivia', 'squad1']:
        print('Dataset:', ds)
        if ds == 'semantic':
            df_ds = df[df['semantic'] == True]
        else:
            df_ds = df[df['task'] == ds]
            df_ds = df_ds[df_ds['semantic'] == False]
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
        df_mean.index = (1 - df_mean.index /2) * 100
        df_mean.index = df_mean.index.astype(int)
        df_mean = df_mean.reindex(df_mean.index[::-1])
        df_std.index = (1 - df_std.index /2) * 100
        df_std.index = df_std.index.astype(int)
        df_std = df_std.reindex(df_mean.index[::-1])


        df_component = df_mean[['retrieval_coverage', 'qa_coverage', 'retrieval_coverage_pac', 'qa_coverage_pac']] * 100
        # df_component = df_mean[['retrieval_coverage', 'qa_coverage']]
        # print(tabulate(df_component.round(2), headers='keys', tablefmt='psql'))
        print(df_component.to_latex(float_format="{:.1f}".format,))
        print()
        plt.figure()
        plt.plot([75, 95], [75, 95], 'k--', linewidth=3)
        for col in df_component.columns:
            plt.errorbar(df_component.index, df_component[col], 
                         yerr=df_std[col]*100, marker='o', 
                         linewidth=2, alpha=0.6)

        plt.legend(['Baseline'] + ['Ret', 'Chatbot', 'Ret-P', 'Chat-P'])
        # plt.legend(list(df_component.columns) + ['Baseline'])
        plt.xlabel('Expected Coverage')
        plt.ylabel('Empirical')
        # plt.title(f'Expected vs. Empirical')
        # plt.xticks(['75', '80', '85', '90', '95'])
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{args.task}_{ds}_coverage (individual).png')

        df_mean.index = 100 - (100-df_mean.index) * 2
        # df_mean.index = df_mean.index * 100
        
        df_end2end = df_mean[['Vanila_coverage', 'TRAC_coverage', 'Bonf_coverage', 'PAC_TRAC_coverage', 'PAC_Bonf_coverage'] ] * 100
        # df_end2end = df_mean[['Vanila_coverage', 'TRAC_coverage', 'Bonf_coverage', 'PAC_Bonf_coverage']]
        # # print(tabulate(df_end2end.round(2), headers='keys', tablefmt='psql'))
        print(df_end2end.to_latex(float_format="{:.1f}".format,))
        print()

        if "Vania_average_semantic" in df_mean.columns:
            df_efficiency = df_mean[['Vanila_average_semantic', 'TRAC_average_semantic', 'Bonf_average_semantic', 'PAC_TRAC_average_semantic', 'PAC_Bonf_average_semantic']]
        else:
            df_efficiency = df_mean[['Vanila_average_semantic', 'TRAC_average_semantic', 'Bonf_average_semantic', 'PAC_TRAC_average_semantic', 'PAC_Bonf_average_semantic']]
        # # print(tabulate(df_efficiency.round(2), headers='keys', tablefmt='psql'))
        print(df_efficiency.to_latex(float_format="{:.1f}".format,))
        print()

        plt.figure()
        efficiency_plot = df_ds[['TRAC_average_semantic', 'Bonf_average_semantic', 'PAC_TRAC_average_semantic', 'PAC_Bonf_average_semantic']]
        efficiency_plot.columns = ['TRAC', 'Bonf', 'TRAC-P', 'Bonf-P']
        efficiency_plot.iloc[-5:,].boxplot()
        plt.xlabel('Method')
        if ds == 'semantic':
            plt.ylabel('Average Unique')  
        else:
            plt.ylabel('Average Semantic')
        plt.tight_layout()
        plt.savefig(f'{args.task}_{ds}_efficiency.png')
        efficiencies.append(df_efficiency[['TRAC_average_semantic']])

    plt.figure()
    efficiency_plot = df[df['task'].isin(['nq', 'fewshot'])]
    # change 'nq' to 'NQ' and 'fewshot' to 'Fewshot'
    efficiency_plot['task'] = efficiency_plot['task'].replace({'nq': 'Zeroshot', 'fewshot': 'Fewshot'})
    # change 'task' to 'Prompt'
    efficiency_plot = efficiency_plot.rename(columns={'task': 'Prompt'})
    result = efficiency_plot.groupby(['Prompt', 'alpha'])[['TRAC_average_semantic']].agg([np.mean, np.std])['TRAC_average_semantic']
    # reorder index to put 'Zeroshot' before 'Fewshot'
    result = result.reindex(['Zeroshot', 'Fewshot'], level=0)
    result = result.unstack(level=0)
    result.dropna(inplace=True)
    result.index = (1 - result.index) * 100
    result.index = result.index.astype(int)
    # flip index order
    result = result.reindex(result.index[::-1])
    plt.style.use('tableau-colorblind10')
    result.plot(kind="bar", y="mean", yerr="std", colormap='tab10', rot=0, width=0.8)
    # plt.errorbar(x=result.index*100, y=result['mean'].to_numpy(), yerr=result['std'].to_numpy(), fmt="o", color="r")
    plt.xlabel('Coverage')
    plt.ylabel('Semantic Count')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f'{args.task}_prompt.png')

    

    # read in json file line by line
    with open(f'{args.task}_results_vanila.txt', 'r') as f:
        lines = f.readlines()
    # convert lines into pandas dataframe as a list of dicts
    lines = [json.loads(line) for line in lines]
    df_vanila = pd.DataFrame.from_records(lines)

    for ds in ['semantic', 'bio', 'nq', 'trivia', 'squad1']:
        print('Dataset:', ds)
        if ds == 'semantic':
            df_ds = df[df['semantic'] == True]
        else:
            df_ds = df[df['task'] == ds]
            df_ds = df_ds[df_ds['semantic'] == False]
        if df_ds.empty:
            print('No results found for this dataset.')
            continue
        # exclude task from df_ds
        df_nq = df_ds.drop(columns=['task'])
        df_mean = df_nq.groupby('alpha').mean()
        df_mean.index = 1 - df_mean.index
        df_mean = df_mean.reindex(df_mean.index[::-1])
        df_coverage = df_mean[['TRAC_coverage', 'Bonf_coverage', 'PAC_Bonf_coverage', 'PAC_TRAC_coverage']]

        plt.figure()
        df_e2e = df_mean[['end_to_end_coverage', 'end_to_end_coverage_pac']]
        plt.plot([50, 90], [50, 90], 'k--', linewidth=3)
        for col in df_e2e.columns:
            plt.errorbar(df_e2e.index*100, df_e2e[col]*100, 
                            yerr=df_std[col]*100, marker='o', 
                            linewidth=2, alpha=0.6)

        plt.legend(['Baseline'] + ['End-to-End', 'End-to-End (PAC)']) 
        plt.xlabel('Expected Coverage')
        plt.ylabel('Empirical')
        # plt.title(f'Expected vs. Empirical')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{args.task}_{ds}_e2e_coverage.png')
        df_e2e.index = df_e2e.index * 100
        df_e2e.index = df_e2e.index.astype(int)
        # df_e2e = df_e2e.reindex(df_e2e.index[::-1])
        df_e2e *= 100
        print(df_e2e.to_latex(float_format="{:.1f}".format,))

        plt.figure()
        plt.plot([50, 90], [50, 90], 'k--', linewidth=3)
        if ds == 'semantic':
            level = df_vanila[df_vanila['semantic']==True]['Vanila_coverage'] * 100
        else:
            df_vanila_tmp = df_vanila[df_vanila['semantic'] == False]
            level = df_vanila_tmp[df_vanila_tmp['task'] == ds]['Vanila_coverage'] * 100
        plt.hlines(level, 50, 90, color='r', linestyles='dotted', linewidth=3)

        for col in df_coverage.columns:
            plt.errorbar(df_coverage.index*100, df_coverage[col]*100, 
                            yerr=df_std[col]*100, marker='o', 
                            linewidth=2, alpha=0.6)

        plt.legend(['Baseline'] + ['Vanila', 'TRAC', 'Bonf', 'TRAC-P', 'Bonf-P']) 
        plt.xlabel('Expected Coverage')
        plt.ylabel('Empirical')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f'{args.task}_{ds}_e2e_coverage2.png')
        print(df_coverage.to_latex(float_format="{:.1f}".format,))

    # select only 'nq', 'trivia', 'squad1'
    df_tmp = df[df['semantic'] == False]
    df_tmp = df_tmp[df_tmp['task'].isin(['nq', 'trivia', 'squad1'])]
    # rename 'nq' to 'NQ', 'trivia' to 'Trivia', 'squad1' to 'SQuAD1'
    df_tmp['task'] = df_tmp['task'].replace({'nq': 'NQ', 'trivia': 'Trivia', 'squad1': 'SQuAD1'})
    df_tmp = df_tmp[['task', 'alpha', 'TRAC_average_semantic', 'Bonf_average_semantic', 'PAC_TRAC_average_semantic', 'PAC_Bonf_average_semantic']]
    # set 'task' as index
    df_tmp = df_tmp.set_index('task')
    # rename columns
    df_tmp.columns = ['Cov', 'TRAC', 'Bonf', 'TRAC-P', 'Bonf-P']
    df_tmp['Cov'] = ((1 - df_tmp['Cov']) * 100).astype(int)
    # df_tmp = df_tmp.drop(columns=['task'])
    # group df by alpha
    df_grouped = df_tmp.groupby(['task', 'Cov'])
    # get mean and std for each task
    df_mean = df_grouped.mean()
    df_std = df_grouped.std()
    # add a new row, which is the average of 'nq', 'trivia', 'squad1'
    df_mean.loc[('Average', 50), :] = df_mean.loc[['NQ', 'Trivia', 'SQuAD1'], :].mean()
    # df_mean.index = (1 - df_mean.index /2) * 100
    # df_mean.index = df_mean.index.astype(int)
    # df_mean = df_mean.reindex(df_mean.index[::-1])
    # df_std.index = (1 - df_std.index /2) * 100
    # df_std.index = df_std.index.astype(int)
    # df_std = df_std.reindex(df_mean.index[::-1])
    print(df_mean.to_latex(float_format="{:.1f}".format,))

    # compute average efficiency improvement
    Bonf = (df_mean.iloc[-1, 1] - df_mean.iloc[-1, 0]) / df_mean.iloc[-1, 1]
    PAC = (df_mean.iloc[-1, 3] - df_mean.iloc[-1, 2]) / df_mean.iloc[-1, 3]
    print('Average Efficiency Improvement:', (Bonf + PAC) / 2)