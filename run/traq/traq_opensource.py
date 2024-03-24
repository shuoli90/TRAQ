import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from misc import utils
from rouge_score import rouge_scorer
import random
import numpy as np
import torch
import argparse
torch.set_grad_enabled(False)
from misc.pac_utils import find_maximum_train_error_allow
import time
import json
from skopt.space import Real
from skopt import gp_minimize
from skopt.utils import use_named_args
from multiprocessing import Value
import run.traq.traq_chatgpt as traq_chatgpt
import matplotlib.pyplot as plt

def read_results(task, end=1000, dir='../collected_data'):
    retrieved_scores = traq_chatgpt.read_list(os.path.join(dir, f'retrieved_scores_{task}.p'))[:end]
    retrieved_true_scores = traq_chatgpt.read_list(os.path.join(dir, f'retrieved_true_scores_{task}.p'))[:end]
    queries = traq_chatgpt.read_list(os.path.join(dir, f'queries_{task}.p'))[:end]
    answers = traq_chatgpt.read_list(os.path.join(dir, f'answers_{task}.p'))[:end]
    opensource_true_scores = traq_chatgpt.read_list(os.path.join(dir, f'opensource_true_scores_{task}.p'))[:end]
    opensource_answers = traq_chatgpt.read_list(os.path.join(dir, f'opensource_answers_{task}.p'))[:end]
    opensource_semantics = traq_chatgpt.read_list(os.path.join(dir, f'opensource_semantics_{task}.p'))[:end]
    opensource_occurances = traq_chatgpt.read_list(os.path.join(dir, f'occurances_{task}.p'))[:end]
    opensource_semantic_ids = traq_chatgpt.read_list(os.path.join(dir, f'semantic_ids_{task}.p'))[:end]
    opensource_probs = traq_chatgpt.read_list(os.path.join(dir, f'probs_{task}.p'))[:end]
    
    return retrieved_scores, retrieved_true_scores, \
           queries, answers, \
           opensource_true_scores, opensource_answers, \
           opensource_occurances, opensource_semantic_ids, opensource_probs

"""
Weight HMP module
"""
w1 = Real(name='w1', low=0.0, high=1.0)
w2 = Real(name='w2', low=0.0, high=1.0)

# Gather the search-space dimensions in a list.
dimensions = [w1, w2]

@use_named_args(dimensions=dimensions)
def objective(w1, w2):
    weights = traq_chatgpt.softmax(np.array([w1, w2])).reshape(-1, 1)
    alpha_retrieve = alpha * weights[0]
    alpha_qa = alpha * weights[1]
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve)
    cal_first_scores = []
    for scores in cal_first_opensource_true_scores:
        cal_first_scores.append(np.max(scores))
    opensource_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa)
    results = traq_chatgpt.evaluate(cal_second_retrieved_scores, cal_second_queries, \
                       cal_second_answers, cal_second_opensource_answers, \
                       cal_second_opensource_semantic_ids, cal_second_opensource_probs, \
                       retrieved_thr, opensource_qa_thr, scorer=scorer,
                       cluster=True
                      )
    coverage = np.mean(results[0])
    average_answer = np.mean(results[1])
    average_semantic = np.mean(results[2])
    return average_semantic

@use_named_args(dimensions=dimensions)
def objective_pac(w1, w2):
    weights = traq_chatgpt.softmax(np.array([w1, w2])).reshape(-1, 1)
    alpha_retrieve = alpha * weights[0]
    alpha_qa = alpha * weights[1]
    alpha_retrieve_pac = find_maximum_train_error_allow(alpha_retrieve, delta/2.0, len(cal_first_indices))
    alpha_qa_pac = find_maximum_train_error_allow(alpha_qa, delta/2.0, len(cal_first_indices))

    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve_pac)
    cal_first_scores = []
    for scores in cal_first_opensource_true_scores:
        cal_first_scores.append(np.max(scores))
    opensource_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa_pac)
    results = traq_chatgpt.evaluate(cal_second_retrieved_scores, cal_second_queries, 
                       cal_second_answers, cal_second_opensource_answers, \
                       cal_second_opensource_semantic_ids, cal_second_opensource_probs, 
                       retrieved_thr, opensource_qa_thr, scorer=traq_chatgpt.scorer,
                       cluster=True)
    coverage = np.mean(results[0])
    average_answer = np.mean(results[1])
    average_semantic = np.mean(results[2])
    return average_semantic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='nq')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--semantic', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    start = time.time()
    results_dict = {}
    alpha = args.alpha

    print("****************************")
    print(f'Task={args.task}, alpha={args.alpha}, seed={args.seed}, semantic={args.semantic}')
    print("****************************")
    results_dict["task"] = args.task
    results_dict["alpha"] = args.alpha
    results_dict["seed"] = args.seed
    results_dict["semantic"] = args.semantic
    alpha = args.alpha
    seed = args.seed
    task = args.task
    semantic = args.semantic


    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                            use_stemmer=True)
    if args.semantic:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        # setup semantic model
        semantic_tokenizer = \
            AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        semantic_model = \
            AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli"
            ).cuda()

    retrieved_scores, retrieved_true_scores, queries, answers, opensource_true_scores, opensource_answers, opensource_occurances, opensource_semantic_ids, opensource_probs = \
        read_results(task, end=1000)
    

    indices = np.arange(len(retrieved_true_scores))
    random.shuffle(indices)
    cal_first_indices = indices[:int(len(indices) * 0.3)]
    cal_second_indices = indices[int(len(indices) * 0.3) : int(len(indices) * 0.6)]
    test_indices = indices[int(len(indices) * 0.6):]

    # indices = np.arange(1000)
    indices = np.arange(len(queries))
    random.shuffle(indices)
    cal_first_indices = indices[:int(len(indices) * 0.3)]
    cal_second_indices = indices[int(len(indices) * 0.3) : int(len(indices) * 0.6)]
    test_indices = indices[int(len(indices) * 0.6):]
    # test_indices = indices[int(len(indices) * 0.3):]

    # indices = np.arange(1000)
    indices = np.arange(len(queries))
    random.shuffle(indices)
    cal_first_indices = indices[:int(len(indices) * 0.3)]
    cal_second_indices = indices[int(len(indices) * 0.3) : int(len(indices) * 0.6)]
    test_indices = indices[int(len(indices) * 0.6):]
    # test_indices = indices[int(len(indices) * 0.3):]

    cal_first_retrieved_true_scores = utils.split(retrieved_true_scores, cal_first_indices)
    cal_second_retrieved_true_scores = utils.split(retrieved_true_scores, cal_second_indices)
    test_retrieved_true_scores = utils.split(retrieved_true_scores, test_indices)
    cal_first_opensource_true_scores = utils.split(opensource_true_scores, cal_first_indices)
    cal_second_opensource_true_scores = utils.split(opensource_true_scores, cal_second_indices)
    test_opensource_true_scores = utils.split(opensource_true_scores, test_indices)
    cal_first_retrieved_scores = utils.split(retrieved_scores, cal_first_indices)
    cal_second_retrieved_scores = utils.split(retrieved_scores, cal_second_indices)
    test_retrieved_scores = utils.split(retrieved_scores, test_indices)
    cal_first_opensource_occurances = utils.split(opensource_occurances, cal_first_indices)
    cal_second_opensource_occurances = utils.split(opensource_occurances, cal_second_indices)
    test_opensource_occurances = utils.split(opensource_occurances, test_indices)
    cal_first_opensource_semantic_ids = utils.split(opensource_semantic_ids, cal_first_indices)
    cal_second_opensource_semantic_ids = utils.split(opensource_semantic_ids, cal_second_indices)
    test_opensource_semantic_ids = utils.split(opensource_semantic_ids, test_indices)
    cal_first_queries = utils.split(queries, cal_first_indices)
    cal_second_queries = utils.split(queries, cal_second_indices)
    test_queries = utils.split(queries, test_indices)
    cal_first_opensource_answers = utils.split(opensource_answers, cal_first_indices)
    cal_second_opensource_answers = utils.split(opensource_answers, cal_second_indices)
    test_opensource_answers = utils.split(opensource_answers, test_indices)
    cal_first_answers = utils.split(answers, cal_first_indices)
    cal_second_answers = utils.split(answers, cal_second_indices)
    test_answers = utils.split(answers, test_indices)
    cal_first_opensource_probs = utils.split(opensource_probs, cal_first_indices)
    cal_second_opensource_probs = utils.split(opensource_probs, cal_second_indices)
    test_opensource_probs = utils.split(opensource_probs, test_indices)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                       use_stemmer=True)

    print("Individual components")
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=args.alpha/2)
    cal_first_scores = []
    for scores in cal_first_opensource_true_scores:
        cal_first_scores.append(np.max(scores))
    opensource_qa_thr = utils.compute_threshold(cal_first_scores, alpha=args.alpha/2)

    retrieved_coverage = np.mean(np.array(cal_second_retrieved_true_scores) >= retrieved_thr)
    cal_second_scores = []
    for scores in cal_second_opensource_true_scores:
        cal_second_scores.append(np.max(scores))
    qa_coverage = np.mean(np.array(cal_second_scores) >= opensource_qa_thr)
    print('retrieval coverage', retrieved_coverage)
    print('qa coverage', qa_coverage)

    retrieved_coverage = np.mean(np.array(test_retrieved_true_scores) >= retrieved_thr)
    test_scores = []
    for scores in test_opensource_true_scores:
        test_scores.append(np.max(scores))
    qa_coverage = np.mean(np.array(test_scores) >= opensource_qa_thr)
    print('test retrieval coverage', retrieved_coverage)
    print('test qa coverage', qa_coverage)

    results_dict["retrieval_coverage"] = retrieved_coverage
    results_dict["qa_coverage"] = qa_coverage

    coverages = traq_chatgpt.coverage(
        test_retrieved_true_scores,
        test_opensource_true_scores,
        retrieved_thr,
        opensource_qa_thr
        )
    print('End-to-end coverage', np.mean(coverages))
    results_dict["end_to_end_coverage"] = np.mean(coverages)

    print("Individual compponents with PAC")
    delta = 0.1
    retrieve_alpha = find_maximum_train_error_allow(args.alpha/2.0, delta/2.0, len(cal_first_indices))
    qa_alpha = find_maximum_train_error_allow(args.alpha/2.0, delta/2.0, len(cal_first_indices))
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=retrieve_alpha)
    cal_first_scores = []
    for scores in cal_first_opensource_true_scores:
        cal_first_scores.append(np.max(scores))
    opensource_qa_thr = utils.compute_threshold(cal_first_scores, alpha=qa_alpha)

    retrieved_coverage = np.mean(np.array(cal_second_retrieved_true_scores) >= retrieved_thr)
    cal_second_scores = []
    for scores in cal_second_opensource_true_scores:
        cal_second_scores.append(np.max(scores))
    qa_coverage = np.mean(np.array(cal_second_scores) >= opensource_qa_thr)
    print('retrieval coverage', retrieved_coverage)
    print('qa coverage', qa_coverage)

    retrieved_coverage = np.mean(np.array(test_retrieved_true_scores) >= retrieved_thr)
    test_scores = []
    for scores in test_opensource_true_scores:
        test_scores.append(np.max(scores))
    qa_coverage = np.mean(np.array(test_scores) >= opensource_qa_thr)
    print('test retrieval coverage', retrieved_coverage)
    print('test qa coverage', qa_coverage)

    coverages = traq_chatgpt.coverage(
        test_retrieved_true_scores, 
        test_opensource_true_scores,
        retrieved_thr,
        opensource_qa_thr
        )
    print('End-to-end coverage', np.mean(coverages))

    results_dict["retrieval_coverage_pac"] = retrieved_coverage
    results_dict["qa_coverage_pac"] = qa_coverage
    results_dict["end_to_end_coverage_pac"] = np.mean(coverages)
    # plot histogram of cal_first_scores
    plt.hist(cal_first_scores, bins=20)
    # plt vline at opensource_qa_thr
    plt.axvline(opensource_qa_thr, color='r', linestyle='dashed', linewidth=1)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(f'cal_first_scores_{task}.png')
    breakpoint()

    """
    Weight Bonf module
    """
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        acq_func="EI",      # the acquisition function
        n_calls=15,
        random_state=args.seed,
        verbose=False,
        x0=[[1, 1]])

    print("Best fitness:", result.fun)
    print("Best parameters:", traq_chatgpt.softmax(result.x))

    weights = traq_chatgpt.softmax(result.x).reshape(-1, 1)
    alpha_retrieve = alpha * weights[0]
    alpha_qa = alpha * weights[1]
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve)
    cal_first_scores = []
    for scores in cal_first_opensource_true_scores:
        cal_first_scores.append(np.max(scores))
    opensource_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa)
    results = traq_chatgpt.evaluate(
        test_retrieved_scores, test_queries,
        test_answers, test_opensource_answers, 
        test_opensource_semantic_ids, test_opensource_probs,
        retrieved_thr, opensource_qa_thr, scorer=scorer,
        cluster=True
    )

    print('TRAC')
    print('Desired coverage rate', 1-args.alpha)
    print('Coverage', np.mean(results[0]))
    # print('Average answer', np.mean(results[1]))
    print('Average semantic', np.mean(results[2]))

    results_dict["TRAC_coverage"] = np.mean(results[0])
    results_dict["TRAC_average_semantic"] = np.mean(results[2])

    alpha_retrieve = alpha * (1/2.0)
    alpha_qa = alpha * (1/2.0)
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve)
    cal_first_scores = []
    for scores in cal_first_opensource_true_scores:
        cal_first_scores.append(np.max(scores))
    opensource_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa)
    results = traq_chatgpt.evaluate(
        test_retrieved_scores, test_queries,
        test_answers, test_opensource_answers, 
        test_opensource_semantic_ids, test_opensource_probs,
        retrieved_thr, opensource_qa_thr, scorer=scorer,
        cluster=True
    )
    print('Bonf')
    print('Desired coverage rate', 1-args.alpha)
    print('Coverage', np.mean(results[0]))
    # print('Average answer', np.mean(results[1]))
    print('Average semantic', np.mean(results[2]))

    results_dict["Bonf_coverage"] = np.mean(results[0])
    results_dict["Bonf_average_semantic"] = np.mean(results[2])

    alpha_retrieve = alpha * (1/2.0)
    alpha_qa = alpha * (1/2.0)

    delta = 0.1
    alpha_retrieve_pac = find_maximum_train_error_allow(alpha_retrieve, delta/2.0, len(cal_first_indices))
    alpha_qa_pac = find_maximum_train_error_allow(alpha_qa, delta/2.0, len(cal_first_indices))

    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve_pac)
    cal_first_scores = []
    for scores in cal_first_opensource_true_scores:
        cal_first_scores.append(np.max(scores))
    opensource_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa_pac)
    results = traq_chatgpt.evaluate(
        test_retrieved_scores, test_queries,
        test_answers, test_opensource_answers, 
        test_opensource_semantic_ids, test_opensource_probs, 
        retrieved_thr, opensource_qa_thr, scorer=scorer, sample=False)
    print('PAC-Bonf')
    print('Desired coverage rate', 1-args.alpha)
    print('Coverage', np.mean(results[0]))
    # print('Average answer', np.mean(results[1]))
    print('Average semantic', np.mean(results[2]))

    results_dict["PAC_Bonf_coverage"] = np.mean(results[0])
    # results_dict["PAC_Bonf_average_answer"] = np.mean(results[1])   
    results_dict["PAC_Bonf_average_semantic"] = np.mean(results[2])

    """
    Weight PAC-TRAC module
    """
    result = gp_minimize(
        func=objective_pac,
        dimensions=dimensions,
        acq_func="EI",      # the acquisition function
        n_calls=15,
        random_state=args.seed,
        verbose=False,
        x0=[[1, 1]])

    print("Best fitness:", result.fun)
    print("Best parameters:", traq_chatgpt.softmax(result.x))

    weights = traq_chatgpt.softmax(result.x).reshape(-1, 1)
    alpha_retrieve = alpha * weights[0]
    alpha_qa = alpha * weights[1]
    alpha_retrieve_pac = find_maximum_train_error_allow(alpha_retrieve, delta/2.0, len(cal_first_indices))
    alpha_qa_pac = find_maximum_train_error_allow(alpha_qa, delta/2.0, len(cal_first_indices))
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve_pac)
    cal_first_scores = []
    for scores in cal_first_opensource_true_scores:
        cal_first_scores.append(np.max(scores))
    opensource_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa_pac)
    results = traq_chatgpt.evaluate(
        test_retrieved_scores, test_queries,
        test_answers, test_opensource_answers, 
        test_opensource_semantic_ids, test_opensource_probs, 
        retrieved_thr, opensource_qa_thr, scorer=scorer,
        cluster=True)

    print('PAC-TRAC')
    print('Desired coverage rate', 1-args.alpha)
    print('Coverage', np.mean(results[0]))
    # print('Average answer', np.mean(results[1]))
    print('Average semantic', np.mean(results[2]))

    results_dict["PAC_TRAC_coverage"] = np.mean(results[0])
    results_dict["PAC_TRAC_average_semantic"] = np.mean(results[2])

    results = traq_chatgpt.evaluate_vanila(
        retrieved_scores, queries,
        answers, opensource_answers, 
        opensource_semantic_ids, opensource_probs, scorer=scorer, sample=False,
        cluster=True)
    
    print('Vanila')
    print('Desired coverage rate', 1-alpha)
    print('Coverage', np.mean(results[0]))
    # print('Average answer', np.mean(results[1]))
    print('Average semantic', np.mean(results[2]))
    results_dict["Vanila_coverage"] = np.mean(results[0])
    # results_dict["PAC_Bonf_average_answer"] = np.mean(results[1])   
    results_dict["Vanila_average_semantic"] = np.mean(results[2])

    print()
    print()

    with open("../collected_results/opensource_results_new.txt", "a") as f:
        json.dump(results_dict, f)
        f.write("\n")
