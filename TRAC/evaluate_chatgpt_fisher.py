import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
from kilt import kilt_utils as utils
# from kilt.retrievers import DPR_connector
import utils
from rouge_score import rouge_scorer
import random
import numpy as np
import torch
import argparse
torch.set_grad_enabled(False)

def read_chatgpt_results(task):
    retrieved_scores = read_list(f'chatgpt_retrieved_scores_{task}.p')
    retrieved_true_scores = read_list(f'chatgpt_retrieved_true_scores_{task}.p')
    queries = read_list(f'chatgpt_queries_{task}.p')
    answers = read_list(f'chatgpt_answers_{task}.p')
    chatgpt_true_scores = read_list(f'chatgpt_true_scores_{task}.p')
    chatgpt_answers = read_list(f'chatgpt_answers_{task}.p')
    chatgpt_passages = read_list(f'chatgpt_passages_{task}.p')
    chatgpt_semantics = read_list(f'chatgpt_semantics_{task}.p')
    chatgpt_occurances = read_list(f'chatgpt_occurances_{task}.p')
    chatgpt_semantic_ids = read_list(f'chatgpt_semantic_ids_{task}.p')
    chatgpt_probs = read_list(f'chatgpt_probs_{task}.p')
    
    retrieved_scores_unc = read_list(f'chatgpt_retrieved_scores_unc_{task}.p')
    retrieved_true_scores_unc = read_list(f'chatgpt_retrieved_true_scores_unc_{task}.p')
    queries_unc = read_list(f'chatgpt_queries_unc_{task}.p')
    answers_unc = read_list(f'chatgpt_answers_unc_{task}.p')
    passages_unc = read_list(f'chatgpt_passages_unc_{task}.p')
    chatgpt_true_scores_unc = read_list(f'chatgpt_true_scores_unc_{task}.p')
    chatgpt_answers_unc = read_list(f'chatgpt_answers_unc_{task}.p')
    chatgpt_occurances_unc = read_list(f'chatgpt_occurances_unc_{task}.p')
    chatgpt_semantic_ids_unc = read_list(f'chatgpt_semantic_ids_unc_{task}.p')
    chatgpt_probs_unc = read_list(f'chatgpt_probs_unc_{task}.p')
    
    return retrieved_scores, retrieved_true_scores, queries, answers, chatgpt_true_scores, chatgpt_answers, chatgpt_passages, chatgpt_semantics, chatgpt_occurances, chatgpt_semantic_ids, chatgpt_probs, retrieved_scores_unc, retrieved_true_scores_unc, queries_unc, answers_unc, passages_unc, chatgpt_true_scores_unc, chatgpt_answers_unc, chatgpt_occurances_unc, chatgpt_semantic_ids_unc, chatgpt_probs_unc


def coverage(
        retrieved_true_scores_list, opensource_true_scores_list,
        retrieved_thr, qa_thr):

    includes = []
    for idx, (retrieved_true_score, opensource_true_score) in enumerate(zip(retrieved_true_scores_list, opensource_true_scores_list)):
#         if idx > 20:
        opensource_true_score = np.max(opensource_true_score)
        include = True if (retrieved_true_score >= retrieved_thr and 
                        opensource_true_score >= qa_thr) \
                    else False
        includes.append(include)
    return includes


from skopt.space import Real
from skopt import gp_minimize
from skopt.utils import use_named_args


def evaluate(
    test_retrieved_scores,
    test_queries, test_answers, test_chatgpt_answers, 
    test_chatgpt_occurances, test_chatgpt_semantic_ids,
    retrieved_thr, chatgpt_qa_thr, cluster=False):

    includes = []
    answer_counts = []
    semantic_counts = []
    includes = []
    coverages = []
    for idx, (retrieved_scores_tmp, \
            query_tmp, answers_tmp, chatgpt_answers_tmp, \
            chatgpt_true_scores_tmp, \
            chatgpt_occurances_tmp, chatgpt_semantic_ids_tmp) \
        in enumerate(zip(
            test_retrieved_scores, \
            test_queries, test_answers, test_chatgpt_answers, \
            test_chatgpt_true_scores, \
            test_chatgpt_occurances, test_chatgpt_semantic_ids)):
        include = False
        retrieved_count = 0
        answers_tmp = []
        semantics = []
        for retrieved_score, answer, item_occurance, semantic_set_ids in zip(retrieved_scores_tmp, chatgpt_answers_tmp, chatgpt_occurances_tmp, chatgpt_semantic_ids_tmp):
            if retrieved_score < retrieved_thr:
                continue
            else:
                retrieved_count += 1
                for predicted_answer in semantic_set_ids.keys():
                    concept_id = semantic_set_ids[predicted_answer]
                    repeat = item_occurance[predicted_answer]
                    prob = repeat / len(answer)
                    if prob >= chatgpt_qa_thr:
                        answers_tmp.extend([predicted_answer] * repeat)
                        semantics.append(predicted_answer)
                        
                        # TODO: check if the semantic is consistent with true answer
                        if include is False:
                            for answer_tmp in answers_tmp:
                                scores = scorer.score(answer_tmp,
                                                    predicted_answer)
                                scores = scores['rouge1'][2]
                                if scores >= 0.3:
                                    include = True
                                    break
        if cluster:
            semantic_set_ids, semantic_probs, item_occurance = \
                        utils.clustering(semantics, "", scorer=scorer)
                    
            answer_counts.append(len(answers_tmp))
            semantic_counts.append(len(semantic_probs.keys()))
        else:
            answer_counts.append(len(answers_tmp))
            semantic_counts.append(len(semantics))
        includes.append(include)
        
    return [includes, answer_counts, semantic_counts]


from scipy.stats import combine_pvalues


def evaluate_fisher(
    test_retrieved_scores,
    test_queries, test_answers, test_opensource_answers, 
    test_opensource_occurances, test_opensource_semantic_ids,
    w0=1/2.0, w1=1/2.0,
    cluster=True):

    includes = []
    answer_counts = []
    semantic_counts = []
    includes = []
    coverages = []
    for idx, (retrieved_scores_tmp, \
            query_tmp, answers_tmp, opensource_answers_tmp, \
            opensource_occurances_tmp, opensource_semantic_ids_tmp) \
        in enumerate(zip(
            test_retrieved_scores, \
            test_queries, test_answers, test_opensource_answers, \
            test_opensource_occurances, test_opensource_semantic_ids)):

        include = False
        retrieved_count = 0
        answers_tmp = []
        semantics = []
        for retrieved_score, answer, item_occurance, semantic_set_ids in zip(retrieved_scores_tmp, opensource_answers_tmp, opensource_occurances_tmp, opensource_semantic_ids_tmp):
            semantic_clusterring, semantic_probs, item_occurance = \
                    utils.compute_keyword_clusterring(
                        answers=answer,
                        scorer=scorer)
            for predicted_answer in semantic_set_ids.keys():
                concept_id = semantic_set_ids[predicted_answer]
                repeat = item_occurance[predicted_answer]
                prob = repeat / len(answer)
                # compute p-values
                retriever_p = np.mean(retrieved_score >= np.array(cal_first_retrieved_true_scores).flatten())
                qa_p = np.mean(prob >= np.array(cal_first_scores))
                combined_p_value = combine_pvalues([retriever_p, qa_p], weights=[w0, w1])[1]
                if combined_p_value >= args.alpha:
                    answers_tmp.extend([predicted_answer] * repeat)
                    semantics.append(predicted_answer)
                    if include is False:
                        for answer_tmp in answers_tmp:
                            scores = scorer.score(answer_tmp,
                                                  predicted_answer)
                            scores = scores['rouge1'][2]
                            if scores >= 0.3:
                                include = True
                                break
        if cluster:
            semantic_set_ids, semantic_probs, item_occurance = \
                        utils.clustering(semantics, "", scorer=scorer)

            answer_counts.append(len(answers_tmp))
            semantic_counts.append(len(semantic_probs.keys()))
        else:
            answer_counts.append(len(answers_tmp))
            semantic_counts.append(len(semantics))
        includes.append(include)
        
    return [includes, answer_counts, semantic_counts]


w1 = Real(name='w1', low=0.0, high=1.0)
w2 = Real(name='w2', low=0.0, high=1.0)

def softmax(vec):
    nom = np.exp(vec - np.mean(vec))
    return nom / np.sum(nom)

# Gather the search-space dimensions in a list.
dimensions = [w1, w2]


@use_named_args(dimensions=dimensions)
def objective(w1, w2):
    weights = softmax(np.array([w1, w2])).reshape(-1, 1)
    alpha_retrieve = args.alpha * weights[0]
    alpha_qa = args.alpha * weights[1]
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve)
    cal_first_scores = []
    for scores in cal_first_chatgpt_true_scores:
        cal_first_scores.append(np.max(scores))
    chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa)
    results = evaluate(cal_second_retrieved_scores, cal_second_queries, 
                    cal_second_answers, cal_second_chatgpt_answers, \
                    cal_second_chatgpt_occurances, cal_second_chatgpt_semantic_ids,
                    retrieved_thr, chatgpt_qa_thr, 
                    cluster=True)
    coverage = np.mean(results[0])
    average_answer = np.mean(results[1])
    average_semantic = np.mean(results[2])
    return average_semantic


@use_named_args(dimensions=dimensions)
def objective_fisher(w1, w2):
    weights = softmax(np.array([w1, w2])).reshape(-1, 1)
    results = evaluate_fisher(
        cal_second_retrieved_scores, cal_second_queries, 
        cal_second_answers, cal_second_chatgpt_answers, \
        cal_second_chatgpt_occurances, cal_second_chatgpt_semantic_ids,
        w0=weights[0], w1=weights[1],
        cluster=True)
    coverage = np.mean(results[0])
    average_answer = np.mean(results[1])
    average_semantic = np.mean(results[2])
    return average_semantic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='nq')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--semantic', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f'Task={args.task}, alpha={args.alpha}, seed={args.seed}, semantic={args.semantic}')

    import pickle
    def write_list(a_list, file_name):
        # store list in binary file so 'wb' mode
        with open(file_name, 'wb') as fp:
            pickle.dump(a_list, fp)
            print('Done writing list into a binary file')
    def read_list(file_name):
        # for reading also binary mode is important
        with open(file_name, 'rb') as fp:
            n_list = pickle.load(fp)
            return n_list

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

    retrieved_scores, retrieved_true_scores, queries, answers, chatgpt_true_scores, chatgpt_answers, chatgpt_passages, chatgpt_semantics, chatgpt_occurances, chatgpt_semantic_ids, chatgpt_probs, retrieved_scores_unc, retrieved_true_scores_unc, queries_unc, answers_unc, passages_unc, chatgpt_true_scores_unc, chatgpt_answers_unc, chatgpt_occurances_unc, chatgpt_semantic_ids_unc, chatgpt_probs_unc = \
            read_chatgpt_results(args.task)

    indices = np.arange(len(retrieved_true_scores))
    random.shuffle(indices)
    cal_first_indices = indices[:int(len(indices) * 0.3)]
    cal_second_indices = indices[int(len(indices) * 0.3) : int(len(indices) * 0.6)]
    test_indices = indices[int(len(indices) * 0.6):]

    cal_first_retrieved_true_scores = utils.split(retrieved_true_scores, cal_first_indices)
    cal_second_retrieved_true_scores = utils.split(retrieved_true_scores, cal_second_indices)
    test_retrieved_true_scores = utils.split(retrieved_true_scores, test_indices)
    cal_first_chatgpt_true_scores = utils.split(chatgpt_true_scores, cal_first_indices)
    cal_second_chatgpt_true_scores = utils.split(chatgpt_true_scores, cal_second_indices)
    test_chatgpt_true_scores = utils.split(chatgpt_true_scores, test_indices)
    cal_first_retrieved_scores = utils.split(retrieved_scores, cal_first_indices)
    cal_second_retrieved_scores = utils.split(retrieved_scores, cal_second_indices)
    test_retrieved_scores = utils.split(retrieved_scores, test_indices)
    cal_first_chatgpt_occurances = utils.split(chatgpt_occurances, cal_first_indices)
    cal_second_chatgpt_occurances = utils.split(chatgpt_occurances, cal_second_indices)
    test_chatgpt_occurances = utils.split(chatgpt_occurances, test_indices)
    cal_first_chatgpt_semantic_ids = utils.split(chatgpt_semantic_ids, cal_first_indices)
    cal_second_chatgpt_semantic_ids = utils.split(chatgpt_semantic_ids, cal_second_indices)
    test_chatgpt_semantic_ids = utils.split(chatgpt_semantic_ids, test_indices)
    cal_first_queries = utils.split(queries, cal_first_indices)
    cal_second_queries = utils.split(queries, cal_second_indices)
    test_queries = utils.split(queries, test_indices)
    cal_first_chatgpt_answers = utils.split(chatgpt_answers, cal_first_indices)
    cal_second_chatgpt_answers = utils.split(chatgpt_answers, cal_second_indices)
    test_chatgpt_answers = utils.split(chatgpt_answers, test_indices)
    cal_first_answers = utils.split(answers, cal_first_indices)
    cal_second_answers = utils.split(answers, cal_second_indices)
    test_answers = utils.split(answers, test_indices)

    # print("Individual components")
    cal_first_scores = []
    for scores in cal_first_chatgpt_true_scores:
        cal_first_scores.append(np.max(scores))

    cal_second_scores = []
    for scores in cal_second_chatgpt_true_scores:
        cal_second_scores.append(np.max(scores))
    test_scores = []
    for scores in test_chatgpt_true_scores:
        test_scores.append(np.max(scores))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                       use_stemmer=True)
    
    print("Vanila Fisher")
    results_fisher = evaluate_fisher(
        test_retrieved_scores, test_queries,
        test_answers, test_chatgpt_answers, 
        test_chatgpt_occurances, test_chatgpt_semantic_ids)

    print('Desired coverage rate', 1-args.alpha)
    print('Coverage', np.mean(results_fisher[0]))
    print('Average answer', np.mean(results_fisher[1]))
    print('Average semantic', np.mean(results_fisher[2]))

    result = gp_minimize(func=objective_fisher,
                        dimensions=dimensions,
                        acq_func="EI",      # the acquisition function
                        n_calls=15,
                        random_state=args.seed,
                        verbose=True,
                        x0=[[1/2.0, 1/2.0]])

    print("Best fitness:", result.fun)
    print("Best parameters:", softmax(result.x))

    weights = softmax(result.x).reshape(-1, 1)
    results = evaluate_fisher(
        test_retrieved_scores, test_queries,
        test_answers, test_chatgpt_answers, 
        test_chatgpt_occurances, test_chatgpt_semantic_ids,
        w0=weights[0], w1=weights[1],
        cluster=True)
    
    print('Desired coverage rate', 1-args.alpha)
    print('Coverage', np.mean(results[0]))
    print('Average answer', np.mean(results[1]))
    print('Average semantic', np.mean(results[2]))

    print()
    print()