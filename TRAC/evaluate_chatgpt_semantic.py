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
from pac_utils import find_maximum_train_error_allow
import time
import pickle
import json
import tasks
from tqdm import tqdm
from skopt.space import Real
from skopt import gp_minimize
from skopt.utils import use_named_args
import multiprocessing
from multiprocessing import Value
import torch.multiprocessing as mp

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
    feasibilities = read_list(f'chatgpt_feasibilities_{task}.p')

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

    return feasibilities, retrieved_scores, retrieved_true_scores, queries, answers, chatgpt_true_scores, chatgpt_answers, chatgpt_passages, chatgpt_semantics, chatgpt_occurances, chatgpt_semantic_ids, chatgpt_probs, retrieved_scores_unc, retrieved_true_scores_unc, queries_unc, answers_unc, passages_unc, chatgpt_true_scores_unc, chatgpt_answers_unc, chatgpt_occurances_unc, chatgpt_semantic_ids_unc, chatgpt_probs_unc


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

def run(i, shared_includes, shared_semantic_counts, retrieved_thr, chatgpt_qa_thr, test_queries_list,
        test_retrieved_scores_list, test_answers_list, \
        test_chatgpt_semantic_ids_list, test_chatgpt_probs_list, scorer, semantic_model, semantic_tokenizer, cluster=True, kernel=40, verbose=True):
    includes = []
    semantics_total = []
    coverages = []
    semantic_counts = []
    for idx, (query_tmp, retrieved_scores_tmp, answers_tmp,\
                chatgpt_semantic_ids_tmp, chatgpt_probs_tmp) \
        in tqdm(enumerate(zip(
            test_queries_list[i], test_retrieved_scores_list[i], test_answers_list[i], \
            test_chatgpt_semantic_ids_list[i], test_chatgpt_probs_list[i])), total=len(test_retrieved_scores_list[i])):

        include = False
        retrieved_count = 0
        semantics = []
        for retrieved_score, semantic_set_ids, probs in zip(retrieved_scores_tmp, chatgpt_semantic_ids_tmp, chatgpt_probs_tmp):
                if retrieved_score < retrieved_thr:
                    continue
                else:
                    retrieved_count += 1
                    for predicted_answer in semantic_set_ids.keys():
                        concept_id = semantic_set_ids[predicted_answer]
                        prob = probs[concept_id]
                        if prob >= chatgpt_qa_thr:
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
            # semantic_set_ids, semantic_probs, item_occurance = \
            #         compute_semantic_clusterring(
            #             answers=semantics, 
            #             question=query_tmp,
            #             model=semantic_model,
            #             tokenizer=semantic_tokenizer,
            #     )
            # semantic_counts.append(len(semantic_probs.keys()))
            unique_generated_texts, item_occurance = \
                utils.unique_items(semantics)
            semantic_counts.append(len(unique_generated_texts))
        else:
            semantic_counts.append(len(semantics))
        semantics_total.append(semantic_set_ids)
#         answer_counts.append(retrieved_count)
        includes.append(include)
    # shared_includes += torch.sum(torch.tensor(includes))
    # shared_semantic_counts += torch.sum(torch.tensor(semantic_counts))
    shared_includes.value += np.sum(includes)
    shared_semantic_counts.value += np.sum(semantic_counts)

def evaluate(
        test_retrieved_scores,
        test_queries, test_answers, test_chatgpt_answers, 
        test_chatgpt_semantic_ids, test_chatgpt_probs,
        retrieved_thr, chatgpt_qa_thr, scorer, semantic_model=None, semantic_tokenizer=None,
        cluster=True, kernel=40, verbose=True):
    
    length = len(test_retrieved_scores)
    lens = np.linspace(0, length, kernel+1)
    test_queries_list = [test_queries[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_retrieved_scores_list = [test_retrieved_scores[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_chatgpt_semantic_ids_list = [test_chatgpt_semantic_ids[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_chatgpt_probs_list = [test_chatgpt_probs[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_answers_list = [test_answers[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    
    # processes = []
    # shared_includes = torch.zeros(1).cuda()
    # shared_includes.share_memory_()
    # shared_semantic_counts = torch.zeros(1).cuda()
    # shared_semantic_counts.share_memory_()

    processes = []
    shared_includes = Value('f', 0.0)
    shared_semantic_counts = Value('f', 0.0)

    # for i in range(0, kernel):
    #     p = multiprocessing.Process(target=run, args=(i, shared_includes, shared_semantic_counts))
    #     processes.append(p)
    #     p.start()

    # for process in processes:
    #     process.join()
        
    # return shared_includes.value/length, 0.0, shared_semantic_counts.value/length

    # mp.set_start_method('spawn')
    for i in range(0, kernel):
        p = multiprocessing.Process(target=run, 
                       args=(
                            i, shared_includes, shared_semantic_counts, retrieved_thr, chatgpt_qa_thr, test_queries_list, test_retrieved_scores_list, test_answers_list, \
                            test_chatgpt_semantic_ids_list, test_chatgpt_probs_list, scorer, semantic_model, semantic_tokenizer, cluster, kernel, verbose))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
    return shared_includes.value/length, 0.0, shared_semantic_counts.value/length

def run_vanila(i, shared_includes, shared_semantic_counts, test_queries_list,
        test_retrieved_scores_list, test_answers_list, \
        test_chatgpt_semantic_ids_list, test_chatgpt_probs_list, scorer, semantic_model, semantic_tokenizer, cluster=True, kernel=40, verbose=True):
        includes = []
        semantics_total = []
        coverages = []
        semantic_counts = []
        for idx, (query_tmp, retrieved_scores_tmp, answers_tmp,\
                chatgpt_semantic_ids_tmp, chatgpt_probs_tmp) \
            in tqdm(enumerate(zip(
                test_queries_list[i], test_retrieved_scores_list[i], test_answers_list[i], \
                test_chatgpt_semantic_ids_list[i], test_chatgpt_probs_list[i])), total=len(test_retrieved_scores_list[i])):

            include = False
            retrieved_count = 0
            semantics = []
            for retrieved_score, semantic_set_ids, probs in zip(retrieved_scores_tmp, chatgpt_semantic_ids_tmp, chatgpt_probs_tmp):
                retrieved_count += 1
                for predicted_answer in semantic_set_ids.keys():
                    concept_id = semantic_set_ids[predicted_answer]
                    prob = probs[concept_id]
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
                # semantic_set_ids, semantic_probs, item_occurance = \
                #     compute_semantic_clusterring(
                #         answers=semantics, 
                #         question=query_tmp,
                #         model=semantic_model,
                #         tokenizer=semantic_tokenizer,
                # )
                # semantic_counts.append(len(semantic_probs.keys()))
                unique_generated_texts, item_occurance = \
                    utils.unique_items(semantics)
                semantic_counts.append(len(unique_generated_texts))
            else:
                semantic_counts.append(len(semantics))
            semantics_total.append(semantic_set_ids)
    #         answer_counts.append(retrieved_count)
            includes.append(include)
        shared_includes.value += np.sum(includes)
        shared_semantic_counts.value += np.sum(semantic_counts)

def evaluate_vanila(
        test_retrieved_scores, 
        test_queries, test_answers, test_chatgpt_answers, 
        test_chatgpt_semantic_ids, test_chatgpt_probs,
        scorer, semantic_model=None, semantic_tokenizer=None,
        cluster=True, kernel=40, verbose=True):
    
    length = len(test_retrieved_scores)
    lens = np.linspace(0, length, kernel+1)
    test_queries_list = [test_queries[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_retrieved_scores_list = [test_retrieved_scores[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_chatgpt_semantic_ids_list = [test_chatgpt_semantic_ids[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_chatgpt_probs_list = [test_chatgpt_probs[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_answers_list = [test_answers[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    
    # processes = []
    # shared_includes = torch.zeros(1).cuda()
    # shared_includes.share_memory_()
    # shared_semantic_counts = torch.zeros(1).cuda()
    # shared_semantic_counts.share_memory_()

    processes = []
    shared_includes = Value('f', 0.0)
    shared_semantic_counts = Value('f', 0.0)

    for i in range(0, kernel):
        p = multiprocessing.Process(target=run_vanila, 
                       args=(
                            i, shared_includes, shared_semantic_counts, test_queries_list, test_retrieved_scores_list, test_answers_list, \
                            test_chatgpt_semantic_ids_list, test_chatgpt_probs_list, scorer, semantic_model, semantic_tokenizer, cluster, kernel, verbose))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    return shared_includes.value/length, 0.0, shared_semantic_counts.value/length

w1 = Real(name='w1', low=0.0, high=1.0)
w2 = Real(name='w2', low=0.0, high=1.0)

def softmax(vec):
    nom = np.exp(vec - np.mean(vec))
    return nom / np.sum(nom)

# Gather the search-space dimensions in a list.
dimensions = [w1, w2]

"""
Weight Bonf module
"""
w1 = Real(name='w1', low=0.0, high=1.0)
w2 = Real(name='w2', low=0.0, high=1.0)

# Gather the search-space dimensions in a list.
dimensions = [w1, w2]

@use_named_args(dimensions=dimensions)
def objective(w1, w2):
    weights = softmax(np.array([w1, w2])).reshape(-1, 1)
    alpha_retrieve = alpha * weights[0]
    alpha_qa = alpha * weights[1]
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve)
    cal_first_scores = []
    for scores in cal_first_chatgpt_true_scores:
        cal_first_scores.append(np.max(scores))
    chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa)
    results = evaluate(cal_second_retrieved_scores, cal_second_queries, 
                       cal_second_answers, cal_second_chatgpt_answers, \
                       cal_second_chatgpt_semantic_ids, cal_second_chatgpt_probs, 
                       retrieved_thr, chatgpt_qa_thr, scorer=scorer, semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer,
                       cluster=True)
    coverage = np.mean(results[0])
    average_answer = np.mean(results[1])
    average_semantic = np.mean(results[2])
    return average_semantic

@use_named_args(dimensions=dimensions)
def objective_pac(w1, w2):
    weights = softmax(np.array([w1, w2])).reshape(-1, 1)
    alpha_retrieve = alpha * weights[0]
    alpha_qa = alpha * weights[1]
    alpha_retrieve_pac = find_maximum_train_error_allow(alpha_retrieve, delta/2.0, len(cal_first_indices))
    alpha_qa_pac = find_maximum_train_error_allow(alpha_qa, delta/2.0, len(cal_first_indices))

    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve_pac)
    cal_first_scores = []
    for scores in cal_first_chatgpt_true_scores:
        cal_first_scores.append(np.max(scores))
    chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa_pac)
    results = evaluate(cal_second_retrieved_scores, cal_second_queries, 
                       cal_second_answers, cal_second_chatgpt_answers, \
                       cal_second_chatgpt_semantic_ids, cal_second_chatgpt_probs, 
                       retrieved_thr, chatgpt_qa_thr, scorer=scorer, semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer,
                       cluster=True)
    coverage = np.mean(results[0])
    average_answer = np.mean(results[1])
    average_semantic = np.mean(results[2])
    return average_semantic

def compute_semantic_clusterring(model, tokenizer,
                                 question, answers):
    # filter out non-unique answers
    unique_generated_texts, item_occurance = \
        utils.unique_items(answers)

    # unique_generated_texts = list(set(answers))
    semantic_set_ids = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    # print('Number of unique answers:', len(unique_generated_texts))

    with torch.no_grad():
        semantics = [0]
        for i in range(1, len(unique_generated_texts)):
            included = False
            for semantic in semantics:
                qa_1 = question + ' ' + unique_generated_texts[semantic]
                qa_2 = question + ' ' + unique_generated_texts[i]

                input = qa_1 + ' [SEP] ' + qa_2
                encoded_input = tokenizer.encode(input, padding=True)
                prediction = model(torch.tensor([encoded_input], device='cuda'))['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = tokenizer.encode(
                    reverse_input,
                    padding=True)
                reverse_prediction = model(torch.tensor([encoded_reverse_input], device='cuda'))['logits']
                reverse_predicted_label = torch.argmax(
                    reverse_prediction,
                    dim=1)

                if 2 in predicted_label and 2 in reverse_predicted_label:
                    semantic_set_ids[unique_generated_texts[i]] = \
                        semantic_set_ids[unique_generated_texts[semantic]]
                    included = True
                    break
            if not included:
                semantics.append(i)

    semantic_probs = utils.get_semantic_prob(semantic_set_ids, item_occurance)
    return semantic_set_ids, semantic_probs, item_occurance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='nq')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--semantic', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fewshot', action='store_true')
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    start = time.time()
    results_dict = {}

    print("****************************")
    print(f'Task={args.task}, Fewshot={args.fewshot}, alpha={args.alpha}, seed={args.seed}, semantic={args.semantic}')
    print("****************************")
    if args.fewshot:
        results_dict["task"] = "fewshot"
    else:
        results_dict["task"] = args.task
    results_dict["alpha"] = args.alpha
    results_dict["seed"] = args.seed
    results_dict["semantic"] = args.semantic
    alpha = args.alpha
    seed = args.seed
    task = args.task
    # semantic = args.semantic
    semantic = True

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                            use_stemmer=True)
    if semantic:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        # setup semantic model
        semantic_tokenizer = \
            AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        semantic_model = \
            AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli"
            ).cuda()

    if args.fewshot:
        feasibilities, retrieved_scores, retrieved_true_scores, queries, answers, chatgpt_true_scores, chatgpt_answers, chatgpt_passages, chatgpt_semantics, chatgpt_occurances, chatgpt_semantic_ids, chatgpt_probs, retrieved_scores_unc, retrieved_true_scores_unc, queries_unc, answers_unc, passages_unc, chatgpt_true_scores_unc, chatgpt_answers_unc, chatgpt_occurances_unc, chatgpt_semantic_ids_unc, chatgpt_probs_unc = \
            read_chatgpt_results("nq_fewshot")
    else:

        feasibilities, retrieved_scores, retrieved_true_scores, queries, answers, chatgpt_true_scores, chatgpt_answers, chatgpt_passages, chatgpt_semantics, chatgpt_occurances, chatgpt_semantic_ids, chatgpt_probs, retrieved_scores_unc, retrieved_true_scores_unc, queries_unc, answers_unc, passages_unc, chatgpt_true_scores_unc, chatgpt_answers_unc, chatgpt_occurances_unc, chatgpt_semantic_ids_unc, chatgpt_probs_unc = \
        read_chatgpt_results(task)
    dataset_dpr = tasks.RQA_dpr(task)
    elements = dataset_dpr.elements
    all_queries = [element['question'] for element in elements]
    answers = []
    for query in queries:
        idx = all_queries.index(query)
        answers.append(elements[idx]['answers'])

    answers_semantic = []
    for true_score, scores, returned_answers in zip(retrieved_true_scores, retrieved_scores, chatgpt_answers):
        idx = list(scores).index(true_score)
        tmp = returned_answers[idx]
        answers_semantic.append(tmp)
    
    if args.collect:
        chatgpt_true_scores_semantic = []
        skip = []
        for idx, [query_tmp, true_answer, generated_answer] in tqdm(enumerate(zip(queries, answers, answers_semantic)), total=len(queries)):
            prompt = utils.get_prompt_template(query_tmp, "", task="Natural Questions")
            semantic_set_ids, semantic_probs, item_occurance = \
                compute_semantic_clusterring(
                    semantic_model, semantic_tokenizer,
                    query_tmp, generated_answer)
            true_scores, matched_answer, semantics = utils.processing_answers(
                semantic_set_ids, semantic_probs, 
                item_occurance, true_answer, scorer,
                threshold=0.3
            )
            if len(true_scores) == 0:
                print(idx)
                skip.append(idx)
                continue
            chatgpt_true_scores_semantic.append(true_scores)
        answers_semantic = []
        for idx, answer in enumerate(answers):
            if idx in skip:
                continue
            answers_semantic.append(answer)
        write_list(answers_semantic, f'chatgpt_true_answers_{task}_semantic.p')

        occurances_semantic = []
        semantic_ids_semantic = []
        probs_semantic = []
        for idx, [query_tmp, generated_answers] in tqdm(enumerate(zip(queries, chatgpt_answers)), total=len(queries)):
            if idx in skip:
                continue
            probs_tmp = []
            answers_tmp = []
            semantic_id_tmp = []
            occurance_tmp = []
            semantic_tmp = []
            for generated_answer in generated_answers:
                prompt = utils.get_prompt_template(query_tmp, "", task="Natural Questions")
                semantic_set_ids, semantic_probs, item_occurance = \
                    compute_semantic_clusterring(
                    answers=generated_answer, 
                    question=query_tmp,
                    model=semantic_model,
                    tokenizer=semantic_tokenizer,
                )
                probs_tmp.append(semantic_probs)
                occurance_tmp.append(item_occurance)
                semantic_id_tmp.append(semantic_set_ids)
            occurances_semantic.append(occurance_tmp)
            semantic_ids_semantic.append(semantic_id_tmp)
            probs_semantic.append(probs_tmp)
        
            if idx % 10 == 0:
                write_list(chatgpt_true_scores_semantic, f'chatgpt_true_scores_{task}_semantic.p')
                write_list(occurances_semantic, f'chatgpt_occurances_{task}_semantic.p')
                write_list(semantic_ids_semantic, f'chatgpt_semantic_ids_{task}_semantic.p')
                write_list(probs_semantic, f'chatgpt_probs_{task}_semantic.p')

        write_list(chatgpt_true_scores_semantic, f'chatgpt_true_scores_{task}_semantic.p')
        write_list(occurances_semantic, f'chatgpt_occurances_{task}_semantic.p')
        write_list(semantic_ids_semantic, f'chatgpt_semantic_ids_{task}_semantic.p')
        write_list(probs_semantic, f'chatgpt_probs_{task}_semantic.p')
    
    # chatgpt_true_scores_semantic = read_list(f'chatgpt_true_scores_{task}_semantic.p')
    # chatgpt_answers_occurances_semantic = read_list(f'chatgpt_occurances_{task}_semantic.p')
    # chatgpt_semantic_ids_semantic = read_list(f'chatgpt_semantic_ids_{task}_semantic.p')
    # chatgpt_probs_semantic = read_list(f'chatgpt_probs_{task}_semantic.p')
    chatgpt_true_scores = read_list(f'chatgpt_true_scores_{task}_semantic.p')
    chatgpt_answers_occurances = read_list(f'chatgpt_occurances_{task}_semantic.p')
    chatgpt_semantic_ids = read_list(f'chatgpt_semantic_ids_{task}_semantic.p')
    chatgpt_probs = read_list(f'chatgpt_probs_{task}_semantic.p')
    answers = read_list(f'chatgpt_true_answers_{task}_semantic.p')
    
    # for count, idx in enumerate(skip):
    #     retrieved_scores.pop(idx - count)
    #     retrieved_true_scores.pop(idx - count)
    #     queries.pop(idx - count)
    #     answers.pop(idx - count)
    #     chatgpt_true_scores.pop(idx - count)
    #     chatgpt_answers.pop(idx - count)
    #     chatgpt_passages.pop(idx - count)
    #     chatgpt_occurances.pop(idx - count)
    #     chatgpt_semantic_ids.pop(idx - count)
    #     chatgpt_probs.pop(idx - count)
    
    # chatgpt_true_scores = chatgpt_true_scores_semantic

    indices = np.arange(len(retrieved_true_scores))
    # indices = np.arange(10)
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

    cal_first_chatgpt_probs = utils.split(chatgpt_probs, cal_first_indices)
    cal_second_chatgpt_probs = utils.split(chatgpt_probs, cal_second_indices)
    test_chatgpt_probs = utils.split(chatgpt_probs, test_indices)

    cal_first_queries = utils.split(queries, cal_first_indices)
    cal_second_queries = utils.split(queries, cal_second_indices)
    test_queries = utils.split(queries, test_indices)

    cal_first_chatgpt_answers = utils.split(chatgpt_answers, cal_first_indices)
    cal_second_chatgpt_answers = utils.split(chatgpt_answers, cal_second_indices)
    test_chatgpt_answers = utils.split(chatgpt_answers, test_indices)

    cal_first_answers = utils.split(answers, cal_first_indices)
    cal_second_answers = utils.split(answers, cal_second_indices)
    test_answers = utils.split(answers, test_indices)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                       use_stemmer=True)

    # print("Individual components")
    # retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=args.alpha/2)
    # cal_first_scores = []
    # for scores in cal_first_chatgpt_true_scores:
    #     cal_first_scores.append(np.max(scores))
    # chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=args.alpha/2)

    # retrieved_coverage = np.mean(np.array(cal_second_retrieved_true_scores) >= retrieved_thr)
    # cal_second_scores = []
    # for scores in cal_second_chatgpt_true_scores:
    #     cal_second_scores.append(np.max(scores))
    # qa_coverage = np.mean(np.array(cal_second_scores) >= chatgpt_qa_thr)
    # print('retrieval coverage', retrieved_coverage)
    # print('qa coverage', qa_coverage)
    # print('retrieve thr', retrieved_thr)
    # print('qa thr', chatgpt_qa_thr)

    # retrieved_coverage = np.mean(np.array(test_retrieved_true_scores) >= retrieved_thr)
    # test_scores = []
    # for scores in test_chatgpt_true_scores:
    #     test_scores.append(np.max(scores))
    # qa_coverage = np.mean(np.array(test_scores) >= chatgpt_qa_thr)
    # print('test retrieval coverage', retrieved_coverage)
    # print('test qa coverage', qa_coverage)

    # results_dict["retrieval_coverage"] = retrieved_coverage
    # results_dict["qa_coverage"] = qa_coverage

    # coverages = coverage(
    #     test_retrieved_true_scores, 
    #     test_chatgpt_true_scores,
    #     retrieved_thr,
    #     chatgpt_qa_thr
    #     )
    # print('End-to-end coverage', np.mean(coverages))
    # results_dict["end_to_end_coverage"] = np.mean(coverages)

    # print("Individual compponents with PAC")
    # delta = 0.1
    # retrieve_alpha = find_maximum_train_error_allow(args.alpha/2.0, delta/2.0, len(cal_first_indices))
    # qa_alpha = find_maximum_train_error_allow(args.alpha/2.0, delta/2.0, len(cal_first_indices))
    # retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=retrieve_alpha)
    # cal_first_scores = []
    # for scores in cal_first_chatgpt_true_scores:
    #     cal_first_scores.append(np.max(scores))
    # chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=qa_alpha)

    # retrieved_coverage = np.mean(np.array(cal_second_retrieved_true_scores) >= retrieved_thr)
    # cal_second_scores = []
    # for scores in cal_second_chatgpt_true_scores:
    #     cal_second_scores.append(np.max(scores))
    # qa_coverage = np.mean(np.array(cal_second_scores) >= chatgpt_qa_thr)
    # print('retrieval coverage', retrieved_coverage)
    # print('qa coverage', qa_coverage)

    # retrieved_coverage = np.mean(np.array(test_retrieved_true_scores) >= retrieved_thr)
    # test_scores = []
    # for scores in test_chatgpt_true_scores:
    #     test_scores.append(np.max(scores))
    # qa_coverage = np.mean(np.array(test_scores) >= chatgpt_qa_thr)
    # print('test retrieval coverage', retrieved_coverage)
    # print('test qa coverage', qa_coverage)

    # coverages = coverage(
    #     test_retrieved_true_scores, 
    #     test_chatgpt_true_scores,
    #     retrieved_thr,
    #     chatgpt_qa_thr
    #     )
    # print('End-to-end coverage', np.mean(coverages))

    # results_dict["retrieval_coverage_pac"] = retrieved_coverage
    # results_dict["qa_coverage_pac"] = qa_coverage
    # results_dict["end_to_end_coverage_pac"] = np.mean(coverages)

    # kernel = 5

    # """
    # Weight Bonf module
    # """
    # result = gp_minimize(
    #     func=objective,
    #     dimensions=dimensions,
    #     acq_func="EI",      # the acquisition function
    #     n_calls=15,
    #     random_state=args.seed,
    #     verbose=False,
    #     x0=[[1, 1]])

    # print("Best fitness:", result.fun)
    # print("Best parameters:", softmax(result.x))

    # weights = softmax(result.x).reshape(-1, 1)
    # alpha_retrieve = alpha * weights[0]
    # alpha_qa = alpha * weights[1]
    # retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve)
    # cal_first_scores = []
    # for scores in cal_first_chatgpt_true_scores:
    #     cal_first_scores.append(np.max(scores))
    # chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa)
    # results = evaluate(
    #     test_retrieved_scores, test_queries,
    #     test_answers, test_chatgpt_answers, 
    #     test_chatgpt_semantic_ids, test_chatgpt_probs,
    #     retrieved_thr, chatgpt_qa_thr, scorer=scorer, semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer,
    #     cluster=True)

    # print('TRAC')
    # print('Desired coverage rate', 1-args.alpha)
    # print('Coverage', np.mean(results[0]))
    # # print('Average answer', np.mean(results[1]))
    # print('Average semantic', np.mean(results[2]))

    # results_dict["TRAC_coverage"] = np.mean(results[0])
    # results_dict["TRAC_average_semantic"] = np.mean(results[2])

    # alpha_retrieve = alpha * (1/2.0)
    # alpha_qa = alpha * (1/2.0)
    # retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve)
    # cal_first_scores = []
    # for scores in cal_first_chatgpt_true_scores:
    #     cal_first_scores.append(np.max(scores))
    # chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa)
    # results = evaluate(
    #     test_retrieved_scores, test_queries,
    #     test_answers, test_chatgpt_answers, 
    #     test_chatgpt_semantic_ids, test_chatgpt_probs,
    #     retrieved_thr, chatgpt_qa_thr, scorer=scorer, semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer,
    #     cluster=True, kernel=kernel)
    # print('Bonf')
    # print('Desired coverage rate', 1-args.alpha)
    # print('Coverage', np.mean(results[0]))
    # print('Average semantic', np.mean(results[2]))

    # results_dict["Bonf_coverage"] = np.mean(results[0])
    # results_dict["Bonf_average_semantic"] = np.mean(results[2])

    # alpha_retrieve = alpha * (1/2.0)
    # alpha_qa = alpha * (1/2.0)

    # delta = 0.1
    # alpha_retrieve_pac = find_maximum_train_error_allow(alpha_retrieve, delta/2.0, len(cal_first_indices))
    # alpha_qa_pac = find_maximum_train_error_allow(alpha_qa, delta/2.0, len(cal_first_indices))

    # retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve_pac)
    # cal_first_scores = []
    # for scores in cal_first_chatgpt_true_scores:
    #     cal_first_scores.append(np.max(scores))
    # chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa_pac)
    # results = evaluate(
    #     test_retrieved_scores, test_queries,
    #     test_answers, test_chatgpt_answers, 
    #     test_chatgpt_semantic_ids, test_chatgpt_probs,
    #     retrieved_thr, chatgpt_qa_thr, scorer=scorer, semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer,
    #     cluster=True, kernel=kernel)
    # print('PAC-Bonf')
    # print('Desired coverage rate', 1-args.alpha)
    # print('Coverage', np.mean(results[0]))
    # # print('Average answer', np.mean(results[1]))
    # print('Average semantic', np.mean(results[2]))

    # results_dict["PAC_Bonf_coverage"] = np.mean(results[0])
    # # results_dict["PAC_Bonf_average_answer"] = np.mean(results[1])   
    # results_dict["PAC_Bonf_average_semantic"] = np.mean(results[2])

    # """
    # Weight PAC-TRAC module
    # """
    # result = gp_minimize(
    #     func=objective_pac,
    #     dimensions=dimensions,
    #     acq_func="EI",      # the acquisition function
    #     n_calls=15,
    #     random_state=args.seed,
    #     verbose=False,
    #     x0=[[1, 1]])

    # print("Best fitness:", result.fun)
    # print("Best parameters:", softmax(result.x))

    # weights = softmax(result.x).reshape(-1, 1)
    # alpha_retrieve = alpha * weights[0]
    # alpha_qa = alpha * weights[1]
    # alpha_retrieve_pac = find_maximum_train_error_allow(alpha_retrieve, delta/2.0, len(cal_first_indices))
    # alpha_qa_pac = find_maximum_train_error_allow(alpha_qa, delta/2.0, len(cal_first_indices))
    # retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha_retrieve_pac)
    # cal_first_scores = []
    # for scores in cal_first_chatgpt_true_scores:
    #     cal_first_scores.append(np.max(scores))
    # chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha_qa_pac)
    # results = evaluate(
    #     test_retrieved_scores, test_queries,
    #     test_answers, test_chatgpt_answers, 
    #     test_chatgpt_semantic_ids, test_chatgpt_probs,
    #     retrieved_thr, chatgpt_qa_thr, scorer=scorer, semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer,
    #     cluster=True, kernel=kernel)

    # print('PAC-TRAC')
    # print('Desired coverage rate', 1-args.alpha)
    # print('Coverage', np.mean(results[0]))
    # # print('Average answer', np.mean(results[1]))
    # print('Average semantic', np.mean(results[2]))

    # results_dict["PAC_TRAC_coverage"] = np.mean(results[0])
    # results_dict["PAC_TRAC_average_semantic"] = np.mean(results[2])

    results = evaluate_vanila(
        retrieved_scores, queries,
        answers, chatgpt_answers, 
        chatgpt_semantic_ids, chatgpt_probs,
        scorer=scorer, semantic_model=semantic_model, semantic_tokenizer=semantic_tokenizer,
        cluster=True, kernel=20)

    print('Vanila')
    print('Desired coverage rate', 1-args.alpha)
    print('Coverage', np.mean(results[0]))
    print('Average answer', np.mean(results[1]))
    print('Average semantic', np.mean(results[2]))

    results_dict["Vanila_coverage"] = np.mean(results[0])
    # results_dict["PAC_Bonf_average_answer"] = np.mean(results[1])   
    results_dict["Vanila_average_semantic"] = np.mean(results[2])
    print()
    print()
    
    with open("chatgpt_results_vanila.txt", "a") as f:
        json.dump(results_dict, f)
        f.write('\n')