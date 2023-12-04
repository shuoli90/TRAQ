import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import TRAC.utils as utils
from rouge_score import rouge_scorer
import random
from skopt.space import Real
from skopt import gp_minimize
from skopt.utils import use_named_args


def filter_zeros(arr):
    return arr[arr != 0]


def concatenate_lists(list_of_lists):
    tmp = []
    for list in list_of_lists:
        tmp += list
    return tmp


def evaluate(indices_list, answers_list,
             retrieve_scores, args, dpr_records,
             most_relevant_scores, scorer,
             thr_most_relevant, thr_qa,
             save=False, task='baseline'):

    true_scores = []
    all_true_context_scores = []
    includes = []
    includes_context = []
    total_answer_sets = []
    unique_answer_sets = []
    semantic_clusters = []
    requests = []
    for idx in tqdm(range(len(answers_list))):
        answers = answers_list[idx]
        retrieve_scores_list = retrieve_scores[idx]
        index = indices_list[idx]
        reference_answer = dpr_records[index]["answers"]
        question = dpr_records[index]["question"].strip()
        most_relevant = most_relevant_scores[index]
        # context = context_list[idx]

        include = most_relevant in retrieve_scores_list
        # include = np.any(np.array(retrieve_scores_list) <= most_relevant)
        if include:
            include_context = most_relevant >= thr_most_relevant
            includes_context.append(include_context)
        if not include:
            continue

        answers_tmp = concatenate_lists(answers)
        include = False
        for answer_tmp in answers_tmp:
            for ref_answer in reference_answer:
                scores = scorer.score(ref_answer,
                                      answer_tmp)
                if args.metric == "rouge1":
                    scores = scores['rouge1'][2]
                else:
                    scores = scores['rougeL'][2]
                if scores >= args.threshold:
                    include = True
        if not include:
            continue

        # first, select context
        indices = []
        for idx_context, score in enumerate(retrieve_scores_list):
            if score >= thr_most_relevant:
                indices.append(idx_context)
        answers_set = [answers[idx]
                       for idx in indices]
        requests.append(len(answers_set))

        includes_context.append(include_context)

        # second, select answer
        question_answers = []
        include = False
        for idx, answers_tmp in enumerate(answers_set):
            if args.semantic:
                semantic_clusterring, semantic_probs, item_occurance = \
                    utils.compute_semantic_clusterring(
                        model=semantic_model,
                        tokenizer=semantic_tokenizer,
                        question=question, answers=answers_tmp,
                        scorer=scorer)
            else:
                semantic_clusterring, semantic_probs, item_occurance = \
                    utils.compute_keyword_clusterring(
                        answers=answers_tmp,
                        scorer=scorer)

            for predicted_answer in semantic_clusterring.keys():
                concept_id = semantic_clusterring[predicted_answer]
                repeat = item_occurance[predicted_answer]
                prob = semantic_probs[concept_id]
                if prob >= thr_qa:
                    question_answers.append([predicted_answer] * repeat)
                    for ref_answer in reference_answer:
                        scores = scorer.score(ref_answer,
                                              predicted_answer)
                        if args.metric == "rouge1":
                            scores = scores['rouge1'][2]
                        else:
                            scores = scores['rougeL'][2]
                        if scores >= args.threshold:
                            include = True
        # flatten both lists
        question_answers = concatenate_lists(question_answers)
        # cluster answers
        question_clusters, semantic_probs, _ = \
            utils.compute_keyword_clusterring(
                answers=question_answers,
                scorer=scorer)
        total_answer_sets.append(question_answers)
        unique_answer_sets.append(set(question_answers))
        semantic_clusters.append([key for key in semantic_probs.keys()])
        includes.append(include)
    e2e_include = np.mean(includes)
    context_include = np.mean(includes_context)
    average_requests = np.mean(requests)
    if save:
        with open(f'Bonf_unique_answer_sets_{task}_{args.alpha}.json', 'w') as f:
            json.dump([list(item) for item in unique_answer_sets], f)
    unique_answer_sets = np.mean([len(item)
                                  for item
                                  in unique_answer_sets])
    total_answer_sets = np.mean([len(item)
                                 for item
                                 in total_answer_sets])
    semantic_clusters = np.mean([len(item)
                                 for item
                                 in semantic_clusters])
    return [e2e_include, context_include, average_requests,
            unique_answer_sets, total_answer_sets, semantic_clusters]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--num_seed', type=int, default=5)
    parser.add_argument('--n_answers', type=int, default=40)
    parser.add_argument('--n_docs', type=int, default=20)
    parser.add_argument('--temp', type=float, default=1.5)
    parser.add_argument('--metric', type=str, default='rouge1')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--retrieve_metric', type=str, default='match')
    parser.add_argument('--retrieve_score', type=str, default='innerproduct')
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--start', type=int, default=1800)
    parser.add_argument('--end', type=int, default=2100)
    args = parser.parse_args()

    print('Alpha', args.alpha)
    print('QA metric', args.metric)
    print('QA threshold', args.threshold)
    print('Retrieve_metric', args.retrieve_metric)
    print('Retrieve_score', args.retrieve_score)
    seeds = list(range(1, args.num_seed+1))
    starts = [1500, 1800, 2133, 2930, 3790]

    answers_list = []
    for start in starts:
        filepath = f'answers_{args.n_answers}_{args.temp}_retrieve_{start}_feasible.jsonl'
        with open(os.path.join(
                        'collected', filepath), "r") as f:
            tmp = []
            for idx, line in enumerate(f):
                item = json.loads(line.strip())
                tmp.append(item)
                if (idx+1) % 20 == 0:
                    answers_list.append(tmp)
                    tmp = []
    indices = []
    for start in starts:
        filepath = f'indices_{args.n_answers}_{args.temp}_retrieve_{start}_feasible.json'
        with open(os.path.join('collected',
                                filepath), "r") as f:
            idx = json.loads(f.readline().strip())
            indices.extend(idx)
    retrieve_scores = []
    for start in starts:
        filepath = f'{args.retrieve_score}_{args.n_answers}_{args.temp}_{start}_feasible.json'
        with open(os.path.join('collected',
                                filepath), "r") as f:
            retrieve_score = json.load(f)
        filepath = f'indices_{args.n_answers}_{args.temp}_retrieve_{start}_feasible.json'
        with open(os.path.join('collected',
                                filepath), "r") as f:
            idx = json.loads(f.readline().strip())
        for i in idx:
            retrieve_scores.append(retrieve_score[i-start])
    
    
    context_list = []
    for start in starts:
        filepath = f'contexts_{args.n_answers}_{args.temp}_{start}_feasible.json' 
        with open(os.path.join(
                        'collected', filepath), "r") as f:
            context = json.load(f)
        filepath = f'indices_{args.n_answers}_{args.temp}_retrieve_{start}_feasible.json'
        with open(os.path.join('collected',
                                filepath), "r") as f:
            idx = json.loads(f.readline().strip())
        for i in idx:
            context_list.append(context[i-start])

    filename = f'most_relevant_{args.retrieve_score}_{args.retrieve_metric}_{args.n_docs}.json'
    with open(os.path.join(
            'collected',
            filename), "r") as src_file:
        most_relevant_scores = json.load(src_file)
    filename = f'all_scores_{args.retrieve_score}_{args.retrieve_metric}_{args.n_docs}.json'
    with open(os.path.join(
            'collected',
            filename), "r") as src_file:
        all_scores = json.load(src_file)

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        # within the first 1,000 questions,
        # select questions whose context contains the most relevent context
        valid_calibration_indices = []
        for idx, all_score in enumerate(all_scores[:1500]):
            # include = np.any(np.array(all_score) <= most_relevant_scores[idx])
            include = most_relevant_scores[idx] in all_score
            if include:
                valid_calibration_indices.append(idx)
        random.shuffle(valid_calibration_indices)
        cal_indices = valid_calibration_indices[:int(len(valid_calibration_indices) * 0.5)]
        test_indices = valid_calibration_indices[int(len(valid_calibration_indices) * 0.5):]

        most_relevant_cosine_cal = [np.array(most_relevant_scores[i])
                                    for i in cal_indices]
        most_relevant_cosine_test = [np.array(most_relevant_scores[i])
                                     for i in test_indices]

        thr_most_relevant = np.quantile(most_relevant_cosine_cal, args.alpha/2, method='higher')
        print(f"Most relevant threshold: {thr_most_relevant}")
        most_relevant_cosine_test = np.array(most_relevant_cosine_test)
        coverage_most_relevant = np.mean(most_relevant_cosine_test >= thr_most_relevant)
        print(f"Most relevant coverage: {coverage_most_relevant}")

        filename = f'true_scores_{args.n_answers}_{args.temp}_{args.metric}_{args.threshold}.json'
        with open(os.path.join("collected", filename), "r") as f:
            true_scores = json.load(f)

        filename = f'all_true_context_scores_{args.n_answers}_{args.temp}_{args.metric}_{args.threshold}.json'
        with open(os.path.join("collected", filename), "r") as f:
            all_true_context_scores = json.load(f)

        valid_calibration_indices = []
        length = len(true_scores) // args.n_answers
        # length = len(true_scores)
        for idx in range(length):
            all_score = all_scores[idx]
            include = most_relevant_scores[idx] in all_score
            if include:
                valid_calibration_indices.append(idx)
        random.shuffle(valid_calibration_indices)
        cal_indices = valid_calibration_indices[:int(len(valid_calibration_indices) * 0.5)]
        test_indices = valid_calibration_indices[int(len(valid_calibration_indices) * 0.5):]

        # construct calibration set
        true_scores_all = np.array(true_scores).reshape(-1, args.n_answers)
        cal_true_scores = filter_zeros(true_scores_all[cal_indices].flatten())
        test_true_scores = filter_zeros(true_scores_all[test_indices].flatten())

        # use conformal prediction
        # first, compute quantiles for retrieve and qa
        thr_qa = np.quantile(cal_true_scores, args.alpha/2, method='higher')
        print(f"QA threshold: {thr_qa}")

        # second, compute size
        all_true_context_scores = np.array(all_true_context_scores)
        size_qa = np.mean(np.sum(all_true_context_scores >= thr_qa, 1))
        print(f"QA size: {size_qa}")
        coverage_qa = np.mean(test_true_scores >= thr_qa)
        print(f"QA coverage: {coverage_qa}")

        # # setup scores
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                          use_stemmer=True)

        # split indices into calibration, test, and dev
        length = np.arange(len(indices))
        np.random.shuffle(length)
        calibration_indices = length[:int(len(indices) * 0.5)].tolist()
        test_indices = length[int(len(indices) * 0.5):].tolist()
        calibration = [indices[i] for i in calibration_indices]
        test = [indices[i] for i in test_indices]

        # split collected data into calibration, test, and dev
        train_answers = [answers_list[i]
                         for i in calibration_indices]
        test_answers = [answers_list[i]
                        for i in test_indices]
        train_retrieve_scores = [retrieve_scores[i]
                                 for i in calibration_indices]
        test_retrieve_scores = [retrieve_scores[i]
                                for i in test_indices]
        # train_context = [context_list[i]
        #                  for i in calibration_indices]
        # test_context = [context_list[i]
        #                 for i in test_indices]

        with open('biencoder-nq-dev.json', "r") as src_file:
            dpr_records = json.load(src_file)
        weights = np.array([0.5, 0.5])
        # weights = np.array([0.29506372, 0.70493628])
        alpha_retrieve = args.alpha * weights[0]
        alpha_qa = args.alpha * weights[1]
        thr_most_relevant = np.quantile(most_relevant_cosine_cal, 
                                        alpha_retrieve)
        thr_qa = np.quantile(cal_true_scores, alpha_qa)
        results = evaluate(test, test_answers,
                           test_retrieve_scores,
                           args, dpr_records,
                           most_relevant_scores, scorer,
                           thr_most_relevant,
                           thr_qa=thr_qa, save=True)
        # results = evaluate(calibration, train_answers,
        #                    train_retrieve_scores,
        #                    args, dpr_records,
        #                    most_relevant_scores, scorer,
        #                    thr_most_relevant,
        #                    thr_qa=thr_qa, save=True)
        # return [e2e_include, context_include, average_requests,
        # unique_answer_sets, total_answer_sets, semantic_clusters]
        print('Baseline coverage', results[0])
        print('Baseline average request', results[2])
        print('Baseline unique answer sets', results[3])
        print('Baseline total answer sets', results[4])
        print('Baseline semantic clusters', results[5])

        """
        Weight HMP module
        """
        w1 = Real(name='w1', low=0.0, high=1.0)
        w2 = Real(name='w2', low=0.0, high=1.0)

        def softmax(vec):
            nom = np.exp(vec - np.mean(vec))
            return nom / np.sum(nom)

        # Gather the search-space dimensions in a list.
        dimensions = [w1, w2]

        @use_named_args(dimensions=dimensions)
        def my_objective_function(w1, w2):
            weights = softmax(np.array([w1, w2])).reshape(-1, 1)
            alpha_retrieve = args.alpha * weights[0]
            alpha_qa = args.alpha * weights[1]
            thr_most_relevant = np.quantile(most_relevant_cosine_cal,
                                            alpha_retrieve, method='higher')
            thr_qa = np.quantile(cal_true_scores, alpha_qa, method='higher')
            results = evaluate(calibration, train_answers,
                               train_retrieve_scores,
                               args, dpr_records,
                               most_relevant_scores, scorer,
                               thr_most_relevant,
                               thr_qa=thr_qa)
            # return [e2e_include, context_include, average_requests,
            # unique_answer_sets, total_answer_sets, semantic_clusters]
            metric = results[3]
            return metric

        result = gp_minimize(func=my_objective_function,
                             dimensions=dimensions,
                             acq_func="EI",      # the acquisition function
                             n_calls=10,
                             random_state=seed,
                             verbose=False)

        print("Best fitness:", result.fun)
        print("Best parameters:", softmax(result.x))

        w1 = result.x[0]
        w2 = result.x[1]
        # weights = softmax(np.array([0.29506372, 0.70493628]))
        weights = softmax(np.array([w1, w2]))
        alpha_retrieve = args.alpha * weights[0]
        alpha_qa = args.alpha * weights[1]
        thr_most_relevant = np.quantile(most_relevant_cosine_cal,
                                        alpha_retrieve)
        thr_qa = np.quantile(cal_true_scores, alpha_qa)
        results = evaluate(test, test_answers,
                           test_retrieve_scores,
                           args, dpr_records,
                           most_relevant_scores, scorer,
                           thr_most_relevant,
                           thr_qa=thr_qa, save=True, task='weighted')
        print('Opt coverage', results[0])
        print('Opt average request', results[2])
        print('Opt unique answer sets', results[3])
        print('Opt total answer sets', results[4])
        print('Opt semantic clusters', results[5])


if __name__ == '__main__':
    main()
