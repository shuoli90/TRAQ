import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import TRAC.utils as utils
from rouge_score import rouge_scorer
import random
from scipy.stats import hmean
import MHT


def filter_zeros(arr):
    return arr[arr != 0]


def concatenate_lists(list_of_lists):
    tmp = []
    for list in list_of_lists:
        tmp += list
    return tmp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--num_seed', type=int, default=10)
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

    # use 1 to 10 as random seed
    seeds = list(range(1, args.num_seed+1))
    e2e_includes_seeds = []
    context_includes_seeds = []
    average_requests_seeds = []
    unique_answer_sets_seeds = []
    total_answer_sets_seeds = []
    semantic_clusters_seeds = []
    for seed in seeds:
        print('*****Seed*****', seed)
        random.seed(seed)
        if args.semantic:
            # setup semantic model
            semantic_tokenizer = \
                AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
            semantic_model = \
                AutoModelForSequenceClassification.from_pretrained(
                    "microsoft/deberta-large-mnli"
                ).cuda()

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
        most_relevant_cosine_cal = [np.array(most_relevant_scores[i])
                                    for i in cal_indices]
        thr_most_relevant = np.quantile(most_relevant_cosine_cal, args.alpha/2)

        filename = f'true_scores_{args.n_answers}_{args.temp}_{args.metric}_{args.threshold}.json'
        with open(os.path.join("collected", filename), "r") as f:
            true_scores = json.load(f)

        filename = f'all_true_context_scores_{args.n_answers}_{args.temp}_{args.metric}_{args.threshold}.json'
        with open(os.path.join("collected", filename), "r") as f:
            all_true_context_scores = json.load(f)

        valid_calibration_indices = []
        length = len(true_scores) // args.n_answers
        for idx in range(length):
            all_score = all_scores[idx]
            include = most_relevant_scores[idx] in all_score
            if include:
                valid_calibration_indices.append(idx)
        random.shuffle(valid_calibration_indices)
        cal_indices = valid_calibration_indices[:int(len(valid_calibration_indices) * 0.5)]

        # construct calibration set
        true_scores_all = np.array(true_scores).reshape(-1, args.n_answers)
        cal_true_scores = filter_zeros(true_scores_all[cal_indices].flatten())
        # second, compute size
        all_true_context_scores = np.array(all_true_context_scores)

        # read in retrieve+qa data
        filepath = f'answers_{args.n_answers}_{args.temp}_retrieve.jsonl'
        answers_list = []
        with open(os.path.join(
                        'collected', filepath), "r") as f:
            for line in f:
                item = json.loads(line.strip())
                answers_list.append(item)
        # # setup scores
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                           use_stemmer=True)

        true_scores = []
        all_true_context_scores = []
        includes = []
        includes_context = []
        total_answer_sets = []
        unique_answer_sets = []
        semantic_clusters = []
        # retrieve_tolerate = 1 / (2/args.alpha - 1)
        retrieve_tolerate = 1 / (2*np.log(3)/args.alpha - 1)
        requests = []
        # ask questions
        with open('biencoder-nq-dev.json', "r") as src_file:
            dpr_records = json.load(src_file)
            for idx_record, dpr_record in enumerate(tqdm(dpr_records)):
                if idx_record < args.start:
                    continue
                elif (idx_record - args.start) >= \
                        (len(answers_list) // args.n_docs):
                    break
                question = dpr_record["question"].strip()
                reference_answer = dpr_record["answers"]
                new_idx = idx_record - args.start
                answers = answers_list[new_idx*args.n_docs:(new_idx+1)*args.n_docs]
                most_relevant = most_relevant_scores[idx_record]
                retrieve_scores_list = all_scores[idx_record]

                include = most_relevant in retrieve_scores_list
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

                # first, for each context, compute its p-value
                indices = []
                p_values = []
                for idx_context, score in enumerate(retrieve_scores_list):
                    p_value = np.mean(score >= np.array(most_relevant_cosine_cal))
                    if p_value >= retrieve_tolerate:
                        indices.append(idx_context)
                        p_values.append(p_value)
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
                        # compute qa p-value
                        qa_p_value = np.mean(prob >= np.array(cal_true_scores))
                        # combing p-values
                        combined_p_value = hmean([qa_p_value, p_values[idx]])
                        # combined_p_value = (qa_p_value + p_values[idx]) / 2
                        # combined_p_value = MHT.Fisher([qa_p_value, p_values[idx]])
                        if combined_p_value >= args.alpha / np.log(3):
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
                if idx_record >= args.end:
                    break
        e2e_include = np.mean(includes)
        context_include = np.mean(includes_context)
        average_requests = np.mean(requests)
        unique_answer_sets = np.mean([len(item)
                                      for item
                                      in unique_answer_sets])
        total_answer_sets = np.mean([len(item)
                                    for item
                                    in total_answer_sets])
        semantic_clusters = np.mean([len(item)
                                    for item
                                    in semantic_clusters])
        print('End-to-end coverage', e2e_include)
        print('Context include', context_include)
        print('Average requests', average_requests)
        print('Unique answer sets', unique_answer_sets)
        print('Total answer sets', total_answer_sets)
        print('Semantic clusters', semantic_clusters)
        e2e_includes_seeds.append(e2e_include)
        context_includes_seeds.append(context_include)
        average_requests_seeds.append(average_requests)
        unique_answer_sets_seeds.append(unique_answer_sets)
        total_answer_sets_seeds.append(total_answer_sets)
        semantic_clusters_seeds.append(semantic_clusters)
    print('*****Summary*****')
    print('End-to-end coverage', np.mean(e2e_includes_seeds))
    print('Context include', np.mean(context_includes_seeds))
    print('Average requests', np.mean(average_requests_seeds))
    print('Unique answer sets', np.mean(unique_answer_sets_seeds))
    print('Total answer sets', np.mean(total_answer_sets_seeds))
    print('Semantic clusters', np.mean(semantic_clusters_seeds))


if __name__ == '__main__':
    main()
