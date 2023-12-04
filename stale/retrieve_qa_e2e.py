import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import TRAC.utils as utils
from rouge_score import rouge_scorer
import random


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
    seeds = list(range(1, args.num_seed+1))
    e2e_includes_seeds = []
    # context_includes_seeds = []
    # average_requests_seeds = []
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
        # split collected data into calibration and test
        length = len(answers_list) // args.n_docs
        indices = list(range(length))
        random.shuffle(indices)
        cal_indices = indices[:int(len(indices) * 0.5)]
        test_indices = indices[int(len(indices) * 0.5):]
        true_scores = []
        test_scores = []
        all_test_uniques = []
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

                all_test_score = []
                all_test_unique = []
                for idx, answers_tmp in enumerate(answers):
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
                        prob = semantic_probs[concept_id]
                        repeat = item_occurance[predicted_answer]
                        for ref_answer in reference_answer:
                            scores = scorer.score(ref_answer,
                                                predicted_answer)
                            if args.metric == "rouge1":
                                scores = scores['rouge1'][2]
                            else:
                                scores = scores['rougeL'][2]
                            if scores >= args.threshold:
                                if new_idx in cal_indices:
                                    true_scores.append(prob)
                                else:
                                    test_scores.append(prob)
                if idx_record >= args.end:
                    break
        # compute threshold
        thr_direct = np.quantile(np.array(true_scores), args.alpha)
        coverage = np.mean(np.array(test_scores) >= thr_direct)
        # print('Direct threshold', thr_direct)
        # print('Coverage', coverage)

        total_answer_sets = []
        unique_answer_sets = []
        semantic_clusters = []
        for i, idx in enumerate(tqdm(test_indices)):
            answers = answers_list[idx*args.n_docs:(idx+1)*args.n_docs]
            question_answers = []
            for answer_tmp in answers:
                question_clusters, semantic_probs, item_occurance = \
                        utils.compute_keyword_clusterring(
                            answers=answer_tmp,
                            scorer=scorer)
                for predicted_answer in question_clusters.keys():
                    prob = semantic_probs[question_clusters[predicted_answer]]
                    repeat = item_occurance[predicted_answer]
                    if prob >= thr_direct:
                        question_answers.append([predicted_answer] * repeat)
            question_answers = concatenate_lists(question_answers)
            question_clusters, semantic_probs, _ = \
                    utils.compute_keyword_clusterring(
                        answers=question_answers,
                        scorer=scorer)
            total_answer_sets.append(question_answers)
            unique_answer_sets.append(set(question_answers))
            semantic_clusters.append([key for key in semantic_probs.keys()])
        # e2e_include = np.mean(includes)
        # context_include = np.mean(includes_context)
        # average_requests = np.mean(requests)
        unique_answer_sets = np.mean([len(item)
                                      for item
                                      in unique_answer_sets])
        total_answer_sets = np.mean([len(item)
                                    for item
                                    in total_answer_sets])
        semantic_clusters = np.mean([len(item)
                                    for item
                                    in semantic_clusters])
        print('End-to-end coverage', coverage)
        # print('Context include', context_include)
        # print('Average requests', average_requests)
        print('Unique answer sets', unique_answer_sets)
        print('Total answer sets', total_answer_sets)
        print('Semantic clusters', semantic_clusters)
        e2e_includes_seeds.append(coverage)
        # context_includes_seeds.append(context_include)
        # average_requests_seeds.append(average_requests)
        unique_answer_sets_seeds.append(unique_answer_sets)
        total_answer_sets_seeds.append(total_answer_sets)
        semantic_clusters_seeds.append(semantic_clusters)
    print('*****Summary*****')
    print('End-to-end coverage', np.mean(e2e_includes_seeds))
    # print('Context include', np.mean(context_includes_seeds))
    # print('Average requests', np.mean(average_requests_seeds))
    print('Unique answer sets', np.mean(unique_answer_sets_seeds))
    print('Total answer sets', np.mean(total_answer_sets_seeds))
    print('Semantic clusters', np.mean(semantic_clusters_seeds))


if __name__ == '__main__':
    main()
