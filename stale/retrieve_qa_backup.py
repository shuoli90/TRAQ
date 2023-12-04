import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import TRAC.utils as utils
from rouge_score import rouge_scorer


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
    parser.add_argument('--seed', type=int, default=42)
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

    if args.semantic:
        # setup semantic model
        semantic_tokenizer = \
            AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        semantic_model = \
            AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli"
            ).cuda()

    filename = f'true_scores_{args.n_answers}_{args.temp}_{args.metric}_{args.threshold}.json'
    with open(os.path.join("collected", filename), "r") as f:
        true_scores = json.load(f)

    filename = f'all_true_context_scores_{args.n_answers}_{args.temp}_{args.metric}_{args.threshold}.json'
    with open(os.path.join("collected", filename), "r") as f:
        all_true_context_scores = json.load(f)

    # filename = f'retrieve_scores_{args.retrieve_score}_{args.retrieve_metric}.json'
    # with open(os.path.join("collected", filename), "r") as f:
    #     retrieve_scores = json.load(f)

    # filename = f'all_scores_{args.retrieve_score}_{args.retrieve_metric}.json'
    # with open(os.path.join("collected", filename), "r") as f:
    #     all_scores = json.load(f)

    # construct calibration set
    true_scores_all = np.array(true_scores).reshape(-1, args.n_answers)
    length = true_scores_all.shape[0]
    indices = np.arange(length)
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    cal_indices = indices[:int(length * 0.5)]
    test_indices = indices[int(length * 0.5):]
    cal_true_scores = filter_zeros(true_scores_all[cal_indices].flatten())
    # cal_retrieve_scores = [score
    #                        for i, score in enumerate(retrieve_scores)
    #                        if i in cal_indices]
    # cal_retrieve_scores = np.array(concatenate_lists(cal_retrieve_scores))
    test_true_scores = filter_zeros(true_scores_all[test_indices].flatten())
    # test_retrieve_scores = [score
    #                         for i, score in enumerate(retrieve_scores)
    #                         if i in test_indices]
    # test_retrieve_scores = np.array(concatenate_lists(test_retrieve_scores))

    # use conformal prediction
    # first, compute quantiles for retrieve and qa
    thr_qa = np.quantile(cal_true_scores, args.alpha)
    # thr_retrieve = np.quantile(cal_retrieve_scores, args.alpha)
    print(f"QA threshold: {thr_qa}")
    # print(f"Retrieve threshold: {thr_retrieve}")

    # second, compute size
    all_true_context_scores = np.array(all_true_context_scores)
    # all_score = np.array(all_scores)
    size_qa = np.mean(np.sum(all_true_context_scores >= thr_qa, 1))
    # size_retrieve = np.mean(np.sum(all_score >= thr_retrieve, 1))
    print(f"QA size: {size_qa}")
    # print(f"Retrieve size: {size_retrieve}")
    coverage_qa = np.mean(test_true_scores >= thr_qa)
    # coverage_retrieve = np.mean(test_retrieve_scores >= thr_retrieve)
    print(f"QA coverage: {coverage_qa}")
    # print(f"Retrieve coverage: {coverage_retrieve}")

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
    filename = f'retrieve_scores_{args.retrieve_score}_{args.retrieve_metric}_{args.n_docs}.json'
    with open(os.path.join(
            'collected',
            filename), "r") as src_file:
        retrieve_scores = json.load(src_file)
    np.random.seed(args.seed)
    length = np.arange(len(most_relevant_scores))
    np.random.shuffle(length)
    calibration_indices = length[:int(len(most_relevant_scores) * 0.5)].tolist()
    test_indices = length[int(len(most_relevant_scores) * 0.5):].tolist()
    most_relevant_cosine_cal = [np.array(most_relevant_scores[i])
                                for i in calibration_indices]
    most_relevant_cosine_test = [np.array(most_relevant_scores[i])
                                 for i in test_indices]

    # within the first 1,000 questions,
    # select questions whose context contains the most relevent context
    valid_calibration_indices = []
    for idx, all_score in enumerate(all_scores[:1500]):
        # include = np.any(np.array(all_score) <= most_relevant_scores[idx])
        include = most_relevant_scores[idx] in all_score
        if include:
            valid_calibration_indices.append(idx)

    thr_most_relevant = np.quantile(most_relevant_cosine_cal, args.alpha)
    print(f"Most relevant threshold: {thr_most_relevant}")
    most_relevant_cosine_test = np.array(most_relevant_cosine_test)
    coverage_most_relevant = np.mean(most_relevant_cosine_test >= thr_most_relevant)
    print(f"Most relevant coverage: {coverage_most_relevant}")

    # read in retrieve+qa data
    filepath = f'answers_{args.n_answers}_{args.temp}_retrieve.jsonl'
    answers_list = []
    with open(os.path.join(
                    'collected', filepath), "r") as f:
        for line in f:
            item = json.loads(line.strip())
            answers_list.append(item)
    length = len(answers_list) // args.n_answers
    retrieve_scores = retrieve_scores[args.start:args.start+length]
    all_scores = all_scores[args.start:args.start+length]
    # filepath = f'{args.retrieve_score}_{args.n_answers}_{args.temp}.json'
    # with open(os.path.join('collected', filepath), "r") as f:
    #     retrieve_scores = json.load(f)
    # filepath = f'contexts_{args.n_answers}_{args.temp}.json'
    # with open(os.path.join('collected', filepath), "r") as f:
    #     contexts = json.load(f)

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
    # ask questions
    with open('biencoder-nq-dev.json', "r") as src_file:
        dpr_records = json.load(src_file)
        for idx_record, dpr_record in enumerate(tqdm(dpr_records)):
            if idx_record < args.start:
                continue
            elif (idx_record - args.start) >= len(retrieve_scores):
                break
            question = dpr_record["question"].strip()
            reference_answer = dpr_record["answers"]
            # reference_context = dpr_record["positive_ctxs"]
            new_idx = idx_record - args.start
            # contexts_list = contexts[new_idx][0]
            retrieve_scores_list = retrieve_scores[new_idx]
            answers = answers_list[new_idx*20:(new_idx+1)*20]
            most_relevant = most_relevant_scores[idx_record]
            include = False
            # for context in contexts_list:
            #     has = utils.has_answer(reference_answer, context,
            #                            tokenizer=None, match_type='regex')
            #     if has:
            #         include = True
            # if not include:
            #     continue
            include = np.any(np.array(retrieve_scores_list) <= most_relevant)
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
            # context_set = [contexts_list[idx] for idx in indices]
            answers_set = [answers[idx] for idx in indices]

            include_context = most_relevant >= thr_most_relevant
            # for context in context_set:
            #     has = utils.has_answer(reference_answer, context,
            #                            tokenizer=None, match_type='regex')
            #     if has:
            #         include_context = True
            includes_context.append(include_context)

            # second, select answer
            question_clusters = []
            question_answers = []
            include = False
            for idx, answers_tmp in enumerate(answers_set):
                # context = context_set[idx]
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

                question_clusters = []
                question_answers = []
                for predicted_answer in semantic_clusterring.keys():
                    concept_id = semantic_clusterring[predicted_answer]
                    repeat = item_occurance[predicted_answer]
                    prob = semantic_probs[concept_id]
                    if prob >= thr_qa:
                        # if prob >= 0.0:
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
    print('Valid Questions', len(includes))
    print('End-to-end coverage', np.mean(includes))
    print('Context include', np.mean(includes_context))
    print('Unique answer sets', np.mean([len(item)
                                         for item
                                         in unique_answer_sets]))
    print('Total answer sets', np.mean([len(item)
                                        for item
                                        in total_answer_sets]))
    print('Semantic clusters', np.mean([len(item)
                                        for item
                                        in semantic_clusters]))


if __name__ == '__main__':
    main()
