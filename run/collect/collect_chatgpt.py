import os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from misc import tasks
from misc import utils
from rouge_score import rouge_scorer
import random
import numpy as np
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from datasets import load_dataset
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune llama-2 model on specific dataset")
    parser.add_argument(
        "--task",
        type=str,
        default="nq",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--semantic",
        action='store_true',
    )
    args = parser.parse_args()
    task = args.task
    alpha = args.alpha
    torch.set_grad_enabled(False)
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", device_map='cuda')
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", device_map='cuda')
    wiki = load_dataset(path='wiki_dpr', name='psgs_w100.multiset.compressed', split='train')
    dataset_dpr = tasks.RQA_dpr(task=task)

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

    indices = np.arange(len(dataset_dpr.elements))
    random.shuffle(indices)
    cal_indices = indices[:int(len(indices) * 0.5)]
    test_indices = indices[int(len(indices) * 0.5):]

    elements = dataset_dpr.elements
    query = [element['question'] for element in elements]

    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")

    question_embedding = q_encoder(**q_tokenizer(query, return_tensors="pt", padding=True))
    question_embedding = question_embedding[0].numpy()
    retrieved_scores, retrieved_examples = wiki.get_nearest_examples_batch('embeddings', question_embedding, k=20)

    import pickle
    def write_list(a_list, file_name):
        # store list in binary file so 'wb' mode
        with open(file_name, 'wb') as fp:
            pickle.dump(a_list, fp)
    #         print('Done writing list into a binary file')
    def read_list(file_name):
        # for reading also binary mode is important
        with open(file_name, 'rb') as fp:
            n_list = pickle.load(fp)
            return n_list

    def save_results(task):
        # save retrieved_scores to a pickle file
        write_list(retrieved_scores, f'chatgpt_retrieved_scores_{task}.p')
        # save retrieved_true_scores to a pickle file
        write_list(retrieved_true_scores, f'chatgpt_retrieved_true_scores_{task}.p')
        # save queries to a pickle file
        write_list(queries, f'chatgpt_queries_{task}.p')
        # save answers to a pickle file
        write_list(answers, f'chatgpt_true_answers_{task}.p')
        # save passages to a pickle file
        write_list(passages, f'chatgpt_passages_{task}.p')
        # save chatgpt_true_scores to a pickle file
        write_list(chatgpt_true_scores, f'chatgpt_true_scores_{task}.p')
        # save chatgpt_texts to a pickle file
    #     write_list(chatgpt_texts, f'chatgpt_texts_{task}.p')
        # save chatgpt_answers to a pickle file
        write_list(chatgpt_answers, f'chatgpt_answers_{task}.p')
        # save chatgpt_semantics to a picle file
        write_list(chatgpt_semantics, f'chatgpt_semantics_{task}.p')
        # save feasibilities to a pickle file
        write_list(feasibilities, f'chatgpt_feasibilities_{task}.p')
        # save occurances to a pickle file
        write_list(occurances, f'chatgpt_occurances_{task}.p')
        # save semantic_ids to a pickle file
        write_list(semantic_ids, f'chatgpt_semantic_ids_{task}.p')
        # save probs to a picle file
        write_list(probs, f'chatgpt_probs_{task}.p')
        
        write_list(retrieved_scores_unc, f'chatgpt_retrieved_scores_unc_{task}.p')
        write_list(retrieved_true_scores_unc, f'chatgpt_retrieved_true_scores_unc_{task}.p')
        write_list(queries_unc, f'chatgpt_queries_unc_{task}.p')
        write_list(answers_unc, f'chatgpt_answers_unc_{task}.p')
        write_list(passages_unc, f'chatgpt_passages_unc_{task}.p')
        write_list(chatgpt_true_scores_unc, f'chatgpt_true_scores_unc_{task}.p')
        write_list(chatgpt_answers_unc, f'chatgpt_answers_unc_{task}.p')
        write_list(occurances_unc, f'chatgpt_occurances_unc_{task}.p')
        write_list(semantic_ids_unc, f'chatgpt_semantic_ids_unc_{task}.p')
        write_list(probs_unc, f'chatgpt_probs_unc_{task}.p')

    def read_results(task):
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

    utils.setup_openai()

    def read_chatgpt_results(task, dir='../collected_data'):
        retrieved_scores = read_list(os.path.join(dir, f'chatgpt_retrieved_scores_{task}.p'))
        retrieved_true_scores = read_list(os.path.join(dir, f'chatgpt_retrieved_true_scores_{task}.p'))
        queries = read_list(os.path.join(dir, f'chatgpt_queries_{task}.p'))
        answers = read_list(os.path.join(dir, f'chatgpt_answers_{task}.p'))
        chatgpt_true_scores = read_list(os.path.join(dir, f'chatgpt_true_scores_{task}.p'))
        chatgpt_answers = read_list(os.path.join(dir, f'chatgpt_answers_{task}.p'))
        chatgpt_passages = read_list(os.path.join(dir, f'chatgpt_passages_{task}.p'))
        chatgpt_semantics = read_list(os.path.join(dir, f'chatgpt_semantics_{task}.p'))
        chatgpt_occurances = read_list(os.path.join(dir, f'chatgpt_occurances_{task}.p'))
        chatgpt_semantic_ids = read_list(os.path.join(dir, f'chatgpt_semantic_ids_{task}.p'))
        chatgpt_probs = read_list(os.path.join(dir, f'chatgpt_probs_{task}.p'))
        retrieved_scores_unc = read_list(os.path.join(dir, f'chatgpt_retrieved_scores_unc_{task}.p'))
        retrieved_true_scores_unc = read_list(os.path.join(dir, f'chatgpt_retrieved_true_scores_unc_{task}.p'))
        queries_unc = read_list(os.path.join(dir, f'chatgpt_queries_unc_{task}.p'))
        answers_unc = read_list(os.path.join(dir, f'chatgpt_answers_unc_{task}.p'))
        passages_unc = read_list(os.path.join(dir, f'chatgpt_passages_unc_{task}.p'))
        chatgpt_true_scores_unc = read_list(os.path.join(dir, f'chatgpt_true_scores_unc_{task}.p'))
        chatgpt_answers_unc = read_list(os.path.join(dir, f'chatgpt_answers_unc_{task}.p'))
        chatgpt_occurances_unc = read_list(os.path.join(dir, f'chatgpt_occurances_unc_{task}.p'))
        chatgpt_semantic_ids_unc = read_list(os.path.join(dir, f'chatgpt_semantic_ids_unc_{task}.p'))
        chatgpt_probs_unc = read_list(os.path.join(dir, f'chatgpt_probs_unc_{task}.p'))
        return retrieved_scores, retrieved_true_scores, queries, answers, \
            chatgpt_true_scores, chatgpt_answers, chatgpt_passages, \
            chatgpt_semantics, chatgpt_occurances, chatgpt_semantic_ids, \
            chatgpt_probs, retrieved_scores_unc, retrieved_true_scores_unc, \
            queries_unc, answers_unc, passages_unc, chatgpt_true_scores_unc, \
            chatgpt_answers_unc, chatgpt_occurances_unc, \
            chatgpt_semantic_ids_unc, chatgpt_probs_unc

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

    retrieved_scores, retrieved_true_scores, queries, answers, chatgpt_true_scores, chatgpt_answers, chatgpt_passages, chatgpt_semantics, chatgpt_occurances, chatgpt_semantic_ids, chatgpt_probs, retrieved_scores_unc, retrieved_true_scores_unc, queries_unc, answers_unc, passages_unc, chatgpt_true_scores_unc, chatgpt_answers_unc, chatgpt_occurances_unc, chatgpt_semantic_ids_unc, chatgpt_probs_unc = \
                read_chatgpt_results('nq')

    dataset_dpr = tasks.RQA_dpr(task)
    elements = dataset_dpr.elements
    all_queries = [element['question'] for element in elements]
    answers = []
    for query in queries:
        idx = all_queries.index(query)
        answers.append(elements[idx]['answers'])

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

    print("Individual components")
    if task == 'bio':
        # cal_first_retrieved_true_scores = [np.max(scores) for scores in cal_first_retrieved_true_scores]
        tmp = []
        for scores in cal_first_retrieved_true_scores:
            tmp.extend(scores)
        cal_first_retrieved_true_scores = tmp
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha/2)
    cal_first_scores = []
    for scores in cal_first_chatgpt_true_scores:
        cal_first_scores.append(np.max(scores))
    chatgpt_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha/2)

    if task == 'bio':
        # cal_second_retrieved_true_scores = [np.max(scores) for scores in cal_second_retrieved_true_scores]
        tmp = []
        for scores in cal_second_retrieved_true_scores:
            tmp.extend(scores)
        cal_second_retrieved_true_scores = tmp
    retrieved_coverage = np.mean(np.array(cal_second_retrieved_true_scores) >= retrieved_thr)
    cal_second_scores = []
    for scores in cal_second_chatgpt_true_scores:
        cal_second_scores.append(np.max(scores))
    qa_coverage = np.mean(np.array(cal_second_scores) >= chatgpt_qa_thr)
    print('retrieval coverage', retrieved_coverage)
    print('qa coverage', qa_coverage)

    if task == 'bio':
        # test_retrieved_true_scores = [np.max(scores) for scores in test_retrieved_true_scores]
        tmp = []
        for scores in test_retrieved_true_scores:
            tmp.extend(scores)
        test_retrieved_true_scores = tmp
    retrieved_coverage = np.mean(np.array(test_retrieved_true_scores) >= retrieved_thr)
    test_scores = []
    for scores in test_chatgpt_true_scores:
        test_scores.append(np.max(scores))
    qa_coverage = np.mean(np.array(test_scores) >= chatgpt_qa_thr)
    print('test retrieval coverage', retrieved_coverage)
    print('test qa coverage', qa_coverage)

    coverages = coverage(
        test_retrieved_true_scores, 
        test_chatgpt_true_scores,
        retrieved_thr,
        chatgpt_qa_thr
        )
    print('End-to-end coverage', np.mean(coverages))

    kernel = 10
    length = len(test_retrieved_scores)
    lens = np.linspace(0, length, kernel+1)
    test_retrieved_scores_list = [test_retrieved_scores[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_chatgpt_semantic_ids_list = [test_chatgpt_semantic_ids[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_chatgpt_probs_list = [test_chatgpt_probs[int(lens[i]):int(lens[i+1])] for i in range(kernel)]
    test_answers_list = [test_answers[int(lens[i]):int(lens[i+1])] for i in range(kernel)]

    def ask_chatgpt(prompts,
                    model="gpt-3.5-turbo-0613",
                    temperature=1.0,
                    max_token=20,
                    n_answers=5):

        messages = [{"role": "user",
                    "content": prompt} for prompt in prompts]
        response = utils.chatcompletions_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,  # this is the degree of randomness of the model's output
            n=n_answers  # how many different answers to return
        )

        choices = [choice.message['content'].strip()
                for choice
                in response.choices]
        input_token_counts = response.usage['prompt_tokens']
        output_token_counts = response.usage['completion_tokens']

        # return response.choices[0].message["content"]
        return choices, input_token_counts, output_token_counts

    import time
    total_times = []
    single_times = []
    multiple_times = []
    cluster_times = []
    for idx, (element, score, retrieved) in enumerate(zip(elements, retrieved_scores, retrieved_examples)):
        if idx >= 20:
            break
        semantics = []  
        time_count = [] 
        total_start = time.time()
        prompts = []
        for s, context in zip(score, retrieved):
            # start = time.time()
            if s < retrieved_thr:
                continue
            prompt = utils.get_prompt_template(query, context, task='Natural Questions')
            prompts.append(prompt)

        single_start = time.time()
        sequences = ask_chatgpt(prompts, n_answers=1)[0]
        single_end = time.time()
        single_times.append(single_end - single_start)
        
        multiple_start = time.time()
        all_sequences = ask_chatgpt(prompts, n_answers=30)[0]
        multiple_end = time.time()
        multiple_times.append(multiple_end - multiple_start)

        cluster_start = time.time()
        for sequences in all_sequences:
            semantic_set_ids, semantic_probs, item_occurance = \
                    utils.clustering(sequences, prompt, scorer=scorer)
            for predicted_answer in semantic_set_ids.keys():
                concept_id = semantic_set_ids[predicted_answer]
                prob = semantic_probs[concept_id]
                if prob >= chatgpt_qa_thr:
                    semantics.append(predicted_answer)

        semantic_set_ids, semantic_probs, item_occurance = \
                    utils.clustering(semantics, "", scorer=scorer)
        total_end = time.time()
        cluster_end = time.time()
        total_time = total_end - total_start
        total_times.append(total_time)
        cluster_time = cluster_end - cluster_start
        cluster_times.append(cluster_time)

    print('total time', (np.sum(total_times) - np.sum(single_times)) / len(total_times))
    print('single time', np.mean(single_times))
    print('multiple time', np.mean(multiple_times))
    print('clustering time', np.mean(cluster_times))