import os
os.environ["CUDA_VISIBLE_DEVICES"]="3,5"
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
from misc import opensource
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect data")
    parser.add_argument(
        "--task",
        type=str,
        default="nq",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--semantic",
        action='store_true',
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", device_map='cuda')
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base", device_map='cuda')
    wiki = load_dataset(path='wiki_dpr', name='psgs_w100.multiset.compressed', split='train')

    task='nq'
    dataset_dpr = tasks.RQA_dpr(task=task)

    semantic = False
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
    relevance, retrieved_examples = wiki.get_nearest_examples_batch('embeddings', question_embedding, k=20)

    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        HfArgumentParser,
        TrainingArguments,
        pipeline,
        logging,
    )
    from peft import LoraConfig, PeftModel
    # from trl import SFTTrainer
    import tasks
    import utils
    # from trl import DataCollatorForCompletionOnlyLM
    from datasets import Dataset
    import json
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    # Fine-tuned model name
    new_model = f"../finetuned_models/llama-2-7b-traq-{task}"
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_new = PeftModel.from_pretrained(base_model, new_model)
    # model_new = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pipe_new = pipeline(
        task="text-generation", 
        model=model_new, 
        tokenizer=tokenizer, 
        max_length=300,
        return_full_text=False)

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

    def save_results(task):
        # save retrieved_scores to a pickle file
        write_list(retrieved_scores, f'retrieved_scores_{task}_new.p')
        # save retrieved_true_scores to a pickle file
        write_list(retrieved_true_scores, f'retrieved_true_scores_{task}_new.p')
        # save queries to a pickle file
        write_list(queries, f'queries_{task}_new.p')
        # save answers to a pickle file
        write_list(answers, f'answers_{task}_new.p')
        # save passages to a pickle file
        write_list(passages, f'passages_{task}_new.p')
        # save opensource_true_scores to a pickle file
        write_list(opensource_true_scores, f'opensource_true_scores_{task}_new.p')
        # save opensource_texts to a pickle file
    #     write_list(opensource_texts, f'opensource_texts_{task}.p')
        # save opensource_answers to a pickle file
        write_list(opensource_answers, f'opensource_answers_{task}_new.p')
        # save opensource_semantics to a picle file
        write_list(opensource_semantics, f'opensource_semantics_{task}_new.p')
        # save feasibilities to a pickle file
        write_list(feasibilities, f'feasibilities_{task}_new.p')
        # save occurances to a pickle file
        write_list(occurances, f'occurances_{task}_new.p')
        # save semantic_ids to a pickle file
        write_list(semantic_ids, f'semantic_ids_{task}_new.p')
        # save probs to a picle file
        write_list(probs, f'probs_{task}_new.p')

    def read_results(task, end=1000, dir='../collected_data'):
        retrieved_scores = read_list(os.path.join(dir, f'retrieved_scores_{task}.p'))[:end]
        retrieved_true_scores = read_list(os.path.join(dir, f'retrieved_true_scores_{task}.p'))[:end]
        queries = read_list(os.path.join(dir, f'queries_{task}.p'))[:end]
        answers = read_list(os.path.join(dir, f'answers_{task}.p'))[:end]
        opensource_true_scores = read_list(os.path.join(dir, f'opensource_true_scores_{task}.p'))[:end]
        opensource_answers = read_list(os.path.join(dir, f'opensource_answers_{task}.p'))[:end]
        opensource_semantics = read_list(os.path.join(dir, f'opensource_semantics_{task}.p'))[:end]
        opensource_occurances = read_list(os.path.join(dir, f'occurances_{task}.p'))[:end]
        opensource_semantic_ids = read_list(os.path.join(dir, f'semantic_ids_{task}.p'))[:end]
        opensource_probs = read_list(os.path.join(dir, f'probs_{task}.p'))[:end]
        
        return retrieved_scores, retrieved_true_scores, \
            queries, answers, \
            opensource_true_scores, opensource_answers, \
            opensource_occurances, opensource_semantic_ids, opensource_probs

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
    retrieved_thr = utils.compute_threshold(cal_first_retrieved_true_scores, alpha=alpha/2)
    cal_first_scores = []
    for scores in cal_first_opensource_true_scores:
        cal_first_scores.append(np.max(scores))
    opensource_qa_thr = utils.compute_threshold(cal_first_scores, alpha=alpha/2)

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

    coverages = coverage(
        test_retrieved_true_scores,
        test_opensource_true_scores,
        retrieved_thr,
        opensource_qa_thr
        )
    print('End-to-end coverage', np.mean(coverages))

    import time
    single_times = []
    multiple_times = []
    clustering_times = []
    total_times = []

    for idx, (element, score, retrieved) in enumerate(zip(elements, retrieved_scores, retrieved_examples)):
        query_start = time.time()
        if idx >= 50:
            break
        semantics = []  
        time_count = [] 
        query = element['question']
        cluster_time = 0.0

        prompts = []
        # for s, context in zip(score, retrieved['text']):
        #     if s < retrieved_thr:
        #         continue
        #     prompt = utils.get_prompt_template(query, context, task='Natural Questions')
        #     prompts.append(prompt)

        # single_start = time.time()
        # sequences = opensource.ask_openmodel(prompts, pipe_new, tokenizer, return_sequences=1)
        # single_end = time.time()
        # single_times.append(single_end-single_start)

        # multiple_start = time.time()
        # all_sequences = opensource.ask_openmodel(prompts, pipe_new, tokenizer, return_sequences=30)
        # multiple_end = time.time()
        # multiple_times.append(multiple_end-multiple_start)

        # for sequences in all_sequences:

        
        for s, context in zip(score, retrieved['text']):
            if s < retrieved_thr:
                continue
            prompt = utils.get_prompt_template(query, context, task='Natural Questions')

            single_start = time.time()
            sequences = opensource.ask_openmodel(prompt, pipe_new, tokenizer, return_sequences=1)
            single_end = time.time() 
            single_times.append(single_end-single_start)

            multiple_start = time.time()
            sequences = opensource.ask_openmodel(prompt, pipe_new, tokenizer, return_sequences=30)
            multiple_end = time.time()
            multiple_times.append(multiple_end-multiple_start)

            clustering_start = time.time()
            generated_texts = []
            for seq in sequences:
                tmp = seq['generated_text']
                idx = np.char.find(tmp, "~!~", start=0, end=None)
                tmp = tmp[:idx].strip()
                generated_texts.append(tmp)
        
            semantic_set_ids, semantic_probs, item_occurance = \
                    utils.clustering(generated_texts, prompt, scorer=scorer)
            for predicted_answer in semantic_set_ids.keys():
                concept_id = semantic_set_ids[predicted_answer]
                prob = semantic_probs[concept_id]
                if prob >= opensource_qa_thr:
                    semantics.append(predicted_answer)
            clustering_end = time.time()
            cluster_time += clustering_end-clustering_start

        start = time.time()
        semantic_set_ids, semantic_probs, item_occurance = \
                    utils.clustering(semantics, "", scorer=scorer)
        end = time.time()
        cluster_time += end-start
        clustering_times.append(cluster_time)
        query_end = time.time()
        total_times.append(query_end-query_start)

    print('total time', (np.sum(total_times) - np.sum(single_times)) / 50)
    print('single time', np.mean(single_times))
    print('multiple time', np.mean(multiple_times))
    print('clustering time', np.mean(clustering_times))

    import time
    single_times_new = []
    multiple_times_new = []
    clustering_times_new = []
    total_times_new = []

    for idx, (element, score, retrieved) in enumerate(zip(elements, retrieved_scores, retrieved_examples)):
        query_start = time.time()
        if idx >= 20:
            break
        semantics = []  
        time_count = [] 
        query = element['question']
        cluster_time = 0.0

        prompts = []
        for s, context in zip(score, retrieved['text']):
            if s < retrieved_thr:
                continue
            prompt = utils.get_prompt_template(query, context, task='Natural Questions')
            prompts.append(prompt)

        single_start = time.time()
        sequences = opensource.ask_openmodel(prompts, pipe_new, tokenizer, return_sequences=1)
        single_end = time.time()
        single_times_new.append(single_end-single_start)

        multiple_start = time.time()
        all_sequences = opensource.ask_openmodel(prompts, pipe_new, tokenizer, return_sequences=30)
        multiple_end = time.time()
        multiple_times_new.append(multiple_end-multiple_start)

        for sequences in all_sequences:

        
        # for s, context in zip(score, retrieved['text']):
        #     if s < retrieved_thr:
        #         continue
        #     prompt = utils.get_prompt_template(query, context, task='Natural Questions')

        #     single_start = time.time()
        #     sequences = opensource.ask_openmodel(prompt, pipe_new, tokenizer, return_sequences=1)
        #     single_end = time.time() 
        #     single_times.append(single_end-single_start)

        #     multiple_start = time.time()
        #     sequences = opensource.ask_openmodel(prompt, pipe_new, tokenizer, return_sequences=30)
        #     multiple_end = time.time()
        #     multiple_times.append(multiple_end-multiple_start)

            clustering_start = time.time()
            generated_texts = []
            for seq in sequences:
                tmp = seq['generated_text']
                idx = np.char.find(tmp, "~!~", start=0, end=None)
                tmp = tmp[:idx].strip()
                generated_texts.append(tmp)
        
            semantic_set_ids, semantic_probs, item_occurance = \
                    utils.clustering(generated_texts, prompt, scorer=scorer)
            for predicted_answer in semantic_set_ids.keys():
                concept_id = semantic_set_ids[predicted_answer]
                prob = semantic_probs[concept_id]
                if prob >= opensource_qa_thr:
                    semantics.append(predicted_answer)
            clustering_end = time.time()
            cluster_time += clustering_end-clustering_start

        start = time.time()
        semantic_set_ids, semantic_probs, item_occurance = \
                    utils.clustering(semantics, "", scorer=scorer)
        end = time.time()
        cluster_time += end-start
        clustering_times_new.append(cluster_time)
        query_end = time.time()
        total_times_new.append(query_end-query_start)

    print('total time', (np.sum(total_times_new) - np.sum(single_times_new)))
    print('single time', np.mean(single_times_new))
    print('multiple time', np.mean(multiple_times_new))
    print('clustering time', np.mean(clustering_times_new))