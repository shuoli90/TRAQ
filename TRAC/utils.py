import regex as re
import string
import unicodedata
import os
import sys
from datasets import load_dataset
import openai
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import math
import torch
import backoff
from transformers import AutoTokenizer, RagRetriever
from transformers import RagModel
import json
import random
# from gensim.summarization.bm25 import BM25

def separate_sentences(paragraph):
    # Split the paragraph into sentences using commas and periods, keeping the punctuation
    sentences = re.split(r'(?<=[.,])\s', paragraph)

    # Combine the sentences into a single string, with each sentence on a new line
    separated_sentences = '\n'.join(sentences)

    return separated_sentences


# Set up chatgpt api
def setup_openai():
    dotenv_path = Path('.env')
    load_dotenv(dotenv_path=dotenv_path)
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.organization = 'org-ZZanmSElxExwouVgEWnZL8zv'


def add_home_directory_to_path():
    home_dir = os.path.expanduser("~")
    if home_dir not in sys.path:
        sys.path.append(home_dir)


def answer_logprob(choices):
    logprobs = choices["logprobs"]["token_logprobs"]
    logprob = np.mean(logprobs)
    return logprob


def filter_unique_items(answers, logprobs):
    if len(answers) != len(logprobs):
        raise ValueError("Both lists must have the same length")

    item_occurrences = {}
    unique_items = []
    unique_labels = []
    repeat_times = {}

    for item, label in zip(answers, logprobs):
        if item in item_occurrences:
            item_occurrences[item] += 1
        else:
            item_occurrences[item] = 1
            unique_items.append(item)
            unique_labels.append(label)

    for item, count in item_occurrences.items():
        repeat_times[item] = count

    return unique_items, unique_labels, repeat_times


def get_predictive_entropy_over_concepts(semantic_logprob,
                                         semantic_set_ids, repeat_times):
    """
        Adopt from semantic uncertainty paper
    """
    """Compute the semantic entropy"""

    # cluster logprobs by semantic meaning
    sets = {}
    for key in semantic_set_ids.keys():
        set = semantic_set_ids[key]
        if set not in sets:
            # sets[set] = [[semantic_logprob[key], repeat_times[key]]]
            sets[set] = [math.exp(semantic_logprob[key]) * repeat_times[key]]
        else:
            sets[set].append(
                math.exp(semantic_logprob[key]) * repeat_times[key])
    # compute p(c \mid x) for each concept c
    concept_probs = []
    totoal_prob = 0
    for set in sets.keys():
        # get logprobs for each concept c
        probs = torch.tensor(sets[set])
        # compute \sum_s p(s \mid x)
        concept_prob = torch.sum(probs)
        totoal_prob += concept_prob
        concept_probs.append(concept_prob)
    return [concept_prob/totoal_prob for concept_prob in concept_probs]


def clusterring(model, tokenizer, question, unique_generated_texts):
    # filter out non-unique answers
    # unique_generated_texts, logprobs, repeat_times = \
    # filter_unique_items(answers, logprobs)

    # # unique_generated_texts = list(set(answers))
    semantic_set_ids = {}
    # semantic_logprob = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    # print('Number of unique answers:', len(unique_generated_texts))

    with torch.no_grad():
        # if len(unique_generated_texts) > 1:

        # Evalauate semantic similarity
        for i, reference_answer in enumerate(unique_generated_texts):
            # semantic_logprob[unique_generated_texts[i]] = logprobs[i]
            for j in range(i + 1, len(unique_generated_texts)):

                qa_1 = question + ' ' + unique_generated_texts[i]
                qa_2 = question + ' ' + unique_generated_texts[j]

                input = qa_1 + ' [SEP] ' + qa_2
                encoded_input = tokenizer.encode(input, padding=True)
                prediction = model(torch.tensor(torch.tensor([encoded_input]),
                                                device='cuda'))['logits']
                predicted_label = torch.argmax(prediction, dim=1)

                reverse_input = qa_2 + ' [SEP] ' + qa_1
                encoded_reverse_input = tokenizer.encode(reverse_input,
                                                         padding=True)
                reverse_prediction = model(torch.tensor(
                    torch.tensor([encoded_reverse_input]),
                    device='cuda'))['logits']
                reverse_predicted_label = torch.argmax(reverse_prediction,
                                                       dim=1)

                # print(qa_1, qa_2, predicted_label, reverse_predicted_label)
                if 0 in predicted_label or 0 in reverse_predicted_label:
                    pass
                elif 2 in predicted_label and 2 in reverse_predicted_label:
                    semantic_set_ids[unique_generated_texts[j]] = \
                        semantic_set_ids[unique_generated_texts[i]]
    return semantic_set_ids


def compute_semantic_similarity(model, tokenizer, question, answers, logprobs):
    # filter out non-unique answers
    unique_generated_texts, logprobs, repeat_times = \
        filter_unique_items(answers, logprobs)

    # unique_generated_texts = list(set(answers))
    semantic_set_ids = {}
    semantic_logprob = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    # print('Number of unique answers:', len(unique_generated_texts))

    with torch.no_grad():
        if len(unique_generated_texts) > 1:

            # Evalauate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                semantic_logprob[unique_generated_texts[i]] = logprobs[i]
                for j in range(i + 1, len(unique_generated_texts)):

                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    encoded_input = tokenizer.encode(input, padding=True)
                    prediction = model(torch.tensor(
                        torch.tensor([encoded_input]),
                        device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = tokenizer.encode(reverse_input,
                                                             padding=True)
                    reverse_prediction = model(
                        torch.tensor(torch.tensor([encoded_reverse_input]),
                                     device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction,
                                                           dim=1)

                    if predicted_label == 2 and reverse_predicted_label == 2:
                        semantic_set_ids[unique_generated_texts[j]] = \
                            semantic_set_ids[unique_generated_texts[i]]

    concept_probs = get_predictive_entropy_over_concepts(
        semantic_logprob, semantic_set_ids, repeat_times)
    return concept_probs


def unique_items(answers):

    item_occurrences = {}
    unique_items = []

    for item in answers:
        if item in item_occurrences:
            item_occurrences[item] += 1
        else:
            item_occurrences[item] = 1
            unique_items.append(item)

    return unique_items, item_occurrences


def get_semantic_prob(semantic_set_ids, repeat_times):
    # first compute all occurance
    occurance = 0
    for key in repeat_times.keys():
        occurance += repeat_times[key]

    semantic_probs = {}
    # compute each semantic prob
    for key in semantic_set_ids.keys():
        semantic_set_id = semantic_set_ids[key]
        if semantic_set_id not in semantic_probs:
            semantic_probs[semantic_set_id] = repeat_times[key]
        else:
            semantic_probs[semantic_set_id] += repeat_times[key]

    # normalize
    for key in semantic_probs.keys():
        semantic_probs[key] /= occurance
    return semantic_probs


def compute_semantic_clusterring(model, tokenizer,
                                 question, answers):
    # filter out non-unique answers
    unique_generated_texts, item_occurance = \
        unique_items(answers)

    # unique_generated_texts = list(set(answers))
    semantic_set_ids = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    # print('Number of unique answers:', len(unique_generated_texts))

    with torch.no_grad():
        if len(unique_generated_texts) > 1:

            # Evalauate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                for j in range(i + 1, len(unique_generated_texts)):

                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    encoded_input = tokenizer.encode(input, padding=True)
                    prediction = model(torch.tensor(
                        torch.tensor([encoded_input]),
                        device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = tokenizer.encode(reverse_input,
                                                             padding=True)
                    reverse_prediction = model(
                        torch.tensor(torch.tensor([encoded_reverse_input]),
                                     device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(
                        reverse_prediction,
                        dim=1)

                    if 2 in predicted_label and 2 in reverse_predicted_label:
                        semantic_set_ids[unique_generated_texts[j]] = \
                            semantic_set_ids[unique_generated_texts[i]]

    semantic_probs = get_semantic_prob(semantic_set_ids, item_occurance)
    return semantic_set_ids, semantic_probs, item_occurance


def compute_keyword_clusterring(answers,
                                scorer):
    # filter out non-unique answers
    unique_generated_texts, item_occurance = \
        unique_items(answers)

    # unique_generated_texts = list(set(answers))
    semantic_set_ids = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    # print('Number of unique answers:', len(unique_generated_texts))

    with torch.no_grad():
        if len(unique_generated_texts) > 1:

            # Evalauate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                for j in range(i + 1, len(unique_generated_texts)):

                    scores = scorer.score(unique_generated_texts[i], 
                                          unique_generated_texts[j])
                    scores = scores['rouge1'][2]
                    if scores > 0.7:
                        semantic_set_ids[unique_generated_texts[j]] = \
                            semantic_set_ids[unique_generated_texts[i]]

    semantic_probs = get_semantic_prob(semantic_set_ids, item_occurance)
    return semantic_set_ids, semantic_probs, item_occurance


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def chatcompletions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def ask_completion(prompt, 
                model='davinci', temperature=1.0):

    # Combine the context and question into a single prompt
    # prompt = f"Evidence: {context}\nQuestion: {question}\nAnswer:"
    # prompt = few_shot + '\n\n' + prompt

    # question += '?'
    # prompt = f""" Answer the following question based on the given context; \
    #         Answer "I don't know" if you don't know the answer; \
    #         Answer the question using only one keyword. \
    #         Question: {question}
    #         Context: {context}
    #         Answer: """

    if model == 'davinci':
        engine = 'text-davinci-003'
    elif model == 'curie':
        engine = 'text-curie-001'
        # engine = 'curie'

    response = completions_with_backoff(
        engine=engine,
        prompt=prompt,
        max_tokens=20,
        n=5,
        stop=None,
        temperature=temperature,
        logprobs=5
    )
    choices = [choice.text.strip() for choice in response.choices]
    logprobs = [answer_logprob(choice) for choice in response.choices]
    probs = [math.exp(logprob) for logprob in logprobs]

    return choices, probs


def ask_chatgpt_clusterring(question, context,
                            model='davinci',
                            temperature=1.0):

    # Combine the context and question into a single prompt
    # prompt = f"Evidence: {context}\nQuestion: {question}\nAnswer:"
    # prompt = few_shot + '\n\n' + prompt

    prompt = f"""
            Answer the following question based on the given context; \
            Answer the question with only one key word.

            Question: '''{question}'''

            Context: '''{context}'''
            """

    if model == 'davinci':
        engine = 'text-davinci-003'
    elif model == 'curie':
        engine = 'text-curie-001'
        # engine = 'curie'

    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=20,
        n=5,
        stop=None,
        temperature=temperature,
        logprobs=5
    )
    choices = [choice.text.strip() for choice in response.choices]
    logprobs = [answer_logprob(choice) for choice in response.choices]
    probs = [math.exp(logprob) for logprob in logprobs]

    return choices, probs


def get_prompt_template(question, context, task='nq', ):

    # Combine the context and question into a single prompt
    # prompt = f"Evidence: {context}\nQuestion: {question}\nAnswer:"
    # prompt = few_shot + '\n\n' + prompt

    # if task in ['Natural Questions', 'TriviaQA']:
    prompt = f"""Answer the following question based on the given context; Answer the question shortly.
            Question: '''{question}'''
            Context: '''{context}'''
            Answer:
            """
    # elif task == 'fever':
    #     prompt = f"""Answer the following question based on the given context; Answer "Support" if the context supports the claim or "Refutes".
    #             Claim: '''{question}'''
    #             Context: '''{context}'''
    #             Answer:
    #             """
    return prompt

def get_prompt_template_wo(question, task='nq', ):


    if task in ['Natural Questions', 'TriviaQA']:
        prompt = f"""Answer the following question; Answer the question shortly.
                Question: '''{question}'''
                Answer:
                """
    elif task == 'fever':
        prompt = f"""Answer the following question; Answer "Support" if the context supports the claim or "Refutes".
                Claim: '''{question}'''
                Answer:
                """
    return prompt


def ask_chatgpt(prompt,
                   model="gpt-3.5-turbo",
                   temperature=1.0,
                   max_token=20,
                   n_answers=5):

    messages = [{"role": "user",
                 "content": prompt}]
    response = chatcompletions_with_backoff(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
        n=n_answers  # how many different answers to return
    )

    choices = [choice.message['content'].strip()
               for choice
               in response.choices]

    # return response.choices[0].message["content"]
    return choices

def get_completion(question, context,
                   model="gpt-3.5-turbo",
                   temperature=1.0,
                   max_token=20,
                   n_answers=5):
    question += '?'
    prompt = f""" Answer the following question based on the given context; \
            Answer "I don't know" if you don't know the answer; \
            Answer the question using only one keyword. \
            Question: {question}
            Context: {context}
            Answer: """
    messages = [{"role": "user",
                 "content": prompt}]
    response = chatcompletions_with_backoff(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
        n=n_answers  # how many different answers to return
    )
    # choices = []
    # for choice in response.choices:
    #     choice_text = choice.message['content'].strip()
    #     if choice_text not in choices:
    #         choices.append(choice_text)

    choices = [choice.message['content'].strip()
               for choice
               in response.choices]

    # return response.choices[0].message["content"]
    return choices

def processing_answers(semantic_clusterring, semantic_probs,
                       item_occurance, reference_answers, 
                       scorer, metric='rouge1', 
                       threshold=0.5, thr_qa=0.0):
        
    true_scores = []
    matched_answers = []
    semantics = []
    for predicted_answer in semantic_clusterring.keys():
        concept_id = semantic_clusterring[predicted_answer]
        repeat = item_occurance[predicted_answer]
        prob = semantic_probs[concept_id]
        if prob >= thr_qa:
            semantics.extend([predicted_answer] * repeat)
            for reference_answer in reference_answers:
                scores = scorer.score(reference_answer, predicted_answer)
                if metric == "rouge1":
                    scores = scores['rouge1'][2]
                else:
                    scores = scores['rougeL'][2]
                if scores > threshold:
                    true_scores.extend([prob] * repeat)
                    matched_answers.append(predicted_answer)
                    break
            # else:
            #     true_scores.extend([0.0] * repeat)
    return true_scores, matched_answers, semantics


def get_completion_clusterring(
        question, context,
        semantic_model, semantic_tokenizer,
        scorer,
        model="gpt-3.5-turbo",
        temperature=1.0,
        max_token=20,
        n_answers=5, semantic=False):
    question += '?'
    prompt = f""" Answer the following question based on the given context; \
            Answer "I don't know" if you don't know the answer; \
            Answer the question using only one keyword. \
            Question: {question}
            Context: {context} 
            Answer: """
    messages = [{"role": "user", "content": prompt}]
    response = chatcompletions_with_backoff(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
        n=n_answers  # how many different answers to return
    )

    choices = [choice.message['content'].strip()
               for choice
               in response.choices]
    if semantic:
        semantic_clusterring, semantic_probs, item_occurance = \
            compute_semantic_clusterring(
                model=semantic_model,
                tokenizer=semantic_tokenizer,
                question=prompt, answers=choices,
                scorer=scorer)
    else:
        semantic_clusterring, semantic_probs, item_occurance = \
            compute_keyword_clusterring(
                answers=choices,
                scorer=scorer)

    # return response.choices[0].message["content"]
    return semantic_clusterring, semantic_probs, item_occurance



def load_hotpotqa_dataset(split, config='fullwiki'):
    dataset = load_dataset("hotpot_qa", config,
                           split=split, cache_dir="/data3/shuoli/data/")
    return dataset


def load_triviaQA_dataset(split="train"):
    """
    Load the TriviaQA dataset from Hugging Face Datasets.

    Args:
        split (str, optional): The dataset split to load.
        Options are 'train', 'validation', and 'test'.
        Defaults to 'train'.

    Returns:
        Dataset: The TriviaQA dataset split.
    """
    dataset = load_dataset("trivia_qa", "rc",
                           split=split, cache_dir="/data3/shuoli/data/")
    return dataset


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def has_answer(answers, text, match_type='string') -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string,
    token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    # text = _normalize(text)
    text = _normalize_answer(text)

    if match_type == "string":

        for single_answer in answers:
            single_answer = _normalize_answer(single_answer)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i:i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            # single_answer = _normalize(single_answer)
            single_answer = _normalize_answer(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE +
                             re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


def context_match(context, gold_contexts, scorer):
    for gold_context in gold_contexts:
        scores = scorer.score(context, gold_context)
        if scores['rouge1'].fmeasure > 0.5:
            return True
    return False

def load_retriever():
    tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq",
                                                index_name="compressed")
    model = RagModel.from_pretrained("facebook/rag-sequence-nq",
                                        index_name='compressed',
                                        retriever=retriever).cuda()
    return model, tokenizer, retriever

def save_results(args, cosines, innerproducts, contexts, answers_list, indices):
    if args.retrieve:
        filepath = f'cosine_{args.n_answers}_{args.temp}_{args.start}_{args.exp}.json'
        with open(os.path.join('collected', filepath), "w") as f:
            json.dump(cosines, f)
        filepath = f'innerproduct_{args.n_answers}_{args.temp}_{args.start}_{args.exp}.json'
        with open(os.path.join('collected', filepath), "w") as f:
            json.dump(innerproducts, f)
        filepath = f'contexts_{args.n_answers}_{args.temp}_{args.start}_{args.exp}.json'
        with open(os.path.join('collected', filepath), "w") as f:
            json.dump(contexts, f)

        filepath = f'answers_{args.n_answers}_{args.temp}_retrieve_{args.start}_{args.exp}.jsonl'
        with open(os.path.join('collected', filepath), "w") as f:
            for item in answers_list:
                f.write(json.dumps(item) + "\n")

        filepath = f'indices_{args.n_answers}_{args.temp}_retrieve_{args.start}_{args.exp}.json'
        with open(os.path.join('collected', filepath), "w") as f:
            json.dump(indices, f)
    else:
        filepath = f'answers_{args.n_answers}_{args.temp}_{args.start}.jsonl'
        with open(os.path.join(
                    'collected', filepath), "w") as f:
            for item in answers_list:
                f.write(json.dumps(item) + "\n")

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def convert_list_to_dict(retrieved):
    convert = {}
    for item in retrieved:
        convert[item['wikipedia_id']] = float(item['score'])
    return convert

def split(scores, indices):
    return [score for idx, score in enumerate(scores) if idx in indices]

def compute_threshold(scores, alpha, portion=0.5, shuffle=True):
    # scores_tmp = []
    # for score in scores:
    #     scores_tmp.extend(score)
    # scores = scores_tmp
    
    # if shuffle:
    #     random.shuffle(scores)
    # cal = scores[:int(len(scores)*portion)]
    # test = scores[int(len(scores)*(1-portion)):]
    
    thr_most_relevant = np.quantile(scores, alpha, interpolation='higher')
    # print(f"Most relevant threshold: {thr_most_relevant}")
    # test = np.array(test)
    # coverage_most_relevant = np.mean(test >= thr_most_relevant)
    # print(f"Most relevant coverage: {coverage_most_relevant}")

    return thr_most_relevant

def dataset_info(element, dataset='nq'):
    query = element['question']
    answer = [ans for ans in element['answers']]
    if dataset == 'nq':
        passage_id = [ctx['passage_id'] for ctx in element['positive_ctxs']]
    else:
        passage_id = [ctx['psg_id'] for ctx in element['positive_ctxs']]
    passage_title = [ctx['title'] for ctx in element['positive_ctxs']]
    passage_text = [ctx['text'] for ctx in element['positive_ctxs']]
    return query, answer, passage_id, passage_title, passage_text


def retrieved_info(score, retrieved, passage_id):
    retrieved_ids = retrieved['id']
    retrieved_texts = retrieved['text']
    retrieved_title = retrieved['title']
    true_score = [s for s, retrieved_id in zip(score, retrieved_ids) if retrieved_id == passage_id]
    return retrieved_ids, retrieved_texts, retrieved_title, true_score

def ask(query, passage_text, chat=False, task="Natural Questions"):
    prompt = get_prompt_template(query, passage_text, task=task)
    if chat:
        sequences = ask_chatgpt(prompt, n_answers=30)
    else:
        sequences, probs = ask_chatgpt(prompt, n_answers=30)
    return sequences, prompt

def clustering(sequences, prompt=None, semantic_model=None, semantic_tokenizer=None, scorer=None, semantic=False):
    if semantic:
            semantic_set_ids, semantic_probs, item_occurance = \
                compute_semantic_clusterring(
                    semantic_model, 
                    semantic_tokenizer,
                    prompt,
                    sequences,
                )
    else:
        semantic_set_ids, semantic_probs, item_occurance = \
            compute_keyword_clusterring(
                sequences,
                scorer
            )
    return semantic_set_ids, semantic_probs, item_occurance

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