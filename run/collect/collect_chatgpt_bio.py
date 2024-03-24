import os
import sys
import inspect
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"
from misc import utils
from misc import tasks
from rouge_score import rouge_scorer
import random
import numpy as np
import torch
import argparse
breakpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune llama-2 model on specific dataset")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2)
    parser.add_argument(
        "--semantic",
        action='store_true')
    torch.set_grad_enabled(False)
    from haystack.nodes import DensePassageRetriever
    from haystack.document_stores.faiss import FAISSDocumentStore
    document_store_faiss = FAISSDocumentStore(faiss_index_path="bio_faiss_index.faiss")
    reloaded_retriever = DensePassageRetriever.load(load_dir='./dpr_ft/bio_dpr_new', document_store=document_store_faiss)
    task='bio'
    questions, contexts, answers = tasks.bio_dpr(task=task).load_dataset()

    def hit_rate(ctx_ids_list, retrieved_ids_list):
        rate = []
        for ctx_ids, retrieved_ids in zip(ctx_ids_list, retrieved_ids_list):
            # check if any id in true ids in retrieved ids
            if any([ctx_id in retrieved_ids for ctx_id in ctx_ids]):
                rate.append(True)
            else:
                rate.append(False)
        return np.mean(rate)

    ctx_ids_list = []
    retrieved_ids_list = []
    ctx_ids_list = [[tmp['id'] for tmp in c] for c in contexts]
    retrieved_docs = reloaded_retriever.retrieve_batch(questions, top_k=20)
    retrieved_ids_list = [[d.id for d in docs] for docs in retrieved_docs]
    rate = hit_rate(ctx_ids_list, retrieved_ids_list)
    print("hit rate: ", rate)

    retrieved_scores = [[d.score for d in docs] for docs in retrieved_docs]

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
        
    indices = np.arange(len(questions))
    random.shuffle(indices)
    cal_indices = indices[:int(len(indices) * 0.5)]
    test_indices = indices[int(len(indices) * 0.5):]

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
    chat = True
    semantic = False
    queries = []
    # answers = []
    passages = []
    retrieved_scores = []
    retrieved_true_scores = []
    chatgpt_true_scores = []
    chatgpt_texts = []
    chatgpt_answers = []
    chatgpt_semantics = []
    semantic_probs = []
    feasibilities = []
    occurances = []
    semantic_ids = []
    probs = []
    input_token_counts = []
    output_token_counts = []

    retrieved_scores_unc = []
    retrieved_true_scores_unc = []
    queries_unc = []
    answers_unc = []
    passages_unc = []
    chatgpt_true_scores_unc = []
    chatgpt_answers_unc = []
    occurances_unc = []
    semantic_ids_unc = []
    probs_unc = []
            
    # for idx, (element, score, retrieved) in enumerate(zip(elements, scores, retrieved_examples)):
    for idx, (question, context, answer, docs) in enumerate(zip(questions, contexts, answers, retrieved_docs)):
        # if len(queries) > 1000:
        #     break
        print(idx)
        print(f'{idx}', file=open(f'chatgpt_{task}.txt', 'a'))
        feasible = False
        if idx % 10 == 0:
            print(idx)
            save_results(task)
        passage_id = [c['id'] for c in context]
        # passage_title = [c['title'] for c in context]
        passage_text = [c['text'] for c in context]

        retrieved_ids = [d.id for d in docs]
        retrieved_texts = [d.content for d in docs]
        # retrieved_title = [d.meta['title'] for d in docs]
        retrieved_score = [d.score for d in docs]
        true_score = [s for s, retrieved_id in zip(retrieved_score, retrieved_ids) if retrieved_id in passage_id]
        if len(true_score) == 0:
            print('no correct context')
            continue
        
        prompt = utils.get_prompt_template(question, passage_text[0], task='Natural Questions')
        if chat:
            sequences, input_token_count, output_token_count = \
                utils.ask_chatgpt(prompt, n_answers=30, model="gpt-3.5-turbo-0613")
        else:
            sequences, probs = utils.ask_chatgpt(prompt)

        semantic_set_ids, semantic_probs, item_occurance = \
            utils.clustering(sequences, prompt, scorer=scorer)
        true_scores, matched_answer, semantics = utils.processing_answers(
            semantic_set_ids, semantic_probs, 
            item_occurance, answer, scorer,
            threshold=0.3
        )

        if len(true_scores) == 0:
            retrieved_scores_unc.append(retrieved_score)
            retrieved_true_scores_unc.append(true_score)
            queries_unc.append(question)
            answers_unc.append(answer)
            passages_unc.append(passage_text)
            chatgpt_true_scores_unc.append(true_scores)
            chatgpt_answers_unc.append(sequences)
            occurances_unc.append(item_occurance)
            semantic_ids_unc.append(semantic_set_ids)
            probs_unc.append(semantic_probs)
            input_token_counts.append(input_token_count)
            output_token_counts.append(output_token_count)
            print('No true response')
        else:
            print('valid', idx)
            feasible = True
            retrieved_scores.append(retrieved_score)
            retrieved_true_scores.append(true_score)
            queries.append(question)
            answers.append(answer)
            passages.append(passage_text)
            chatgpt_true_scores.append(true_scores)

            probs_tmp = []
            answers_tmp = []
            semantic_id_tmp = []
            occurance_tmp = []
            semantic_tmp = []
            for context, s in zip(retrieved_texts, retrieved_score):
                prompt = utils.get_prompt_template(question, context, task='Natural Questions')
                if chat:
                    sequences, input_token_count_tmp, output_token_count_tmp = \
                        utils.ask_chatgpt(prompt, n_answers=30, model="gpt-3.5-turbo-0613")
                else:
                    sequences, probs = utils.ask_chatgpt(prompt, n_answers=30)
                input_token_count += input_token_count_tmp
                output_token_count += output_token_count_tmp

                if semantic:
                    semantic_set_ids, semantic_probs, item_occurance = \
                        utils.compute_semantic_clusterring(
                            semantic_model, 
                            semantic_tokenizer,
                            prompt,
                            sequences,
                        )
                else:
                    semantic_set_ids, semantic_probs, item_occurance = \
                        utils.clustering(sequences, prompt, scorer=scorer)

                probs_tmp.append(semantic_probs)
                answers_tmp.append(sequences)
                occurance_tmp.append(item_occurance)
                semantic_id_tmp.append(semantic_set_ids)

            chatgpt_answers.append(answers_tmp)
            feasibilities.append(feasible)
            occurances.append(occurance_tmp)
            semantic_ids.append(semantic_id_tmp)
            probs.append(probs_tmp)
            input_token_counts.append(input_token_count)
            output_token_counts.append(output_token_count)
    print('Finished!', file=open(f'chatgpt_{task}.txt', 'a'))