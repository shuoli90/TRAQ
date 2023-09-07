import os
import json
import argparse
from tqdm import tqdm
import rag_uncertainty
import utils
import time
import openai
from tasks import RQA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieve", action='store_true')
    parser.add_argument("--temp", type=float, default=1.5)
    parser.add_argument("--n_docs", type=int, default=20)
    parser.add_argument("--n_answers", type=int, default=40)
    parser.add_argument("--end", type=float, default=float('inf'))
    parser.add_argument("--semantic", action='store_true')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--exp", type=str, default='feasible')
    args = parser.parse_args()

    # setup chatgpt
    utils.setup_openai()

    if args.retrieve:
        retrieve_model, retrieve_tokenizer, retriever = \
            utils.load_retriever()

    # ask questions
    questions = []
    reference_answers = []
    context_texts = []
    answers_list = []
    innerproducts = []
    cosines = []
    contexts = []
    indices = []

    data = RQA()
    breakpoint()
    with open('./logs/biencoder-nq-dev.json', "r") as src_file:
        dpr_records = json.load(src_file)
        for idx, dpr_record in enumerate(tqdm(dpr_records)):
            if idx < args.start:
                continue
            question = dpr_record["question"].strip()
            questions.append(question)
            context_text = [context['text']
                            for context
                            in dpr_record['positive_ctxs']][0]
            context_id = [context['passage_id']
                          for context
                          in dpr_record['positive_ctxs']][0]
            context_texts.append(context_text)
            reference_answer = dpr_record["answers"][0]
            reference_answers.append(reference_answer)

            if args.retrieve:
                # 2 use retrieved context

                # retrieve context
                dict, innerproduct, cosine, all_docs = \
                    rag_uncertainty.retrieve(retrieve_model, retrieve_tokenizer,
                                             retriever, question,
                                             n_docs=args.n_docs)
                # docs_titles = all_docs['title']
                docs_texts = all_docs['text']
                # docs_id = docs_dict['doc_ids'][0]
                innerproducts.append(innerproduct[0].tolist())
                cosines.append(cosine[0].tolist())
                contexts.append(docs_texts)
                docs_ids = dict['doc_ids'][0].tolist()
                include = int(context_id) in docs_ids
                if not include:
                    continue
                else:
                    indices.append(idx)
                # for text in docs_texts:
                try:
                    for text in docs_texts:
                        # Make your OpenAI API request here
                        answers = utils.get_completion(
                            question, text,
                            temperature=args.temp,
                            n_answers=args.n_answers)
                        time.sleep(0.1)
                        answers_list.append(answers)
                except openai.error.APIError as e:
                    # Handle API error here, e.g. retry or log
                    print(f"OpenAI API returned an API Error: {e}")
                    utils.setup_openai()
                    continue
            else:
                answers = utils.get_completion(
                            question, context_text,
                            temperature=args.temp,
                            n_answers=args.n_answers)
                time.sleep(0.1)
                answers_list.append(answers)

            utils.save_results(args, cosines, innerproducts, contexts, answers_list, indices)
            if idx > args.end:
                break
    print('Collecting done.')


if __name__ == "__main__":
    main()
