import os
import json
import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer
import rag_uncertainty
from transformers import AutoTokenizer, RagRetriever
from transformers import AutoModelForSequenceClassification, RagModel
import TRAC.utils as utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieve", action='store_true')
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--n_docs", type=int, default=20)
    parser.add_argument("--n_answers", type=int, default=40)
    parser.add_argument("--collect", type=float, default=float('inf'))
    parser.add_argument("--semantic", action='store_true')
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    # setup chatgpt
    utils.setup_openai()

    # setup retriever
    if args.retrieve:
        tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
        retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq",
                                                index_name="compressed")
        model = RagModel.from_pretrained("facebook/rag-sequence-nq",
                                        index_name='compressed',
                                        retriever=retriever).cuda()

    # ask questions
    questions = []
    reference_answers = []
    context_texts = []
    answers_list = []
    innerproducts = []
    cosines = []
    contexts = []
    answer_probs = []
    with open('biencoder-nq-dev.json', "r") as src_file:
        dpr_records = json.load(src_file)
        for idx, dpr_record in enumerate(tqdm(dpr_records)):
            if idx < args.start:
                continue
            question = dpr_record["question"].strip()
            questions.append(question)
            context_text = [context['text']
                            for context
                            in dpr_record['positive_ctxs']][0]
            context_texts.append(context_text)
            reference_answer = dpr_record["answers"][0]
            reference_answers.append(reference_answer)

            # retrieve context
            docs_dict, doc_scores_innerproduct, doc_scores_cosine, all_docs = \
                rag_uncertainty.retrieve(model, tokenizer,
                                         retriever, question,
                                         n_docs=20)
            # docs_titles = all_docs['title']
            docs_texts = all_docs['text']
            # docs_id = docs_dict['doc_ids'][0]

            if args.retrieve:
                # 2 use retrieved context

                # retrieve context
                dict, innerproduct, cosine, all_docs = \
                    rag_uncertainty.retrieve(model, tokenizer,
                                             retriever, question,
                                             n_docs=args.n_docs)
                # docs_titles = all_docs['title']
                docs_texts = all_docs['text']
                innerproducts.append(innerproduct[0].tolist())
                cosines.append(cosine[0].tolist())
                contexts.append(docs_texts)

                for text in docs_texts:
                    answers, probs = utils.ask_chatgpt(
                        question, context_text,
                        temperature=args.temp,
                        n_answers=args.n_answers)
                    answers_list.append(answers)
                    answer_probs.append(probs)
            else:
                answers, probs = utils.ask_chatgpt(
                            question, context_text,
                            temperature=args.temp,
                            n_answers=args.n_answers)
                answers_list.append(answers)
                answer_probs.append(probs)

            if args.retrieve:
                filepath = f'cosine_{args.n_answers}_{args.temp}.json'
                with open(os.path.join('collected', filepath), "w") as f:
                    json.dump(cosines, f)
                filepath = f'innerproduct_{args.n_answers}_{args.temp}.json'
                with open(os.path.join('collected', filepath), "w") as f:
                    json.dump(innerproducts, f)
                filepath = f'contexts_{args.n_answers}_{args.temp}.json'
                with open(os.path.join('collected', filepath), "w") as f:
                    json.dump(contexts, f)

            with open(os.path.join(
                            'collected', f'answers_{args.n_answers}_{args.temp}_logprob.jsonl'), "w") as f:
                for item in answers_list:
                    f.write(json.dumps(item) + "\n")
            with open(os.path.join(
                            'collected', f'answer_probs_{args.n_answers}_{args.temp}_logprob.jsonl'), "w") as f:
                for item in answer_probs:
                    f.write(json.dumps(item) + "\n")
            if idx > args.collect:
                break
    print('Collecting done.')


if __name__ == "__main__":
    main()
