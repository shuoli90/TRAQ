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
    parser.add_argument("--n_answers", type=int, default=5)
    parser.add_argument("--collect", type=float, default=float('inf'))
    parser.add_argument("--semantic", action='store_true')
    args = parser.parse_args()

    # setup chatgpt
    utils.setup_openai()

    # setup scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    if args.retrieve:
        # setup retriever
        tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
        retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq",
                                                 index_name="compressed")
        model = RagModel.from_pretrained("facebook/rag-sequence-nq",
                                         index_name='compressed',
                                         retriever=retriever).cuda()

    if args.semantic:
        # setup semantic model
        semantic_tokenizer = \
            AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        semantic_model = \
            AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli"
            ).cuda()

    # ask questions
    questions = []
    # true_scores = []
    retrieved_context_scores = []
    # all_true_context_scores = []
    all_retrieved_context_scores = []
    reference_answers = []
    context_texts = []
    answers_list = []
    with open('biencoder-nq-dev.json', "r") as src_file:
        dpr_records = json.load(src_file)
        for idx, dpr_record in enumerate(tqdm(dpr_records)):
            if idx < 776:
                continue
            question = dpr_record["question"].strip()
            questions.append(question)
            context_text = [context['text']
                            for context
                            in dpr_record['positive_ctxs']][0]
            context_texts.append(context_text)
            reference_answer = dpr_record["answers"][0]
            reference_answers.append(reference_answer)

            if args.retrieve:
                # 2 use retrieved context

                # retrieve context
                dict, innerproduct, cosine, all_docs = \
                    rag_uncertainty.retrieve(model, tokenizer,
                                             retriever, question,
                                             n_docs=args.n_docs)
                # docs_titles = all_docs['title']
                docs_texts = all_docs['text']
                # docs_id = docs_dict['doc_ids'][0]

                probs = []
                for text in docs_texts:
                    if args.semantic:
                    #     semantic_clusterring, semantic_probs, item_occurance \
                    #         = utils.get_completion(
                    #             question, text,
                    #             semantic_model=semantic_model,
                    #             semantic_tokenizer=semantic_tokenizer,
                    #             temperature=args.temp,
                    #             n_answers=args.n_answers,
                    #             scorer=None,
                    #             semantic=True)
                    # else:
                    #     semantic_clusterring, semantic_probs, item_occurance \
                    #         = utils.get_completion(
                    #             question, text,
                    #             semantic_model=None,
                    #             semantic_tokenizer=None,
                    #             temperature=args.temp,
                    #             n_answers=args.n_answers,
                    #             scorer=scorer,
                    #             semantic=False)

                    # for predicted_answer in semantic_clusterring.keys():
                    #     concept_id = semantic_clusterring[predicted_answer]
                    #     repeat = item_occurance[predicted_answer]
                    #     prob = semantic_probs[concept_id]
                    #     probs.extend([prob] * repeat)
                    #     scores = scorer.score(reference_answer,
                    #                           predicted_answer)
                    #     scores = scores['rouge1'][2]
                    #     if scores > 0.5:
                    #         retrieved_context_scores.extend([prob] * repeat)
                    #     else:
                    #         retrieved_context_scores.extend([0.0] * repeat)
                    # all_retrieved_context_scores.append(probs)

            answers = utils.get_completion(
                        question, context_text,
                        temperature=args.temp,
                        n_answers=args.n_answers)
            answers_list.append(answers)

            if idx < 5:
                tmp = {}
                tmp['question'] = question
                tmp['context'] = context_text
                tmp['reference_answer'] = reference_answer
                tmp['answers'] = answers
                with open(
                        os.path.join('collected', f"{idx}.json"), "w") \
                        as outfile:
                    # Write each key as a separate line in the file
                    for key, value in tmp.items():
                        outfile.write(json.dumps({key: value}) + '\n')
            if idx > args.collect:
                break

            if args.retrieve:
                with open(os.path.join(
                            'collected',
                            f'retrieved_context_scores_{args.n_answers}_{args.temp}.json'),
                        "w") as f:
                    json.dump(retrieved_context_scores, f)
                with open(os.path.join(
                            'collected',
                            f'all_retrived_context_scores_{args.n_answers}_{args.temp}.json'),
                        "w") as f:
                    json.dump(all_retrieved_context_scores, f)
            with open(os.path.join(
                            'collected', f'answers_{args.n_answers}_{args.temp}_new_new.jsonl'), "w") as f:
                for item in answers_list:
                    f.write(json.dumps(item) + "\n")
            print('Collecting done.')


if __name__ == "__main__":
    main()
