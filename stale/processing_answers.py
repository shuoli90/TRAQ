import os
import json
import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import TRAC.utils as utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", action='store_true')
    parser.add_argument("--temp", type=float, default=1.5)
    parser.add_argument("--n_answers", type=int, default=40)
    parser.add_argument("--metric", type=str, default="rouge1")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Read the JSONL file and load the data into a list
    answers_list = []
    with open(os.path.join(
                    'collected',
                    f'answers_{args.n_answers}_{args.temp}.jsonl'), "r") as f:
        for line in f:
            item = json.loads(line.strip())
            answers_list.append(item)

    # # setup scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    if args.semantic:
        # setup semantic model
        semantic_tokenizer = \
            AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        semantic_model = \
            AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli"
            ).cuda()
    
    true_scores = []
    all_true_context_scores = []
    # ask questions
    with open('biencoder-nq-dev.json', "r") as src_file:
        dpr_records = json.load(src_file)
        for idx, dpr_record in enumerate(tqdm(dpr_records)):
            question = dpr_record["question"].strip()
            context_text = [context['text']
                            for context
                            in dpr_record['positive_ctxs']][0]
            reference_answer = dpr_record["answers"][0]
            prompt = f"""
            Answer the following question based on the given context; \
            Answer the question with only one key word.

            Question: '''{question}'''

            Context: '''{context_text}'''
            """

            answers = answers_list[idx]

            if args.semantic:
                semantic_clusterring, semantic_probs, item_occurance = \
                    utils.compute_semantic_clusterring(
                        model=semantic_model,
                        tokenizer=semantic_tokenizer,
                        question=prompt, answers=answers,
                        scorer=scorer)
            else:
                semantic_clusterring, semantic_probs, item_occurance = \
                    utils.compute_keyword_clusterring(
                        answers=answers,
                        scorer=scorer)
            probs = []
            for predicted_answer in semantic_clusterring.keys():
                concept_id = semantic_clusterring[predicted_answer]
                repeat = item_occurance[predicted_answer]
                prob = semantic_probs[concept_id]
                probs.extend([prob] * repeat)
                scores = scorer.score(reference_answer, predicted_answer)
                if args.metric == "rouge1":
                    scores = scores['rouge1'][2]
                else:
                    scores = scores['rougeL'][2]
                if scores > args.threshold:
                    true_scores.extend([prob] * repeat)
                else:
                    true_scores.extend([0.0] * repeat)
            all_true_context_scores.append(probs)

            if idx == len(answers_list) - 1:
                break
    with open(os.path.join("collected",
                           f'true_scores_{args.n_answers}_{args.temp}_{args.metric}_{args.threshold}.json'),
              "w") as f:
        json.dump(true_scores, f)
    with open(os.path.join("collected",
                           f'all_true_context_scores_{args.n_answers}_{args.temp}_{args.metric}_{args.threshold}.json'),
              "w") as f:
        json.dump(all_true_context_scores, f)
    print('Collected Scores')




if __name__ == "__main__":
    main()
