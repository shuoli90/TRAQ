import argparse
from tqdm import tqdm
import utils as utils
import time
from tasks import RQA
import json
from kilt.retrievers import DPR_connector

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
    parser.add_argument(
        "--test_config",
        dest="test_config",
        type=str,
        default="kilt/configs/test_data.json",
        help="Test Configuration.",
    )
    parser.add_argument(
        "--model_configuration",
        "-c",
        dest="model_configuration",
        type=str,
        default="kilt/configs/retriever/default_dpr.json",
        help="model configuration",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        dest="model_name",
        type=str,
        default='dpr',
        help="retriever model name in {drqa,solr,dpr,blink,bm25}",
    )
    args = parser.parse_args()
    # load configs
    with open(args.test_config, "r") as fin:
        test_config_json = json.load(fin)
    
    if args.model_configuration:
        retriever = DPR_connector.DPR.from_config_file(
            args.model_name, args.model_configuration
        )
    else:
        retriever = DPR_connector.DPR.from_default_config("dpr")

    # setup chatgpt
    utils.setup_openai()

    if args.retrieve:
        retrieve_model, retrieve_tokenizer, retriever = \
            utils.load_retriever()
    breakpoint()

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
    corpus = data.corpus
    for idx, sample in enumerate(tqdm(data.samples)):
        question = sample.question
        context_ids = sample.context
        paragraphs = sample.paragraphs
        answers = sample.answer
        context_texts = [data.ks.get_page_by_id(id[0]) for id in context_ids if len(id) > 0]


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
