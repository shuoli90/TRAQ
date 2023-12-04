from transformers import AutoTokenizer, RagModel, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
from datasets import load_dataset
import TRAC.utils as utils
import argparse
from rouge_score import rouge_scorer


def strip_title(title):
        if title.startswith('"'):
            title = title[1:]
        if title.endswith('"'):
            title = title[:-1]
        return title


def retrieve(model, tokenizer, retriever, question, n_docs=5):

    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # 1. Encode
    question_hidden_states = model.question_encoder(input_ids.cuda())[0].cpu()
    # 2. Retrieve
    docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), n_docs=n_docs, return_tensors="pt")
    all_docs = retriever.index.get_doc_dicts(docs_dict["doc_ids"])
    doc_scores_innerproduct = torch.bmm(
        question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    ).squeeze(1)
    doc_scores_cosine = F.cosine_similarity(question_hidden_states.unsqueeze(1), 
                                     docs_dict["retrieved_doc_embeds"].float(), dim=2)
    return docs_dict, doc_scores_innerproduct, doc_scores_cosine, all_docs[0]

def relevance(model, tokenizer, question, embeddings):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]
    question_hidden_states = model.question_encoder(input_ids.cuda())[0].cpu()

    relevance_scores_innerproduct = torch.bmm(
        question_hidden_states.unsqueeze(1), embeddings.float().transpose(1, 2)
    ).squeeze(1)
    relevance_scores_cosine = F.cosine_similarity(question_hidden_states.unsqueeze(1), 
                                           embeddings.float(), dim=2)
    return relevance_scores_innerproduct, relevance_scores_cosine

def load_triviaQA_dataset(datatype='rc', split="train"):
    """
    Load the TriviaQA dataset from Hugging Face Datasets.

    Args:
        split (str, optional): The dataset split to load. Options are 'train', 'validation', and 'test'. Defaults to 'train'.

    Returns:
        Dataset: The TriviaQA dataset split.
    """
    dataset = load_dataset("trivia_qa", "rc", split=split, cache_dir="/data3/shuoli/data/")
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='triviaQA')
    parser.add_argument('--metric', type=str, default='regex')
    args = parser.parse_args()

    split = 'validation'
    dataset = args.dataset
    metric = args.metric
    if dataset == 'triviaQA':
        if metric == 'regex':
            qa_dataset = load_triviaQA_dataset('rc', split)
        elif metric == 'context_match':
            qa_dataset = load_triviaQA_dataset('rc-wikipedia', split)
    elif dataset == 'fever':
        qa_dataset = load_dataset('copenlu/fever_gold_evidence', split)[split]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], 
                                      use_stemmer=True)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", 
                                             index_name="compressed")
    model = RagModel.from_pretrained("facebook/rag-sequence-nq", 
                                     index_name='compressed',
                                     retriever=retriever).cuda()
    questions = []
    true_scores_innerproduct = []
    reference_answers = []
    most_relevant_innerproduct = []
    true_scores_cosine = []
    most_relevant_cosine = []
    retrieve_scores_innerproduct = []
    retrieve_scores_cosine = []
    all_scores_cosine = []
    all_scores_innerproduct = []

    for idx, sample in enumerate(tqdm(qa_dataset)):
        if dataset == 'tirviaQA':
            # Extract the context, question, and answer from the sample
            context = sample["entity_pages"]["wiki_context"]
            if len(context) == 0:
                continue
            question = sample["question"]
            answer = sample["answer"]["normalized_value"]
        elif dataset == 'fever':
            context = [evidence[-1] for evidence in sample['evidence']]
            question = sample['claim']
            answer = sample['label']
        
        docs_dict, doc_scores_innerproduct, doc_scores_cosine, all_docs = \
            retrieve(model, tokenizer,
                     retriever, question, n_docs=10)
        docs_titles = all_docs['title']
        docs_texts = all_docs['text']
        docs_id = docs_dict['doc_ids'][0]
        retrieve_scores_innerproduct_tmp = []
        retrieve_scores_cosine_tmp = []
        all_scores_innerproduct.append(doc_scores_innerproduct[0].tolist())
        all_scores_cosine.append(doc_scores_cosine[0].tolist())
        if metric == 'regex':
            for i, docs_text in enumerate(docs_texts):
                has = utils.has_answer([answer], docs_text,
                                    tokenizer=tokenizer, match_type='regex')
                if has:
                    retrieve_scores_innerproduct_tmp.append(doc_scores_innerproduct[0][i].item())
                    retrieve_scores_cosine_tmp.append(doc_scores_cosine[0][i].item())
        else:
            for i, docs_text in enumerate(docs_texts):
                match = utils.context_match(docs_text, context, scorer)
                if match:
                    retrieve_scores_innerproduct_tmp.append(doc_scores_innerproduct[0][i].item())
                    retrieve_scores_cosine_tmp.append(doc_scores_cosine[0][i].item())
        retrieve_scores_cosine.append(retrieve_scores_cosine_tmp)
        retrieve_scores_innerproduct.append(retrieve_scores_innerproduct_tmp)
        

    with open(f'{dataset}_{metric}_retrieve_scores_innerproduct_new.json', "w") as f:
        json.dump(retrieve_scores_innerproduct, f)
    with open(f'{dataset}_{metric}_retrieve_scores_cosine_new.json', "w") as f:
        json.dump(retrieve_scores_cosine, f)
    with open(f'{dataset}_{metric}_all_scores_innerproduct_new.json', "w") as f:
        json.dump(all_scores_innerproduct, f)
    with open(f'{dataset}_{metric}_all_scores_cosine_new.json', "w") as f:
        json.dump(all_scores_cosine, f)
   
