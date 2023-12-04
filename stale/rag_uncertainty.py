from transformers import AutoTokenizer, RagModel, RagRetriever
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import TRAC.utils as utils
import argparse


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
    docs_dict = retriever(input_ids.numpy(),
                          question_hidden_states.detach().numpy(),
                          n_docs=n_docs, return_tensors="pt")
    all_docs = retriever.index.get_doc_dicts(docs_dict["doc_ids"])
    doc_scores_innerproduct = torch.bmm(
        question_hidden_states.unsqueeze(1),
        docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    ).squeeze(1)
    doc_scores_cosine = F.cosine_similarity(
        question_hidden_states.unsqueeze(1),
        docs_dict["retrieved_doc_embeds"].float(), dim=2)
    return docs_dict, doc_scores_innerproduct, doc_scores_cosine, all_docs[0]


def relevance(model, tokenizer, question, embeddings):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]
    question_hidden_states = model.question_encoder(input_ids.cuda())[0].cpu()

    relevance_scores_innerproduct = torch.bmm(
        question_hidden_states.unsqueeze(1), embeddings.float().transpose(1, 2)
    ).squeeze(1)
    relevance_scores_cosine = F.cosine_similarity(
        question_hidden_states.unsqueeze(1),
        embeddings.float(), dim=2)
    return relevance_scores_innerproduct, relevance_scores_cosine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='match')
    parser.add_argument('--n_docs', type=int, default=100)
    parser.add_argument('--exact', action='store_true')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
    if args.exact:
        retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq",
                                                 index_name="exact")
        model = RagModel.from_pretrained("facebook/rag-sequence-nq",
                                         index_name='exact',
                                         retriever=retriever).cuda()
    else:
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
    with open('biencoder-nq-dev.json', "r") as src_file:
        dpr_records = json.load(src_file)
        for idx, dpr_record in enumerate(tqdm(dpr_records)):
            if idx != 1501:
                continue
            question = dpr_record["question"].strip()
            questions.append(question)
            context_titles = [context["title"]
                              for context
                              in dpr_record["positive_ctxs"]]
            context_text = [context['text']
                            for context
                            in dpr_record['positive_ctxs']][0]
            # reference_answer = dpr_record["answers"][0]
            reference_answer = dpr_record["answers"]
            reference_answers.append(reference_answer)
            context_id = torch.tensor([int(context['passage_id'])
                                       for context
                                       in dpr_record['positive_ctxs']])

            all_context = retriever.index.get_doc_dicts(context_id)
            context_embeddings = torch.tensor([context['embeddings']
                                               for context
                                               in all_context]).unsqueeze(0)
            relevance_scores_innerproduct, relevance_scores_cosine = \
                relevance(model, tokenizer, question, context_embeddings)
            true_scores_innerproduct.append(
                relevance_scores_innerproduct[0].tolist())
            most_relevant_innerproduct.append(
                relevance_scores_innerproduct[0].max().item())
            true_scores_cosine.append(
                relevance_scores_cosine[0].tolist())
            most_relevant_cosine.append(
                relevance_scores_cosine[0].max().item())

            docs_dict, doc_scores_innerproduct, doc_scores_cosine, all_docs = \
                retrieve(model, tokenizer,
                         retriever, question, n_docs=args.n_docs)
            docs_titles = all_docs['title']
            docs_texts = all_docs['text']
            docs_id = docs_dict['doc_ids'][0]
            breakpoint()

            retrieve_score_innerproduct_tmp = []
            retrieve_score_cosine_tmp = []
            all_scores_innerproduct.append(doc_scores_innerproduct[0].tolist())
            all_scores_cosine.append(doc_scores_cosine[0].tolist())
            for i, docs_text in enumerate(docs_texts):
                has = utils.has_answer(reference_answer, docs_text,
                                       tokenizer=tokenizer, match_type='regex')
                if has:
                    # print(docs_text)
                    retrieve_score_innerproduct_tmp.append(
                        doc_scores_innerproduct[0][i].item())
                    retrieve_score_cosine_tmp.append(
                        doc_scores_cosine[0][i].item())
            retrieve_scores_innerproduct.append(
                retrieve_score_innerproduct_tmp)
            retrieve_scores_cosine.append(retrieve_score_cosine_tmp)

            # for title, score_innerprodcut, scores_cosine in zip(docs_titles,
            #                                                     doc_scores_innerproduct[0],
            #                                                     doc_scores_cosine[0]):
            #     title = strip_title(title)
            #     if title in context_titles:
            #         retrieve_score_innerproduct_tmp.append(score_innerprodcut.item())
            #         retrieve_score_cosine_tmp.append(scores_cosine.item())
            # retrieve_scores_innerproduct.append(retrieve_score_innerproduct_tmp)
            # retrieve_scores_cosine.append(retrieve_score_cosine_tmp)
    with open(f'retrieve_scores_innerproduct_{args.exp}_{args.n_docs}_comp.json', "w") as f:
        json.dump(retrieve_scores_innerproduct, f)
    with open(f'retrieve_scores_cosine_{args.exp}_{args.n_docs}_comp.json', "w") as f:
        json.dump(retrieve_scores_cosine, f)
    with open(f'true_scores_innerproduct_{args.exp}_{args.n_docs}_comp.json', "w") as f:
        json.dump(true_scores_innerproduct, f)
    with open(f'true_scores_cosine_{args.exp}_{args.n_docs}_comp.json', "w") as f:
        json.dump(true_scores_cosine, f)
    with open(f'most_relevant_innerproduct_{args.exp}_{args.n_docs}_comp.json', "w") as f:
        json.dump(most_relevant_innerproduct, f)
    with open(f'most_relevant_cosine_{args.exp}_{args.n_docs}_comp.json', "w") as f:
        json.dump(most_relevant_cosine, f)
    with open(f'all_scores_innerproduct_{args.exp}_{args.n_docs}_comp.json', "w") as f:
        json.dump(all_scores_innerproduct, f)
    with open(f'all_scores_cosine_{args.exp}_{args.n_docs}.json', "w") as f:
        json.dump(all_scores_cosine, f)
    with open(f'questions_{args.exp}_{args.n_docs}_comp.json', "w") as f:
        json.dump(questions, f)
