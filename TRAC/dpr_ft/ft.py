# Here are some imports that we'll need
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
import argparse
from generateData import load_bioasq
from pathlib import Path
from haystack.document_stores.faiss import FAISSDocumentStore
from haystack import Document
from haystack.nodes.preprocessor import PreProcessor
import os
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_dir", type=str, default=".")
    parser.add_argument("--train_filename", type=str, default="bio.train.json")
    parser.add_argument("--dev_filename", type=str, default="bio.dev.json")
    parser.add_argument("--test_filename", type=str, default="bio.test.json")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--save_dir", type=str, default="dpr_bio")
    args = parser.parse_args()

    doc_dir = "."
    train_filename = "bio.train.json"
    dev_filename = "bio.dev.json"
    test_filename = "bio.test.json"

    query_model = "facebook/dpr-question_encoder-single-nq-base"
    passage_model = "facebook/dpr-ctx_encoder-single-nq-base"

    dataset = load_bioasq(Path("BioASQ-training11b/training11b.json"))
    datapoints = load_bioasq(Path("BioASQ-training11b/training11b.json"))['questions']
    questions = [d['body'] for d in datapoints]
    contexts = [d['snippets'] for d in datapoints]
    answers = [d['ideal_answer'] for d in datapoints]
    titles = [d['documents'] for d in datapoints]
    data = {'questions': questions, 'contexts': contexts, 'answers': answers, 'titles':titles}

    # document_store = ElasticsearchDocumentStore()
    document_store = InMemoryDocumentStore(use_bm25=True)
    
    retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=query_model,
            passage_embedding_model=passage_model,
            max_seq_len_query=64,
            max_seq_len_passage=256,
        )
    
    if args.train:
        retriever.train(
            data_dir=doc_dir,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=dev_filename,
            n_epochs=5,
            batch_size=16,
            grad_acc_steps=8,
            save_dir=args.save_dir,
            evaluate_every=50,
            embed_title=False,
            num_positives=1,
            num_hard_negatives=1,
        )
    
    # Load the saved index into a new DocumentStore instance:
    if os.path.exists("bio_faiss_index.faiss"):
        document_store_faiss = FAISSDocumentStore(faiss_index_path="bio_faiss_index.faiss")
    else:
        document_store_faiss = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

        documents = []
        count = 0
        for context in contexts:
            for tmp in context:
                documents.append(
                    Document(
                        content=tmp['text'],
                        content_type='text',
                        id=str(count),
                        meta={'name': tmp['document']}
                    )
                )
                count += 1

        preprocessor = PreProcessor(
            split_length=100,
            split_overlap=0,
            clean_empty_lines=False,
            split_respect_sentence_boundary=False,
            clean_whitespace=False,
        )

        document_store.write_documents(documents)
        document_store_faiss.write_documents(documents,)

        document_store_faiss.update_embeddings(retriever)
        document_store_faiss.save(index_path="bio_faiss_index.faiss")

    reloaded_retriever = DensePassageRetriever.load(load_dir=args.save_dir, document_store=document_store_faiss)

    def hit_rate(ctx_ids_list, retrieved_ids_list):
        rate = []
        for ctx_ids, retrieved_ids in zip(ctx_ids_list, retrieved_ids_list):
            # check if any id in true ids in retrieved ids
            if any([ctx_id in retrieved_ids for ctx_id in ctx_ids]):
                rate.append(True)
            else:
                rate.append(False)
        return np.mean(rate)

    # load in test data from bio.test.json
    with open("bio.train.json", "r") as f:
        train_data = json.load(f)

    # load in test data from bio.test.json
    with open("bio.dev.json", "r") as f:
        dev_data = json.load(f)

    # load in test data from bio.test.json
    with open("bio.test.json", "r") as f:
        test_data = json.load(f)
    
    # test_data = dev_data + test_data
    # test_data = train_data
    test_data = dev_data

    ctx_ids_list = []
    retrieved_ids_list = []
    queries = [data['question'] for data in test_data]
    ctx_ids_list = [[d['passage_id'] for d in data['positive_ctxs']] for data in test_data]
    retrieved_docs = reloaded_retriever.retrieve_batch(queries, top_k=20)
    retrieved_ids_list = [[d.id for d in docs] for docs in retrieved_docs]
    rate = hit_rate(ctx_ids_list, retrieved_ids_list)
    print("hit rate: ", rate)
