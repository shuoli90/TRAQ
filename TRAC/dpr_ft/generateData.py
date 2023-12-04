"""
Script to convert a SQuAD-like QA-dataset format JSON file to DPR Dense Retriever training format

Usage:
    squad_to_dpr.py --squad_input_filename <squad_input_filename> --dpr_output_filename <dpr_output_filename> [options]
Arguments:
    <squad_file_path>                   SQuAD file path
    <dpr_output_path>                   DPR output folder path
    --num_hard_negative_ctxs HNEG       Number of hard negative contexts [default: 30:int]
    --split_dataset                     Whether to split the created dataset or not [default: False]

SQuAD format
{
    version: "Version du dataset"
    data:[
            {
                title: "Titre de l'article Wikipedia"
                paragraphs:[
                    {
                        context: "Paragraph de l'article"
                        qas:[
                            {
                                id: "Id du pair question-réponse"
                                question: "Question"
                                answers:[
                                    {
                                        "answer_start": "Position de la réponse"
                                        "text": "Réponse"
                                    }
                                ],
                                is_impossible: (not in v1)
                            }
                        ]
                    }
                ]
            }
    ]
}


DPR format
[
    {
        "question": "....",
        "answers": ["...", "...", "..."],
        "positive_ctxs": [{
            "title": "...",
            "text": "...."
        }],
        "negative_ctxs": ["..."],
        "hard_negative_ctxs": ["..."]
    },
    ...
]
"""

from typing import Dict, Iterator, Tuple, List, Union

import json
import logging
import argparse
import subprocess
from time import sleep
from pathlib import Path
from itertools import islice
from haystack import Document
import os

from tqdm import tqdm

from haystack.document_stores.base import BaseDocumentStore
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore  # keep it here !
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores.faiss import FAISSDocumentStore  # keep it here !
from haystack.nodes.retriever.sparse import BM25Retriever  # keep it here !  # pylint: disable=unused-import
from haystack.nodes.retriever.dense import DensePassageRetriever  # keep it here !  # pylint: disable=unused-import
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.retriever.base import BaseRetriever
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install farm-haystack[elasticsearch]'") as es_import:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class HaystackDocumentStore:
    def __init__(self, store_type: str = "ElasticsearchDocumentStore", **kwargs):
        es_import.check()

        if store_type not in ["ElasticsearchDocumentStore", "FAISSDocumentStore"]:
            raise Exception(
                "At the moment we only deal with one of these types: ElasticsearchDocumentStore, FAISSDocumentStore"
            )

        self._store_type = store_type
        self._kwargs = kwargs
        self._preparation = {
            "ElasticsearchDocumentStore": self.__prepare_ElasticsearchDocumentStore,
            "FAISSDocumentStore": self.__prepare_FAISSDocumentStore,
        }

    def get_document_store(self):
        self._preparation[self._store_type]()
        return globals()[self._store_type](**self._kwargs)

    @staticmethod
    def __prepare_ElasticsearchDocumentStore():
        es = Elasticsearch(["http://localhost:9200/"], verify_certs=True)
        if not es.ping():
            logger.info("Starting Elasticsearch ...")
            status = subprocess.run(
                ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'], shell=True
            )
            if status.returncode:
                raise Exception("Failed to launch Elasticsearch.")
            sleep(30)

        es.indices.delete(index="document", ignore=[400, 404])

    def __prepare_FAISSDocumentStore(self):
        pass

class HaystackRetriever:
    def __init__(self, document_store: BaseDocumentStore, retriever_type: str, **kwargs):
        if retriever_type not in ["BM25Retriever", "DensePassageRetriever", "EmbeddingRetriever"]:
            raise Exception("Use one of these types: BM25Retriever", "DensePassageRetriever", "EmbeddingRetriever")
        self._retriever_type = retriever_type
        self._document_store = document_store
        self._kwargs = kwargs

    def get_retriever(self):
        return globals()[self._retriever_type](document_store=self._document_store, **self._kwargs)


def get_number_of_questions(bioasq_data: list):
    return len(bioasq_data)

def load_bioasq(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    with open(path, "r") as f:
        data = json.load(f)
    
    return data

def get_hard_negative_contexts(retriever: BaseRetriever, question: str, answers: List[str], n_ctxs: int = 30, positive_ids: List[int] = []):
    list_hard_neg_ctxs = []
    retrieved_docs = retriever.retrieve(query=question, top_k=n_ctxs, index="document")
    for retrieved_doc in retrieved_docs:
        retrieved_doc_id = retrieved_doc.meta.get("name", "")
        retrieved_doc_text = retrieved_doc.content
        retrieved_doc_idx = retrieved_doc.id
        if any(str(answer).lower() in retrieved_doc_text.lower() for answer in answers):
            continue
        if retrieved_doc_idx in positive_ids:
            continue
        list_hard_neg_ctxs.append({"title": retrieved_doc_id, "text": retrieved_doc_text, "passage_id": retrieved_doc_idx})

    return list_hard_neg_ctxs

def create_dpr_training_dataset(data: dict, retriever: BaseRetriever, num_hard_negative_ctxs: int = 30):
    n_non_added_questions = 0
    n_questions = 0
    
    for question, answers, contexts, titles in zip(data['questions'], data['answers'], data['contexts'], data['titles']):
        positive_ctxs = [{"title": title, "text": context['text'], "passage_id": context['id']} for title, context in zip(titles, contexts)]
        positive_ids = [ctx["passage_id"] for ctx in positive_ctxs]
        hard_negative_ctxs = get_hard_negative_contexts(
                retriever=retriever, question=question, answers=answers, n_ctxs=num_hard_negative_ctxs, positive_ids=positive_ids,
            )
        if not hard_negative_ctxs or not positive_ctxs:
            logger.error(
                "No retrieved candidates for article %s, with question %s", titles[0], question
            )
            n_non_added_questions += 1
            continue
        dict_DPR = {
            "question": question,
            "answers": answers,
            "positive_ctxs": positive_ctxs,
            "negative_ctxs": [],
            "hard_negative_ctxs": hard_negative_ctxs,
        }
        n_questions += 1
        yield dict_DPR

    logger.info("Number of skipped questions: %s", n_non_added_questions)
    logger.info("Number of added questions: %s", n_questions)

def save_dataset(iter_dpr: Iterator, dpr_output_filename: Path, total_nb_questions: int, split_dataset: bool):
    if split_dataset:
        nb_train_examples = int(total_nb_questions * 0.8)
        nb_dev_examples = int(total_nb_questions * 0.19)

        train_iter = islice(iter_dpr, nb_train_examples)
        dev_iter = islice(iter_dpr, nb_dev_examples)

        dataset_splits = {
            dpr_output_filename.parent / f"{dpr_output_filename.stem}.train.json": train_iter,
            dpr_output_filename.parent / f"{dpr_output_filename.stem}.dev.json": dev_iter,
            dpr_output_filename.parent / f"{dpr_output_filename.stem}.test.json": iter_dpr,
        }
    else:
        dataset_splits = {dpr_output_filename: iter_dpr}
    for path, set_iter in dataset_splits.items():
        with open(path, "w", encoding="utf-8") as json_ds:
            json.dump(list(set_iter), json_ds, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a SQuAD JSON format dataset to DPR format.")
    parser.add_argument(
        "--num_hard_negative_ctxs",
        dest="num_hard_negative_ctxs",
        help="Number of hard negative contexts to use",
        metavar="num_hard_negative_ctxs",
        default=30,
    )
    parser.add_argument(
        "--split_dataset",
        dest="split_dataset",
        action="store_true",
        help="Whether to split the created dataset or not (default: False)",
    )
    parser.add_argument(
        "--dpr_output_filename",
        dest="dpr_output_filename",
        help="The name of the DPR JSON formatted output file",
        metavar="DPR_out",
        required=True,
    )

    args = parser.parse_args()

    num_hard_negative_ctxs = args.num_hard_negative_ctxs
    split_dataset = args.split_dataset

    retriever_dpr_config = {"use_gpu": True}
    store_dpr_config = {"embedding_field": "embedding", "embedding_dim": 768}

    datapoints = load_bioasq(Path("BioASQ-training11b/training11b.json"))['questions']
    contexts = [d['snippets'] for d in datapoints]

    # document_store = ElasticsearchDocumentStore()
    document_store = InMemoryDocumentStore(use_bm25=True)
    
    documents = []
    count = 0
    for context in contexts:
        for tmp in context:
            tmp['id'] = str(count)
            documents.append(
                Document(
                    content=tmp['text'],
                    content_type='text',
                    id=count,
                    meta={'name': tmp['document']}
                )
            )
            count += 1
    breakpoint()

    preprocessor = PreProcessor(
        split_length=100,
        split_overlap=0,
        clean_empty_lines=False,
        split_respect_sentence_boundary=False,
        clean_whitespace=False,
    )

    document_store.write_documents(documents)

    retriever_bm25_config: dict = {}
    retriever_type_config: Tuple[str, dict] = ("BM25Retriever", retriever_bm25_config)
    retriever_factory = HaystackRetriever(
        document_store=document_store, retriever_type=retriever_type_config[0], **retriever_type_config[1]
    )
    retriever = retriever_factory.get_retriever()

    questions = []
    contexts = []
    answers = []
    titles = []
    questions_fact = []
    contexts_fact = []
    answers_fact = []
    titles_fact = []
    for d in datapoints:
        if d['type'] == 'factoid':
            questions_fact.append(d['body'])
            contexts_fact.append(d['snippets'])
            answers_fact.append(d['exact_answer'])
            titles_fact.append(d['documents'])
        else:
            questions.append(d['body'])
            contexts.append(d['snippets'])
            answers.append(d['ideal_answer'])
            titles.append(d['documents'])
    data = {'questions': questions, 'contexts': contexts, 'answers': answers, 'titles': titles}
    data_fact = {'questions': questions_fact, 'contexts': contexts_fact, 'answers': answers_fact, 'titles': titles_fact}
    # save data_fact to json file
    with open('bio_fact.json', 'w') as f:
        json.dump(data_fact, f)
    
    iter_DPR = create_dpr_training_dataset(
        data, retriever=retriever, num_hard_negative_ctxs=num_hard_negative_ctxs
    )

    save_dataset(
        iter_dpr=iter_DPR,
        dpr_output_filename=Path(args.dpr_output_filename),
        total_nb_questions=len(questions),
        split_dataset=split_dataset,
    )
    print('done')
