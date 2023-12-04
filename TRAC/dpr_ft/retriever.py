from haystack.document_stores import FAISSDocumentStore
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Any, Literal, Dict
from dataclasses import dataclass
from haystack import Document
from haystack.nodes.preprocessor import PreProcessor

def load_bioasq(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    with open(path, "r") as f:
        data = json.load(f)
    
    return data

if __name__ == "__main__":
    datapoints = load_bioasq(Path("BioASQ-training11b/training11b.json"))['questions']
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    contexts = [d['snippets'] for d in datapoints]
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
    breakpoint()