from datasets import load_dataset
from dataclasses import dataclass
from typing import List
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Sample:
    question: str
    context: List[str]
    paragraphs: List[List[int]]
    answer: List[str]  


class RQA:
    def __init__(self, task='nq') -> None:
        with open("kilt/configs/dev_data.json", "r") as fin:
            self.dev_config_json = json.load(fin)
        self.task = task
        self.query_data, self.validated_data, self.elements = self.load_dataset()
    
    def load_dataset(self) -> None:
        for task_family, datasets in self.dev_config_json.items():
            for dataset_name, dataset_file in datasets.items():
                if dataset_name == self.task:
                    raw_data = utils.load_data(dataset_file)

                    # consider only valid data - filter out invalid
                    validated_data = {}
                    query_data = []
                    elements = []
                    for element in raw_data:
                        elements.append(element)
                        # if utils.validate_datapoint(element, logger=None):
                        if element["id"] in validated_data:
                            raise ValueError("ids are not unique in input data!")
                        validated_data[element["id"]] = element
                        query_data.append(
                            {"query": element["input"], "id": element["id"]}
                        )
                    return query_data, validated_data, elements

class RQA_dpr:
    def __init__(self, task='nq') -> None:
        assert task in ['nq', 'trivia', 'squad1']
        self.task = task
        self.query_data, self.validated_data, self.elements = self.load_dataset()

    def load_dataset(self) -> None:
        with open(f"data/biencoder-{self.task}-dev.json", "r") as fin:
            nq_dpr = json.load(fin)

        elements = []
        query_data = []
        validated_data = {}
        for idx, record in enumerate(nq_dpr):
            elements.append(record)
            validated_data[idx] = record
            query_data.append(
                {"query": record["question"], "id": idx}
            )
        return query_data, validated_data, elements




class bio_dpr:
    def __init__(self, task='nq') -> None:
        assert task in ['bio']
        self.task = task
        # self.questions, self.contexts, self.answers = self.load_dataset()
    
    def load_bioasq(self, path):
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")

        with open(path, "r") as f:
            data = json.load(f)
    
        return data

    def load_dataset(self) -> None:
        bio_dpr = json.load(open("bio_fact.json", "r"))
        return bio_dpr['questions'], bio_dpr['contexts'], bio_dpr['answers']
