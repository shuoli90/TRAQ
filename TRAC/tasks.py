from datasets import load_dataset
from dataclasses import dataclass
from typing import List
import numpy as np
import logging
from kilt.knowledge_source import KnowledgeSource


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
        self.task = task
        self.load_dataset()
        self.ks = KnowledgeSource()

    def load_dataset(self) -> None:
        self.corpus = load_dataset("kilt_wikipedia", cache_dir="/data3/shuoli/data/")['full']
        kilt_nq = load_dataset("kilt_tasks", self.task, 'validation', cache_dir="/data3/shuoli/data/")['validation']
        self.samples = [self.build_sample(example) for example in kilt_nq]

    def build_sample(self, example) -> Sample:
        question = example["input"].strip()
        context_texts = [
            [ctx['wikipedia_id'] for ctx in context['provenance']]
            for context in example['output']
        ]
        paragraphs = [
            [[ctx['start_paragraph_id'], ctx['end_paragraph_id']] for ctx in context['provenance']]
            for context in example['output']
        ]
        reference_answers = [
            context['answer']
            for context in example['output']
        ]
        return Sample(question=question, context=context_texts, 
                      paragraphs=paragraphs, answer=reference_answers)

    def sample_subset(self, num=100, exclude=None):
        samples = self.samples
        lens = len(samples)
        index = np.random.permutation(lens).tolist()[:num if exclude is None else num+1]
        if exclude is not None and exclude in index:
            index.remove(exclude)
        else:
            index = index[:num]
        return [samples[i] for i in index]

    # get the context text from the context id
    def get_context_text(self, context_id):
        return self.ks.get_page_by_id(context_id)['text']