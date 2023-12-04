import torch
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRReader, DPRConfig
import faiss
import numpy as np
import requests
from datasets import load_dataset

def load_dpr_wiki_data():
    dataset = load_dataset("wiki_dpr", embeddings_name="nq", 
                                index_name="compressed", 
                                cache_dir="/data3/shuoli/data/")['train']
    
    return dataset

# Load the pretrained DPR models
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-multiset-base')
reader = DPRReader.from_pretrained('facebook/dpr-reader-multiset-base')


dataset = load_dpr_wiki_data()
breakpoint()

# # Load the DPR Wiki dataset
# url = 'https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz'
# response = requests.get(url, stream=True)
# data = response.iter_lines()
# breakpoint()
# dataset = [line.decode('utf-8').split('\t') for line in data]

# Encode the questions and contexts
questions = ["What is artificial intelligence?", "Who wrote The Great Gatsby?"]
contexts = [d[1] for d in dataset]
question_encodings = question_encoder(questions, return_tensors='pt').last_hidden_state
context_encodings = context_encoder(contexts, return_tensors='pt').last_hidden_state

# Perform nearest neighbor search with Faiss
d = question_encodings.size(1)
index = faiss.IndexFlatIP(d)
index.add(context_encodings.cpu().numpy())
k = 5  # number of documents to retrieve for each question
_, I = index.search(question_encodings.cpu().numpy(), k)

# Print the retrieved documents
for i, question in enumerate(questions):
    print(f"\nQuestion: {question}")
    for j in range(k):
        doc_id = I[i][j]
        print(f"\nDocument {j+1}:")
        print(f"Title: {dataset[doc_id][0]}")
        print(f"Context: {dataset[doc_id][1]}")
