import os
import requests
from datasets import load_dataset

def load_dpr_wiki_data():
    dataset = load_dataset("wiki_dpr", embeddings_name="nq", 
                                index_name="compressed", 
                                cache_dir="/data3/shuoli/data/")['train']
    
    return dataset

def search_passages(dataset, title):
    passages = []

    for document in dataset:
        if document['title'] == title:
            passages.append(document['text'])

    return passages

def main():
    dataset = load_dpr_wiki_data()
    print(f"Loaded DPR Wikipedia dataset with {len(dataset)} records.")

    wikipedia_title = "Aaron"
    passages = search_passages(dataset, wikipedia_title)
    if passages:
        print(f"Passages found for '{wikipedia_title}':")
        for passage in passages:
            print(f"\n{passage}")
    else:
        print(f"No passages found for '{wikipedia_title}'")

if __name__ == "__main__":
    main()
