from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from datasets import load_dataset
import torch

torch.set_grad_enabled(False)
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
wiki = load_dataset(path='wiki_dpr', name='psgs_w100.multiset.compressed', split='train')

# from datasets import load_dataset
# ds = load_dataset('crime_and_punish', split='train[:10]')
# ds_with_embeddings = ds.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example["line"], return_tensors="pt"))[0][0].numpy()})

# ds_with_embeddings.add_faiss_index(column='embeddings')
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question = "Is it serious ?"
question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
scores, retrieved_examples = wiki.get_nearest_examples('embeddings', question_embedding, k=3)
# retrieved_examples["line"][0]
# ds_with_embeddings.save_faiss_index('embeddings', 'my_index.faiss')

breakpoint()