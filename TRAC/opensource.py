import transformers
from transformers import AutoTokenizer
import torch


def setup_openmodel(model='daryl149/llama-2-7b-chat-hf'):
    model = 'daryl149/llama-2-7b-chat-hf'

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        'text-generation',
        model=model,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    return model, pipeline, tokenizer


def ask_openmodel(prompt, pipeline, tokenizer, return_sequences=10):

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        max_length=400,
    )
    return sequences

def delete_model(model):
    del model
    torch.cuda.empty_cache()