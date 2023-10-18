import transformers
from transformers import AutoTokenizer
import torch


def setup_openmodel(model='tiiuae/falcon-7b-instruct', max_length=250):
    # model = 'tiiuae/falcon-7b'

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast = False)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=max_length,
    )
    return model, pipeline, tokenizer


def ask_openmodel(prompt, pipeline, tokenizer, return_sequences=10, max_length=250):

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )
    return sequences

def delete_model(model):
    del model
    torch.cuda.empty_cache()