import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune llama-2 model on specific dataset")
    parser.add_argument(
        "--task",
        type=str,
        default="nq",
    )
    parser.add_argument(
        "--train",
        action='store_true',
    )
    args = parser.parse_args()

    # The model that you want to train from the Hugging Face hub
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    # Fine-tuned model name
    new_model = f"../../finetuned_models/llama-2-7b-shuo-{args.task}-new"

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = "./results"

    # Number of training epochs
    num_train_epochs = 3

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = True

    # Batch size per GPU for training
    per_device_train_batch_size = 4

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 4

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule (constant a bit better than cosine)
    lr_scheduler_type = "constant"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 25

    # Log every X updates steps
    logging_steps = 25

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = 1024

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load the entire model on the GPU 0
    device_map = 'auto'

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    split = 'train'
    with open(f'data/biencoder-{args.task}-train.json', 'r') as f:
        dataset = json.load(f)
    dataset = Dataset.from_list(dataset[:15000])

    def formatting_prompts_func(element):
        output_texts = []
        for i in range(len(element['question'])):
            try:
                # add specifial token ~!~ to label the end of the answer
                text = f"### Question: {element['question'][i]}\n ### Context: {element['positive_ctxs'][i][0]['text']}\n ### Answer: {element['answers'][i][0]} ~!~"
                output_texts.append(text)
            except IndexError:
                pass
        return output_texts

    response_template = "### Answer:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer)

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        args=training_arguments,
        packing=packing,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    if args.train:
        # Train model
        trainer.train()

        # Save trained model
        trainer.model.save_pretrained(new_model)
        print('Training finished')

    # Reload model in FP16 and merge it with LoRA weights
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     low_cpu_mem_usage=True,
    #     return_dict=True,
    #     torch_dtype=torch.float16,
    #     device_map=device_map,
    # )
    # model = PeftModel.from_pretrained(base_model, new_model)
    # model = model.merge_and_unload()

    # # Reload tokenizer to save it
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"
