# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import math
import torch
import argparse
import datetime

from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

from masked_lm.datasets.lm_dataset import LMDataset

TOKENIZER_TYPE = [
    'byte-bpe',
    'bpe'
]

def train_language_model(
    data_dir: str='data/rewrite',
    pretrained_path: str=None,
    max_len: int=128,
    output_dir: str='./models/rewrite',
    learning_rate: float=5e-4,
    num_train_epochs: int=40,
    per_device_train_batch_size: int=64,
    save_steps: int=10_000,
    save_total_limit: int=2,
    prediction_loss_only: bool=True,
    **kwargs
):

    config = RobertaConfig.from_pretrained(pretrained_path, cache_dir=None)
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_path, do_lower_case=False, cache_dir=None, use_fast=False)

    model = RobertaForMaskedLM(config=config)
    print(f"\n The numbers of parameters: {model.num_parameters()}\n")

    train_dataset = LMDataset(
        root=data_dir, 
        mode='train',
        tokenizer=tokenizer,
        max_length=max_len,
    )
    eval_dataset = LMDataset(
        root=data_dir,
        mode='train',
        tokenizer=tokenizer,
        max_length=max_len,
    )
    print(f"\n---------- DATASET INFO ----------")
    print(f"The length of Train Dataset: {len(train_dataset)}")
    print(f"The length of Eval Dataset: {len(eval_dataset)}")
    print(f"----------------------------------")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.2
    )

    print(f"Check that PyTorch sees it: {torch.cuda.is_available()}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        prediction_loss_only=prediction_loss_only
    )
    print(f"\n---------- TRAINING PRETRAIN LANGUAGE MODEL ----------")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    now = datetime.datetime.now()
    trainer.train()

    print(f"\n Training time: {datetime.datetime.now() - now}.")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = {}

    results = trainer.evaluate()
    print(results)

    perplexity = math.exp(results["eval_loss"]) if "eval_loss" in results.keys() else None
    results["perplexity"] = perplexity

    output_eval_file = os.path.join(output_dir, "eval_results_lm.txt")
    with open(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key, value in sorted(results.items()):
            print(f" {key} = {value}")
            writer.write(f"{key} = {value}\n")

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default='data/rewrite',
                        help="The path to the dataset to use.",)
    parser.add_argument("--bs", type=int, default=48, 
                        help="Total number of batch size to perform.")
    parser.add_argument("--nepochs", type=int, default=5, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=123,
                        help="A seed for reproducible training.")

    args = parser.parse_args()

    LEARNING_RATE = 5e-5
    
    ### TRAIN LANGUAGE MODEL
    print("\nTRAINING PRE_TRAINING LANGUAGE MODEL...")
    train_language_model(
        data_dir=args.dataset_path,
        pretrained_path='models/BDIRoBerta',
        learning_rate=LEARNING_RATE, 
        output_dir='./models/rewrite',
        num_train_epochs=args.nepochs,
        per_device_train_batch_size=args.bs,
        save_steps=10_000,
        save_total_limit=2
    )
