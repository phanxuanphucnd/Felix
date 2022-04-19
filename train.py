# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import os
import math
import torch
import random
import argparse
import numpy as np

from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    set_seed,
    get_scheduler,
    SchedulerType,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM
)

from masked_lm.constants import *
from masked_lm.pointing_converter import PointingConverter
from masked_lm.insertion_converter import InsertionConverter
from masked_lm.datasets.insertion_dataset import InsertionDataset

def main(args):
     # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    print(f"\n{accelerator.state}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = RobertaConfig.from_pretrained(args.lm_pretrained_path, cache_dir=None)
    tokenizer = RobertaTokenizer.from_pretrained(
        args.lm_pretrained_path, do_lower_case=False, 
        cache_dir=None, use_fast=False
    )

    model = RobertaForMaskedLM(config=config)
    print(f"\nThe numbers parameters of model: {model.num_parameters()}\n")

    point_converter = PointingConverter({}, do_lower_case=True)

    insertion_converter = InsertionConverter(
        max_seq_length=args.max_seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
        label_map_file='./ext/label_map.json',
        tokenizer=tokenizer,
        fall_back_mode='random'
    )

    train_dataset = InsertionDataset(
        mode='train',
        data_path=args.train_path,
        tokenizer=tokenizer,
        label_map_file=args.label_map_file,
        point_converter=point_converter,
        insertion_converter=insertion_converter,
        use_open_vocab=True,
        max_seq_length=args.max_seq_length,
        do_lower_case=True
    )
    collate_fn = train_dataset._collate_func
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_dataloader = None
    if args.eval_path:
        eval_dataset = InsertionDataset(
            mode='eval',
            data_path=args.eval_path,
            tokenizer=tokenizer,
            label_map_file=args.label_map_file,
            point_converter=point_converter,
            insertion_converter=insertion_converter,
            use_open_vocab=True,
            max_seq_length=args.max_seq_length,
            do_lower_case=True
        )
        
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    print("\n")
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Batch size = {args.batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print("\n")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=False)
    completed_steps = 0

    best_loss = float('inf')

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            # loss.backward()
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            # losses.append(loss.item())
            losses.append(accelerator.gather(loss.repeat(args.batch_size)))
        
        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        mean_loss = torch.mean(losses)
        try:
            perplexity = math.exp(mean_loss)
        except OverflowError:
            perplexity = float("inf")

        print(f"Epoch {epoch}: perplexity: {perplexity}")

        if mean_loss <= best_loss:
            best_loss = mean_loss
            print(f"Save best model to `{args.output_dir}`.")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
    
    if args.save_final_model:
        print(f"Save final model to `{args.output_dir}`.")
        model.save_pretrained(os.path.join(args.output_dir, 'final_model'))
        tokenizer.save_pretrained(os.path.join(args.output_dir, 'final_model'))
        

def init_seed(SEED):
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # DATA PARAMS
    parser.add_argument(
        "--train_path", type=str, default='./data/train.csv',
        help="The path to the train dataset to use."
    )
    parser.add_argument(
        "--eval_path", type=str, default='./data/test.csv',
        help="The path to the eval dataset to use."
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=100, 
        help="The maximum total input sequence length after tokenization."
    )
    parser.add_argument(
        "--max_predictions_per_seq", type=int, default=20, 
        help="Maximum predictions per sequence_output."
    )
    parser.add_argument(
        "--label_map_file", type=str, default="ext/label_map.json",
        help="Path to the label map file. Either a JSON file ending with .json, or a text file has one tag per line."
    )
    parser.add_argument(
        "--lm_pretrained_path", type=str, default="./models/BDIRoBerta",
        help="Path to the pretrained language model."
    )

    # TRAINING PARAMS
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType,default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--batch_size", type=int, default=16, 
        help="The batch size value."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6, 
        help="The learning rate value."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=5, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--output_dir", type=str, default='./models/FELIXMaskedLM', 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--save_final_model", type=bool, default=True, 
        help="Save the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="A seed for reproducible training."
    )

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
