# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import os
import sys
import json
import time
import torch
import argparse
import itertools
import numpy as np
import underthesea
import pandas as pd
import tagging.constants as constants

from tqdm import tqdm
from tagging.tagging_model import FelixTagger
from transformers import RobertaTokenizer, AutoModelWithLMHead

# replacement = {
#     "lúc nào?": ["năm nào?", "ngày tháng năm nào?", "ngày nào?", "tháng nào?"],
#     "?": ["tỉnh nào?", "thành phố nào?", "quận nào?", "huyện nào?", "xã nào?", "nơi nào?", "chỗ nào?", "khu nào?", "thị trấn nào?", "phường nào?"],
# }

# replacement = {
#     key: sorted(v, key=lambda x: len(x), reverse=True) for key, v in replacement.items()
# }


# def preprocess_input(sentence):
#     for key, rep in replacement.items():
#         for r in rep:
#             sentence = sentence.replace(r, key)
#     return sentence


def insertion(model, tokenizer, sequence, device):

    start = time.time()
    input_ids = tokenizer.encode(sequence, return_tensors="pt").to(device)
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    token_logits = model(input_ids)[0]

    mask_token_logits = token_logits[0, mask_token_index, :]
    mask_token_logits = torch.softmax(mask_token_logits, dim=1)
    ids = torch.argmax(mask_token_logits, dim=1)

    ids = ids.cpu().tolist()
    sequence = sequence.split()
    for id in ids:
        idx = sequence.index(tokenizer.mask_token)
        sequence[idx] = tokenizer.decode([id]).strip()

    sequence = " ".join(sequence)
    # print(round(time.time() - start, 5) * 1000, "ms")
    return sequence


def bpe_tokenizer(words, tokenizer, max_word_length, max_subword_length):
    token_tmp = [tokenizer.bos_token] + words + [tokenizer.eos_token]

    attention_mask_words = np.ones(len(token_tmp))
    attention_mask_words = attention_mask_words[:max_word_length]
    attention_mask_words = np.hstack(
        [attention_mask_words, np.zeros(max_word_length - len(attention_mask_words))]
    )

    sub_words = [
        tokenizer.encode(token, add_special_tokens=False) for token in token_tmp
    ]
    sub_words = sub_words[:max_subword_length]

    word_matrix = np.zeros((max_word_length, max_subword_length))

    j = 0
    for i, tks in enumerate(sub_words):
        if tks[0] == tokenizer.pad_token_id:
            break
        for _ in tks:
            word_matrix[i, j] = 1
            j += 1
    sub_word_ids = list(itertools.chain.from_iterable(sub_words))
    sub_word_ids.extend(
        [tokenizer.pad_token_id] * (max_subword_length - len(sub_word_ids))
    )  # <pad> index
    attention_mask = np.ones(len(sub_word_ids))
    attention_mask[np.array(sub_word_ids) == tokenizer.pad_token_id] = 0
    return sub_word_ids, attention_mask, attention_mask_words, word_matrix


def ner_extract(text, model, tokenizer, devide="cuda", debug=False):
    model.eval()
    # from unicodedata import normalize as nl
    # text = nl('NFKC', text)
    # text = preprocess_input(text)

    words = " ".join(underthesea.word_tokenize(text)).split()
    words = [tokenizer.bos_token] + words + [tokenizer.eos_token]
    # print(f"{words} {len(words)}")

    # print("\nWord tokenized:", words)

    len_seq = len(words)

    with torch.no_grad():
        sub_word_ids, attention_mask, attention_mask_words, word_matrix = bpe_tokenizer(
            words, tokenizer, 128, 192
        )
        sub_word_ids = torch.tensor(sub_word_ids).unsqueeze(dim=0).to(devide)
        attention_mask = torch.tensor(attention_mask).unsqueeze(dim=0).to(devide)
        attention_mask_words = (
            torch.tensor(attention_mask_words).unsqueeze(dim=0).to(devide)
        )
        word_matrix = (
            torch.tensor(word_matrix, dtype=torch.float32).unsqueeze(dim=0).to(devide)
        )
        inputs = (sub_word_ids, word_matrix, attention_mask, attention_mask_words)

        # start = time.time()
        tag_logits, point_logits = model(inputs)
        # print("output model", time.time() - start)
    tag_logits = torch.argmax(tag_logits, dim=-1)[0]
    tag_logits = tag_logits.detach().cpu().numpy()
    # print(tag_logits[:len_seq])
    tag_outputs = [constants.ID2TAGS[i] for i in tag_logits]
    # print("\nTags:", tag_outputs[:len_seq])

    new_tokens = []
    for i, (w, lb) in enumerate(zip(words, tag_outputs)):
        if lb in constants.ID2TAGS:
            if not lb.startswith("KEEP|"):
                new_tokens.append(w)
            else:
                num_mask = int(lb.split("|")[-1])
                postfix = " ".join([tokenizer.mask_token] * num_mask)
                new_tokens.append(f"{w} {postfix}")
        else:
            new_tokens.append("")

    # print("\nToken tagging:", new_tokens)

    mat = point_logits.detach().cpu().numpy()
    if debug:
        with open("pointer.txt", "wb") as f:
            for line in mat:
                np.savetxt(f, line, fmt="%.2f")

    point_loop = np.array([], dtype=np.int64)
    mat = mat[0]
    while len(point_loop) < len_seq:
        line = mat[point_loop[-1] if point_loop.size > 0 else 0]
        if point_loop.size > 0:
            line[point_loop] = -np.inf
        point_loop = np.append(point_loop, np.argmax(line[:len_seq]))

    # print("\nPointer index", len(point_loop), point_loop)

    pointer_s = []
    for idx in point_loop:
        try:
            if idx == 0 or new_tokens[idx] == tokenizer.eos_token:
                break
        except:
            print(idx)
            print(text)
        pointer_s.append(new_tokens[idx].strip())

    pointer_s = " ".join(pointer_s)

    return pointer_s


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, type=str)

    args_main = parser.parse_args()

    model_path = "models/tagging" #if len(sys.argv) < 2 else sys.argv[1]
    pretrained_path = "models/BDIRoBerta"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(os.path.join(model_path, "args.json")) as f:
        args = json.load(f)

    model_insertion = AutoModelWithLMHead.from_pretrained(pretrained_path).to(device)
    model = FelixTagger(
        model_name=pretrained_path,
        device=device,
        num_classes=len(constants.ID2TAGS),
        is_training=False,
        position_embedding_dim=args["position_embedding_dim"],
        query_dim=args["query_dim"],
    )
    model.load_state_dict(
        torch.load(
            os.path.join(model_path, "best_model_correct_tagging.pt"),
            map_location=torch.device(device),
        )
    )
    model.to(device)
    model.eval()
    model_insertion.to(device)
    model_insertion.eval()
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
    print(f"Model {model_path} loading is done!")

    if args_main.file is None:
        while True:
            text = input("Enter text: ").strip().lower()
            # text = "cửa khẩu nào thuộc tỉnh điện biên ? tây trạng"
            if not text:
                exit()
            start_time = time.time()
            seq_out = ner_extract(text, model, tokenizer, device)
            # print(ners)
            print(round((time.time() - start_time) * 1000, 2), "ms")
            if tokenizer.mask_token in seq_out:
                seq_out = insertion(model_insertion, tokenizer, seq_out, device)
            print("\nFULL ANSWER:", seq_out)
            print("*" * 50)
    else:
        df = pd.read_csv(args_main.file, encoding='utf-8')
        gen_strs = []
        for i in tqdm(range(len(df)), desc="Predicting"):
            comp_text = df['complex_text'][i]
            seq_out = ner_extract(comp_text, model, tokenizer, device)
            if tokenizer.mask_token in seq_out:
                seq_out = insertion(model_insertion, tokenizer, seq_out, device)

            gen_strs.append(seq_out)

        df['gen_text'] = gen_strs

        df.to_csv('./prediction.csv', encoding='utf-8', index=False)
