# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import time
import torch

from transformers import RobertaForMaskedLM, RobertaTokenizer


model_path = "models/FELIXMaskedLM"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForMaskedLM.from_pretrained(model_path).to(device)

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=model_path,
    tokenizer=model_path
)

while True:
    sequence = input("Enter input: ")
    sequence = sequence.replace("_", tokenizer.mask_token)
    result = fill_mask(sequence)
    from pprint import pprint
    pprint(result)


# while True:
#     sequence = input("Enter input: ")
#     sequence = sequence.replace("_", tokenizer.mask_token)

#     start = time.time()

#     input_ids = tokenizer.encode(sequence.lower(), return_tensors="pt").to(device)
    
#     mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
#     token_logits = model(input_ids)[0]
#     mask_token_logits = token_logits[0, mask_token_index, :]
#     mask_token_logits = torch.softmax(mask_token_logits, dim=1)
    
#     ids = torch.argmax(mask_token_logits, dim=1)
#     ids = ids.cpu().tolist()
    
#     sequence = sequence.split()

#     for id in ids:
#         idx = sequence.index(tokenizer.mask_token)
#         sequence[idx] = tokenizer.decode([id])
    
#     print(" ".join([i.strip() for i in sequence]))
        
#     print(round(time.time() - start, 5)*1000, "ms")
