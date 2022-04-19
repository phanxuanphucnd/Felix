# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

from transformers import RobertaConfig
from transformers import RobertaTokenizer

pretrained_path = 'models/BDIRoBerta'

config = RobertaConfig.from_pretrained(pretrained_path, cache_dir=None)
tokenizer = RobertaTokenizer.from_pretrained(pretrained_path, do_lower_case=False, cache_dir=None, use_fast=False)

tokens = tokenizer.tokenize('xin chào việt nam')
print(tokens)

print(tokenizer.convert_tokens_to_ids(tokens))