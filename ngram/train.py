# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import pickle
from pathlib import Path
from utils import load_data
from ngram_lm import LanguageModel

data_path = Path('./data/rewrite/')
train, test = load_data(data_path)

n = 5
laplace = 0.01

print("Loading {}-gram model...".format(n))
lm = LanguageModel(train, n, laplace=laplace)
print("Vocabulary size: {}".format(len(lm.vocab)))

with open('./ngram.pkl', 'wb') as f:
    pickle.dump(lm, f)
