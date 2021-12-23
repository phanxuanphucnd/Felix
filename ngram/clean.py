import re

from tqdm import tqdm
from unicodedata import normalize

with open('./data/rewrite1.txt', 'r+', encoding='utf-8') as f:
    lines = f.readlines()

TRAIN_SIZE = int(0.96*len(lines))

wf_train = open('./data/rewrite/train.txt', 'w', encoding='utf-8')
wf_test = open('./data/rewrite/test.txt', 'w', encoding='utf-8')

for i, line in tqdm(enumerate(lines)):
    line = normalize('NFKC', line)
    line = re.sub(',{2,}', ' , ', line)
    line = re.sub(',', ' , ', line)
    line = re.sub(',\.', ' . ', line)
    line = re.sub('\.{2,}', ' . ', line)
    line = re.sub(r"(?<=[^0-9])\.", " . ", line)
    line = re.sub('[\"\'\?/\\~@:#$%^&*\(\)“”\-!=\{\}\[\]]', ' ', line)
    line = re.sub('\s{2,}', ' ', line)

    if i <= TRAIN_SIZE:
        if line.strip():
            wf_train.writelines(line.lower().strip() + '\n')
    else:
        if line.strip():
            wf_test.writelines(line.lower().strip() + '\n')

wf_train.close()
wf_test.close()
