import os
import torch

from torch.utils.data import Dataset
from tokenizers.processors import BertProcessing
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class LMDataset(Dataset):
    def __init__(
        self, 
        root: str='./data', 
        mode: str='train', 
        tokenizer: PreTrainedTokenizerBase=None,
        vocab_file: str='models/BDITokenizer/vocab.json',
        merges_file: str='models/BDITokenizer/merges.txt',
        max_length: int=128,
        lower_case: bool=True
    ) -> None:
        super(LMDataset, self).__init__()
        self.examples = []

        print(f"MODE: {mode.upper()}")
        for i, file in enumerate(os.listdir(root)):
            if mode in file and file.endswith('.txt'):
                src_file = f"{root}/{file}"
                print(f"{i}: ", src_file)
                with open(src_file, 'r+', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                
                if lower_case:
                    lines = [line.strip().lower() for line in lines]

                lines = list(filter(None, lines))
                if not tokenizer:
                    tokenizer = ByteLevelBPETokenizer(
                        vocab_file,
                        merges_file
                    )
                    tokenizer._tokenizer.post_processor = BertProcessing(
                        ('</s>', tokenizer.token_to_id('</s>')),
                        ('<s>', tokenizer.token_to_id('<s>')),
                    )
                    tokenizer.enable_truncation(max_length=max_length)
                    
                    self.examples += [x.ids for x in tokenizer.encode_batch(lines)]
                else:
                    batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=max_length)
                    self.examples.extend([{"input_ids": torch.tensor(e, dtype=torch.long)} for e in batch_encoding["input_ids"]])

                    del lines
                    del batch_encoding

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

