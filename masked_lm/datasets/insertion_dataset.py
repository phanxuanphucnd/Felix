# -*- coding: utf-8 -*-

import torch
import frozendict
import pandas as pd
from torch._C import dtype
import masked_lm.constants as constants

from tqdm import tqdm
from unicodedata import normalize
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from masked_lm.utils import read_label_map
from masked_lm.pointing_converter import PointingConverter
from masked_lm.insertion_converter import InsertionConverter

class InsertionDataset(Dataset):
    def __init__(
        self, 
        mode: str='train',
        data_path: str=None,
        tokenizer: PreTrainedTokenizerBase=None,
        lm_pretrained_path: str='models/BDIRoBerta',
        label_map_file: str='ext/label_map.json',
        point_converter: PointingConverter=None,
        insertion_converter: InsertionConverter=None,
        use_open_vocab: bool=True,
        max_seq_length: int=128,
        do_lower_case: bool=True
    ) -> None:
        super(InsertionDataset, self).__init__()
        self.mlm_probability = 0.15
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.point_converter = point_converter
        self.insertion_converter = insertion_converter
        self._use_open_vocab = use_open_vocab
        self.label_map = read_label_map(label_map_file)
        
        inversed_label_map = {}
        for label, label_id in self.label_map.items():
            if label_id in inversed_label_map:
                raise ValueError(f"Multiple labels with the same ID: {label_id}.")
            inversed_label_map[label_id] = label
        
        self.inversed_label_map = frozendict.frozendict(inversed_label_map)

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                lm_pretrained_path, 
                do_lower_case=False, 
                cache_dir=None, 
                use_fast=False
            )

        if data_path.endswith('.csv'):
            data_df = pd.read_csv(data_path, encoding='utf-8')

        self.examples = []
        
        src_texts = data_df['complex_text'].tolist()
        tar_texts = data_df['simple_text'].tolist()

        for src_text, tar_text in tqdm(zip(src_texts, tar_texts), desc=f"Create {mode.upper()} Dataset"):
            # print("----------------------")
            # print(f"SOURCE TEXT: {src_text}")
            # print(f"TARGET TEXT: {tar_text}")

            src_tokens = self.tokenizer.tokenize(src_text)
            src_tokens = self._truncate_list(src_tokens)
            src_tokens = [constants.CLS] + src_tokens + [constants.SEP]
            try:
                tar_tokens = self.tokenizer.tokenize(tar_text)
            except:
                raise ValueError(f"ERROR in src: {src_text} | tar: {tar_text}")
            tar_tokens = self._truncate_list(tar_tokens)
            tar_tokens = [constants.CLS] + tar_tokens + [constants.SEP]

            points = self.point_converter.compute_points(
                ' '.join(src_tokens).split(),
                ' '.join(tar_tokens).split()
            )
            # print(f"POINTS: {[str(point) for point in points]}")
            if points:
                labels = [t.added_phrase for t in points]
                point_indexes = [t.point_index for t in points]
                point_indexes_set = set(point_indexes)

                try:
                    new_labels = []
                    for i, added_phrase in enumerate(labels):
                        if i not in point_indexes_set:
                            new_labels.append(self.label_map[constants.DELETE])
                        elif not added_phrase:
                            new_labels.append(self.label_map[constants.KEEP])
                        else:
                            if self._use_open_vocab:
                                new_labels.append(self.label_map['KEEP|' + str(len(added_phrase.split()))])
                            else:
                                new_labels.append(self.label_map['KEEP|' + str(added_phrase)])

                        labels = new_labels

                    # print(f"new_labels: {new_labels}")

                    if labels:
                        insertion_example = self.insertion_converter.create_insertion_example(
                            src_tokens, labels, point_indexes, tar_tokens
                        )
                        if insertion_example:
                            self.examples.extend(insertion_example)

                except KeyError as e:
                    pass

    def _truncate_list(self, x):
        return x[: self.max_seq_length-2]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def _collate_func(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors='pt', padding=True)
        if self.mlm_probability > 0:
            batch["input_ids"], batch["labels"] = self.random_mask_tokens(
                batch["input_ids"], 
                batch["labels"]
            )

        return batch

    def random_mask_tokens(self, inputs, labels):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        special_tokens_mask = []
        mask_token_id = self.tokenizer.mask_token_id
        all_special_ids = self.tokenizer.all_special_ids

        for val in inputs.tolist():
            if mask_token_id in val:
                tmp_mask = [1]*len(val)
            else:
                tmp_mask = [1 if token in all_special_ids else 0 for token in val] 
            
            special_tokens_mask.append(tmp_mask)

        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[masked_indices] = inputs[masked_indices]

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)

        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
