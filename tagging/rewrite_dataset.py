import numpy as np
import torch
import itertools
import tagging.constants as constants


class RewriteDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer, max_word_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_word_length = max_word_length
        self.max_subword_length = int(self.max_word_length * 1.5)
        

    def __getitem__(self, idx):
        # for i, sample in enumerate(tqdm(self.samples, desc="Converting text data to tensor", bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}')):
        sample = self.samples[idx]
        words = [tk[0] for tk in sample]
        sub_word_ids, attention_mask, attention_mask_words, word_matrix = self.bpe_tokenizer(
            words)
        tag_labels = [constants.TAGS2ID[tk[1]] for tk in sample]
        # tag_labels = [constants.TAGS2ID['O']] + tag_labels + [constants.TAGS2ID['O']]
        tag_labels = tag_labels[:self.max_word_length]
        tag_labels += [constants.TAGS2ID["PAD"]] * \
            (self.max_word_length - len(tag_labels))
        # self.label_weight.update(Counter(tag_labels))

        point_labels = [int(tk[2]) for tk in sample]
        # point_labels = [1] + point_labels + [0] # CLS point to
        point_labels = point_labels[:self.max_word_length]
        point_labels += [-100] * (self.max_word_length - len(point_labels))

        item = {}
        item["input_ids"] = torch.tensor(sub_word_ids)
        item["input_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        item["input_mask_words"] = torch.tensor(
            attention_mask_words, dtype=torch.long)
        item["word_matrix"] = torch.tensor(
            word_matrix, dtype=torch.float32)
        item["tag_labels"] = torch.tensor(tag_labels, dtype=torch.long)
        item["point_labels"] = torch.tensor(point_labels, dtype=torch.long)

        # self.samples[i] = item
        return item #self.samples[idx]

    def bpe_tokenizer(self, words):
        # if 'phobert' in self.tokenizer.name_or_path:
        #     token_tmp = [self.tokenizer.bos_token] + words + [self.tokenizer.eos_token]
        # else:
        token_tmp = [self.tokenizer.cls_token] + \
            words + [self.tokenizer.cls_token]
        attention_mask_words = np.ones(len(token_tmp))
        attention_mask_words = attention_mask_words[:self.max_word_length]
        attention_mask_words = np.hstack([attention_mask_words, np.zeros(
            self.max_word_length - len(attention_mask_words))])

        sub_words = [
            self.tokenizer.encode(token, add_special_tokens=False)
            for token in token_tmp
        ]
        sub_words = sub_words[: self.max_word_length]

        word_matrix = np.zeros((self.max_word_length, self.max_subword_length))

        j = 0
        for i, tks in enumerate(sub_words):
            if tks[0] == self.tokenizer.pad_token_id:
                break
            for tk in tks:
                try:
                    word_matrix[i, j] = 1
                    j += 1
                except:
                    print(f"i={i} j={j}")
                    np.savetxt('text.txt', word_matrix,fmt='%.2f')
                    exit()

        sub_word_ids = list(itertools.chain.from_iterable(sub_words))
        sub_word_ids.extend(
            [self.tokenizer.pad_token_id] *
            (self.max_subword_length - len(sub_word_ids))
        )  # <pad> index
        attention_mask = np.ones(len(sub_word_ids))
        attention_mask[np.array(sub_word_ids) == self.tokenizer.pad_token_id] = 0
        return sub_word_ids, attention_mask, attention_mask_words, word_matrix

    def __len__(self):
        return len(self.samples)