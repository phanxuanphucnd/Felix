import json
import torch
import numpy as np
import masked_lm.constants as constants


def get_token_list(text):
    """Returns a list of tokens.
    
    This function expects that the tokens in the text are separated by space
    character(s). Example: "ca n't , touch". This is the case at least for the
    public DiscoFuse and WikiSplit datasets.

    Args:
        text: String to be split into tokens.
    """
    return text.split()


def read_label_map(path, use_str_keys=False):
    """Returns label map read from the given path.

    Args:
        path: Path to the label map file.
        use_str_keys: Whether to use label strings as keys instead of
        (base tag, num insertions) tuple keys. The latter is only used by
        FelixInsert.
    """
    label_map = {}
    with open(path, 'r+', encoding='utf-8') as f:
        if path.endswith(".json"):
            label_map = json.load(f)
        else:
            for tag in f:
                tag = tag.strip()
                # Empty lines are skipped.
                if tag:
                    if tag in label_map:
                        raise ValueError("Duplicate label in label_map: {}".format(tag))
                    label_map[tag] = len(label_map)

    return label_map

def build_feed_dict(
    tokens,
    tokenizer,
    target_tokens=None,
    max_seq_length=128,
    max_predictions_per_seq=20
):
    """Returns a dictionary used for predicting/training the insertion model.

    Converts a list of source tokens, containing masks, to a dictionary of
    features used by a Insertion model. If a target sequence is provided, then the
    targets for the MASKs are set.

    Args:
        tokens: Input tokens, with mask tokens.
        tokenizer: Tokenizer used to convert tokens to IDs.
        target_tokens: (Optional) The targets of the mask tokens.
        max_seq_length: Maximum sequence length.
        max_predictions_per_seq: Maximum number of mask tokens.

    Returns:
        Dictionary with model features or None if `len(tokens) > max_seq_length` or
        if the number of MASKs is larger than `max_predictions_per_seq`.
    """
    mask_position = []
    mask_target_id = []
    for idx, token in enumerate(tokens):
        if token != constants.MASK:
            continue

        mask_position.append(idx)
        if target_tokens:
            mask_target_id += tokenizer.convert_tokens_to_ids([target_tokens[idx]])
        else:
            mask_target_id.append(0)

    input_mask = [1] * len(tokens)
    while len(tokens) < max_seq_length:
        tokens.append(tokenizer.pad_token)
        input_mask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    
    if len(input_ids) > max_seq_length:
        return None

    assert len(input_ids) == max_seq_length, "len(input_ids) = {}".format(len(input_ids))
    assert len(input_mask) == max_seq_length, "len(input_mask) = {}".format(len(input_mask))
    
    if len(mask_position) > max_predictions_per_seq:
        return None
    
    target_ids = np.array(target_ids)
    target_ids[mask_position] = mask_target_id
    labels = np.full(np.shape(target_ids), -100)
    labels[mask_position] = target_ids[mask_position]
    labels = labels.tolist()
    labels.extend([-100]*(len(input_ids) - len(labels)))
    feed_dict = {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(input_mask),
        'labels': torch.tensor(labels)
    }

    return feed_dict

def build_feed_dict_without_mask(
    tokens,
    tokenizer,
    target_tokens=None,
    max_seq_length=128,
    max_predictions_per_seq=20
):
    mask_position = []
    mask_target_id = []
    for idx, token in enumerate(tokens):
        if token != constants.MASK:
            continue

        mask_position.append(idx)
        if target_tokens:
            mask_target_id += tokenizer.convert_tokens_to_ids([target_tokens[idx]])

    input_mask = [1] * len(tokens)
    while len(tokens) < max_seq_length:
        tokens.append(tokenizer.pad_token)
        input_mask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    
    if len(input_ids) > max_seq_length:
        return None

    assert len(input_ids) == max_seq_length, "len(input_ids) = {}".format(len(input_ids))
    assert len(input_mask) == max_seq_length, "len(input_mask) = {}".format(len(input_mask))
    
    if len(mask_position) > max_predictions_per_seq:
        return None
    
    target_ids = np.array(target_ids)
    target_ids[mask_position] = mask_target_id
    labels = np.full(np.shape(target_ids), -100)
    labels[mask_position] = target_ids[mask_position]
    labels = labels.tolist()
    labels.extend([-100]*(len(input_ids) - len(labels)))
    feed_dict = {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(input_mask),
        'labels': torch.tensor(labels)
    }

    return feed_dict
