import random

from numpy.lib.utils import source
import masked_lm.constants as constants

from masked_lm.utils import *
# from tokenizer import FullTokenizer
from transformers import RobertaTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def get_number_of_masks(label):
    """Convert a tag to the number of MASK tokens it represents."""
    if '|' not in label:
        return 0

    return int(label.split('|')[1])


class InsertionConverter:
    def __init__(
        self,
        max_seq_length,
        max_predictions_per_seq,
        tokenizer: PreTrainedTokenizerBase=None,
        label_map_file='ext/label_map.json',
        lm_pretrained_path='models/BDIRoBerta',
        fall_back_mode='random'
    ) -> None:
        """Initializes an instance of InsertionConverter.

        Args:
        max_seq_length: Maximum length of source sequence.
        max_predictions_per_seq: Maximum number of MASK tokens.
        label_map: Dictionary to convert labels_ids to labels.
        vocab_file: Path to BERT vocabulary file.
        do_lower_case: text is lowercased.
        fall_back_mode: In the case no MASK tokens are generated:
                        'random': Randomly add MASK tokens.
                        'force':  Leave the output unchanged (not recommended). Otherwise 
                                return None and terminate early (saving computation time).
        """
        self._max_seq_length = max_seq_length
        self._max_predictions_per_seq = max_predictions_per_seq
        if tokenizer:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = RobertaTokenizer.from_pretrained(
                lm_pretrained_path, do_lower_case=False, cache_dir=None, use_fast=False)
        # self._label_map = read_label_map(label_map_file)
        # self._label_map_inverse = {v: k for k, v in self._label_map.items()}
        self._label_map = constants.TAGS2ID
        self._label_map_inverse = constants.ID2TAGS
        
        if fall_back_mode.lower() == 'random':
            self._do_random_mask = True
            self._do_lazy_generation = False
            self._do_augment_mask = False
        elif fall_back_mode.lower() == 'force':
            self._do_random_mask = False
            self._do_lazy_generation = False
            self._do_augment_mask = False
        elif fall_back_mode.lower() == 'augment':
            self._do_random_mask = False
            self._do_lazy_generation = False
            self._do_augment_mask = True
        else:
            self._do_random_mask = False
            self._do_lazy_generation = True
            self._do_augment_mask = False

    def _create_masked_source(
        self,
        source_tokens,
        labels,
        source_indexes,
        target_tokens
    ):
        """Realizes source_tokens & adds deleted to source_tokens and target_tokens.

        Args:
            source_tokens: List of source tokens.
            labels: List of label IDs, which correspond to a list of labels (KEEP,
                DELETE, MASK|1, MASK|2...).
            source_indexes: List of next tokens (see pointing converter for more
                details) (ordered by source tokens)
            target_tokens: Optional list of target tokens. Only provided when
                constructing training examples.

        Returns:
            masked_tokens: The source input for the insertion model, including MASK
                tokens and bracketed deleted tokens.
            target_tokens: The target tokens for the insertion model, where mask
                tokens are replaced with the actual token, also includes bracketed
                deleted tokens.
        """

        current_index = 0
        masked_tokens = []

        kept_tokens = set([0])
        for _ in range(len(source_tokens)):
            current_index = source_indexes[current_index]
            kept_tokens.add(current_index)
            # Token is deleted.
            if current_index == 0:
                break
        current_index = 0
        for _ in range(len(source_tokens)):
            source_token = source_tokens[current_index]
            deleted_tokens = []
            # Looking forward finding all deleted tokens.
            for i in range(current_index + 1, len(source_tokens)):
                ## If not a deleted token.
                if i in kept_tokens:
                    break

                deleted_tokens.append(source_tokens[i])

            # Add deleted tokens to masked_tokens and target_tokens.
            masked_tokens.append(source_token)
            # number_of_masks specifies the number MASKED tokens which
            # are added to masked_tokens.
            number_of_masks = get_number_of_masks(
                self._label_map_inverse[labels[current_index]])
            for _ in range(number_of_masks):
                masked_tokens.append(constants.MASK)
            if deleted_tokens:
                masked_tokens_length = len(masked_tokens)
                bracketed_deleted_tokens = ([constants.DELETE_SPAN_START] +
                                            deleted_tokens +
                                            [constants.DELETE_SPAN_END])
                target_tokens = (
                    target_tokens[:masked_tokens_length] + bracketed_deleted_tokens +
                    target_tokens[masked_tokens_length:])
                masked_tokens += bracketed_deleted_tokens

            current_index = source_indexes[current_index]
            if current_index == 0:
                break

        return masked_tokens, target_tokens


    def create_insertion_example(
        self,
        source_tokens,
        labels,
        source_indexes,
        target_tokens
    ):
        """Creates training/test features for insertion model.

        Args:
            source_tokens: List of source tokens.
            labels: List of label IDs, which correspond to a list of labels (KEEP,
                DELETE, MASK|1, MASK|2...).
            source_indexes: List of next tokens (see pointing converter for more
                details) (ordered by source tokens).
            target_tokens: List of target tokens.

        Returns:
            A dictionary of features needed by the insertion model.
        """

        masked_tokens, target_tokens = self._create_masked_source(
            source_tokens, 
            labels,
            source_indexes,
            target_tokens
        )
        # print(f"\nSOURCE TOKENS: {source_tokens}")
        # print(f"MASKED TOKENS: {masked_tokens}")
        # print(f"TARGET TOKENS: {target_tokens}")

        unused1_ids = [i for i, v in enumerate(masked_tokens) if v == constants.DELETE_SPAN_START]
        unused2_ids = [i for i, v in enumerate(masked_tokens) if v == constants.DELETE_SPAN_END]

        for i in range(len(unused1_ids)):
            del masked_tokens[unused1_ids[i]: unused2_ids[i]+1]
            del target_tokens[unused1_ids[i]: unused2_ids[i]+1]

        if target_tokens and constants.MASK not in masked_tokens:
            return None

        # if target_tokens and constants.MASK not in masked_tokens:
        #     if self._do_lazy_generation:
        #         return None
        #     else:   # Generate random MASK
        #         # Don't mask the start or end token.
        #         indexes  = list(range(1, len(masked_tokens) - 1))
        #         random.shuffle(indexes)
        #         # Limit MASK to ~10% of the source tokens.
        #         indexes = indexes[: int(len(masked_tokens) * 0.1)]
        #         for index in indexes:
        #             masked_tokens[index] = constants.MASK

        #     return [build_feed_dict(
        #         masked_tokens, 
        #         self._tokenizer, 
        #         target_tokens, 
        #         self._max_seq_length, 
        #         self._max_predictions_per_seq
        #     )]

        # elif target_tokens and constants.MASK in masked_tokens:
        #     if self._do_augment_mask:
        #         examples = [
        #             build_feed_dict(
        #                 masked_tokens,
        #                 self._tokenizer,
        #                 target_tokens,
        #                 self._max_seq_length,
        #                 self._max_predictions_per_seq
        #             )
        #         ]

        #         assert (
        #             len(masked_tokens) == len(target_tokens), 
        #             f"Lengths `masked_tokens={len(masked_tokens)}` and `target_tokens={len(target_tokens)}"
        #         )

        #         # Don't mask the start or end or mask token.
        #         indexes  = list(range(1, len(target_tokens) - 1))
        #         random.shuffle(indexes)
        #         # Limit MASK to ~10% of the source tokens.
        #         indexes = indexes[: int(len(target_tokens) * 0.1)]
        #         example_tokens = np.copy(target_tokens)
        #         for index in indexes:
        #             example_tokens[index] = constants.MASK

        #         examples.append(
        #             build_feed_dict(
        #                 example_tokens.tolist(),
        #                 self._tokenizer,
        #                 target_tokens,
        #                 self._max_seq_length,
        #                 self._max_predictions_per_seq
        #             )
        #         )

        #         return examples

        return [
            build_feed_dict(
                masked_tokens, 
                self._tokenizer, 
                target_tokens, 
                self._max_seq_length, 
                self._max_predictions_per_seq
            )]
        # return [
        #     build_feed_dict_without_mask(
        #         masked_tokens, 
        #         self._tokenizer, 
        #         target_tokens, 
        #         self._max_seq_length, 
        #         self._max_predictions_per_seq
        #     )]
