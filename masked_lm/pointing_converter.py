import collections

import masked_lm.pointing as pointing

class PointingConverter:
    def __init__(self, phrase_vocabulary, do_lower_case=True) -> None:
        """Initializes an instance of PointingConverter.

        Args:
            phrase_vocabulary: Iterable of phrase vocabulary items (strings), if empty
                            we assume an unlimited vocabulary.
            do_lower_case: Should the phrase vocabulary be lower cased.
        """
        self._do_lower_case = do_lower_case
        self._phrase_vocabulary = set()

        for phrase in phrase_vocabulary:
            if do_lower_case:
                phrase = phrase.lower()
            # Remove the KEEP/DELETE flags for vocabulary phrases.
            if "|" in phrase:
                self._phrase_vocabulary.add(phrase.split("|")[1])
            else:
                self._phrase_vocabulary.add(phrase)

    def compute_points(self, src_tokens, tar_tokens):
        """Computes points needed for converting the source into the target.

        Args:
            source_tokens: Source tokens.
            target: Target text.

        Returns:
            List of pointing.Point objects. If the source couldn't be converted into
            the target via pointing, returns an empty list.
        """
        if self._do_lower_case:
            src_tokens = [x.lower() for x in src_tokens]
            tar_tokens = [x.lower() for x in tar_tokens]

        points = self._compute_points(src_tokens, tar_tokens)

        return points

    def _compute_points(self, src_tokens, tar_tokens):
        """Computes points needed for converting the source into the target.

        Args:
            src_tokens: List of source tokens.
            tar_tokens: List of target tokens.

        Returns:
            List of pointing.Pointing objects. If the source couldn't be converted
            into the target via pointing, returns an empty list.
        """
        src_tokens_indexes = collections.defaultdict(set)
        
        for i, src_token in enumerate(src_tokens):
            src_tokens_indexes[src_token].add(i)

        tar_points = {}
        last = 0
        token_buffer = ""

        def find_nearest(indexes, index):
            # In the case that two indexes are equally far apart
            # the lowest index is returned
            return min(indexes, key=lambda x: abs(x - index))

        for tar_token in tar_tokens[1:]:
            if (src_tokens_indexes[tar_token] and 
                (not token_buffer or not self._phrase_vocabulary or token_buffer in self._phrase_vocabulary)):
                # Maximum length expected of source_tokens_indexes[tar_token] is 512,
                # median length is 1.

                src_idx = find_nearest(src_tokens_indexes[tar_token], last)

                # We can only point to a token once
                src_tokens_indexes[tar_token].remove(src_idx)
                tar_points[last] = pointing.Point(src_idx, token_buffer)
                last = src_idx
                token_buffer = ""

            else:
                token_buffer = (token_buffer + " " + tar_token).strip()

        # Buffer needs to be empty at the end
        if token_buffer.strip():
            return []

        points = []
        for i in range(len(src_tokens)):
            # If a source token is not pointed to,
            # then it should point to the start of the sequence.
            if i not in tar_points:
                points.append(pointing.Point(0))
            else:
                points.append(tar_points[i])

        return points
