# -*- coding: utf-8 -*-

import math
import collections


__all__ = [
    "compute_bleu",
    "bleu_corpus_level",
    "bleu_sentence_level",
    "BleuScore",
]

# Hold the result of bleu_corpus_level().
BleuScore = collections.namedtuple('BleuScore', ['bleu', 'geo_mean', 'precisions', 'brevity_penalty'])

DEFAULT_MAX_ORDER = 4


def _get_ngrams(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            # For ngram to be hashable.
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def bleu_sentence_level(hypothesis, references, max_order=None, smooth=False):
    return bleu_corpus_level([hypothesis], [references], max_order, smooth)


def bleu_corpus_level(hypothesis_corpus, reference_corpus, max_order=None, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
            Note: This is a nested list of references. Each translation can have one or more references.
        hypothesis_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        4-Tuple with the BLEU score, geometric mean, n-gram precisions and brevity penalty.
    """

    # The use of & and | operators of Counter to implement
    # max_ref_count and clipped_by_max_ref_count, which underlies the modified n-grams count,
    # is a very smart idea.

    if len(hypothesis_corpus) != len(reference_corpus):
        raise ValueError("""
        You passed a translation_corpus of len %d and a reference_corpus of len %d.
        Their lens don't match.
        Perhaps some of them is not a *corpus*?
        """ % (len(hypothesis_corpus), len(reference_corpus)))

    max_order = max_order or DEFAULT_MAX_ORDER
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, hypothesis_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            # The | operator computes the maximum reference count as in the original paper.
            # In fact, for any instance of n-grams, we takes its max count among all references.
            # For example, ref1 is "the", ref2 is "the the", we are using n=1 (unigram),
            # then the max reference count for "the" will be 2.
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)

        # The & operator does the clipping as in the original paper.
        # It ensures that the counts in overlap does not exceed that in the merged counts.
        # The clipping prevents meaningless translation consisting of many repeated words being overestimated,
        # like "the the the..." against "the cat sat on the mat".
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]

        # Compute normalizer or dividend of the precisions.
        # This computes the counts of all n-grams ranging from 1 to max_order in a translation,
        # stored in `possible_matches_by_order[i]`.
        # This term serves as the normalizer or dividend of the modified-ngrams-precision.
        # The for loop below defines a simple function, see test_ngram_count.py for its behaviours.
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    # Compute the modified n-grams precision.
    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    # Note the rough behaviour here. See README.md.
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    # Compute the brevity penalty or BP for short.
    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return BleuScore(bleu=bleu, geo_mean=geo_mean, precisions=precisions, brevity_penalty=bp)


# Compatible
compute_bleu = bleu_corpus_level


if __name__ == '__main__':
    # TODO: test function calculate bleu score
    decoded = 'It is to insure the troops forever hearing the activity guidebook that party direct .'
    target = ['It is a guide to action that ensures that the military will forever heed Party commands .']

    bleu_score = bleu_sentence_level(hypothesis=decoded, references=target)

    print(f"Bleu Score w/o smooth      : {bleu_score}.")

    bleu_score = bleu_sentence_level(hypothesis=decoded, references=target, smooth=True)

    print(f"Bleu Score w smooth        : {bleu_score}.")


    # TODO: Compare with NLTK implementation
    import nltk.translate.bleu_score as blue_nltk

    blue_score = blue_nltk.sentence_bleu(target, decoded, smoothing_function=blue_nltk.SmoothingFunction().method1)
    print(f"Bleu Score of NLTK w smooth: {bleu_score}.")
