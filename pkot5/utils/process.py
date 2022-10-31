import copy
from typing import List


def fill_in_the_blank(rng, words: List[int], extra_ids: List[int]):
    mask_id = -1

    min_prob = 1 / (len(words) + 1)
    max_prob = 1 / 2
    inputs = copy.deepcopy(words)
    targets = words
    for i in range(len(words)):
        prob = rng.random()
        if min_prob < prob < max_prob:
            inputs[i] = mask_id
        else:
            targets[i] = mask_id

    def merge_mask(words_):
        mask_spans = []
        begin, end = None, None
        for i, w in enumerate(words_):
            if w == mask_id:
                if begin is None:
                    begin = i
                end = i + 1
            else:
                if end is not None:
                    mask_spans.append((begin, end))
                    begin, end = None, None
        if begin is not None and end is not None:
            mask_spans.append((begin, end))

        new_words_ = []
        last_offset = 0
        assert len(mask_spans) <= len(extra_ids), f"mask_spans={len(mask_spans)} is over length of extra_ids"
        for i, (begin, end) in enumerate(mask_spans):
            new_words_ += words_[last_offset:begin]
            new_words_.append(extra_ids[i])
            last_offset = end
        new_words_ += words_[last_offset:]

        return new_words_

    inputs = merge_mask(inputs)
    targets = merge_mask(targets)

    return {'inputs': inputs, 'targets': targets}
