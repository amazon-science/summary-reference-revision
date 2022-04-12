"""
Resolve overlap among finder candidates.
"""

from collections import namedtuple

CANDIDATE_FIELDS = ['start', 'end', 'match_text', 'regex', 'other']
Candidate = namedtuple('_Candidate', CANDIDATE_FIELDS)

# initialize all fields to None
Candidate.__new__.__defaults__ = (None,) * len(Candidate._fields)


###############################################################################
def has_overlap(a1, b1, a2, b2):
    """
    Determine if intervals [a1, b1) and [a2, b2) overlap at all.
    """

    assert a1 <= b1
    assert a2 <= b2

    if b2 <= a1:
        return False
    elif a2 >= b1:
        return False
    else:
        return True


###############################################################################
def remove_overlap(candidates):
    """
    Given a set of match candidates, resolve into nonoverlapping matches.
    Take the longest match at any given position.
    ASSUMES that the candidate list has been sorted by matching text length,
    from longest to shortest.
    """

    results = []
    overlaps = []
    indices = [i for i in range(len(candidates))]

    i = 0
    while i < len(indices):
        index_i = indices[i]
        start_i = candidates[index_i].start
        end_i = candidates[index_i].end
        len_i = end_i - start_i

        overlaps.append(i)
        candidate_index = index_i

        j = i + 1
        while j < len(indices):
            index_j = indices[j]
            start_j = candidates[index_j].start
            end_j = candidates[index_j].end
            len_j = end_j - start_j

            # does candidate[j] overlap candidate[i] at all
            if has_overlap(start_i, end_i, start_j, end_j):
                overlaps.append(j)
                # keep the longest match at any overlap region
                if len_j > len_i:
                    start_i = start_j
                    end_i = end_j
                    len_i = len_j
                    candidate_index = index_j
            j += 1

        results.append(candidates[candidate_index])

        # remove all overlaps
        new_indices = []
        for k in range(len(indices)):
            if k not in overlaps:
                new_indices.append(indices[k])
        indices = new_indices

        if 0 == len(indices):
            break

        # start over
        i = 0
        overlaps = []

    return results
