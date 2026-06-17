from collections import defaultdict


def extract_phrase_pairs(src_tokens, tgt_tokens, alignment):
    """
    Implements the Och & Ney phrase extraction algorithm.

    Args:
        src_tokens: list[str]
        tgt_tokens: list[str]
        alignment: list[tuple[int, int]]  # (src_index, tgt_index)

    Returns:
        list of (src_start, src_end, tgt_start, tgt_end)
        inclusive indices
    """

    I = len(src_tokens)
    J = len(tgt_tokens)

    # Build alignment lookup tables
    aligned_to_src = defaultdict(set)
    aligned_to_tgt = defaultdict(set)

    for i, j in alignment:
        aligned_to_src[i].add(j)
        aligned_to_tgt[j].add(i)

    phrase_pairs = []
    # Loop over all possible source phrases
    for i1 in range(I):
        for i2 in range(i1, I):

            # Find minimal matching target span
            js = []
            for i in range(i1, i2 + 1):
                js.extend(aligned_to_src.get(i, []))

            if not js:
                continue  # no alignment points inside

            j1 = min(js)
            j2 = max(js)

            # Check consistency: no source word outside span
            # aligns to target word inside span
            consistent = True
            for j in range(j1, j2 + 1):
                for i in aligned_to_tgt.get(j, []):
                    if i < i1 or i > i2:
                        consistent = False
                        break
                if not consistent:
                    break

            if not consistent:
                continue

            # Expand target phrase to include unaligned words
            j_start = j1
            while j_start > 0 and len(aligned_to_tgt.get(j_start - 1, [])) == 0:
                j_start -= 1

            j_end = j2
            while j_end + 1 < J and len(aligned_to_tgt.get(j_end + 1, [])) == 0:
                j_end += 1

            # Add all expanded phrases
            for js_exp in range(j_start, j1 + 1):
                for je_exp in range(j2, j_end + 1):
                    phrase_pairs.append((i1, i2, js_exp, je_exp))

    return phrase_pairs
