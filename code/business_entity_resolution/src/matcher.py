from typing import Set, Tuple, List

try:
    from .preprocess import clean_text, get_name_tokens, get_addr_tokens, get_digits
except ImportError:
    from preprocess import clean_text, get_name_tokens, get_addr_tokens, get_digits

def jaccard(s1: Set, s2: Set) -> float:
    if not s1 or not s2:
        return 0.0
    u = s1 | s2
    return len(s1 & s2) / len(u) if u else 0.0

def score_and_match(
    s1_name: str,
    s1_addr: str,
    candidates: Set[str],
    records_db: dict,
    max_candidates: int = 8
) -> Tuple[List[str], List[str]]:
    """
    Score candidate pairs, rank them, and select high-precision entity matches.
    Guarantees:
      - Returned matched_ids is strictly a subset of returned top_candidates.
      - Number of candidates <= max_candidates (minimizing candidate bloat).
    """
    if not candidates:
        return [], []

    s1_ntok = set(get_name_tokens(s1_name))
    s1_cname = " ".join(sorted(s1_ntok))
    s1_atok = set(get_addr_tokens(s1_addr))
    s1_dig = get_digits(s1_name + " " + s1_addr)
    s1_nclean = clean_text(s1_name).replace(" ", "")

    scored = []

    for cid in candidates:
        t_name, t_addr, t_ntok, t_cname, t_atok, t_dig = records_db[cid]

        exact_name = (s1_cname == t_cname) and bool(s1_cname)
        name_jacc = jaccard(s1_ntok, t_ntok)
        name_overlap = len(s1_ntok & t_ntok)

        # Domain name / hashtag / substring check
        t_clean = clean_text(t_name).replace(" ", "")
        sub_match = any(len(tok) >= 4 and tok in t_clean for tok in s1_ntok)

        addr_jacc = jaccard(s1_atok, t_atok)
        addr_overlap = len(s1_atok & t_atok)
        dig_overlap = len(s1_dig & t_dig)

        # High-Precision Classification Rules (Optimized for F_0.5 > 0.99 Precision)
        is_match = False

        # Rule 1: Exact canonical clean name + some address concordance
        if exact_name and (addr_overlap >= 1 or dig_overlap >= 1 or not s1_atok or not t_atok):
            is_match = True
        # Rule 2: Strong name overlap + address agreement
        elif (name_jacc >= 0.5 or name_overlap >= 2) and (addr_overlap >= 2 or (dig_overlap >= 1 and addr_overlap >= 1)):
            is_match = True
        # Rule 3: Domain / hashtag / company acronym in name + address agreement
        elif sub_match and (addr_overlap >= 2 or (dig_overlap >= 1 and addr_overlap >= 1)):
            is_match = True
        # Rule 4: Deep address agreement (DBA / parent entity / regional transliteration)
        elif dig_overlap >= 1 and addr_overlap >= 3:
            is_match = True
        # Rule 5: Identical name when addresses are sparse/missing
        elif name_jacc >= 0.85 and (addr_overlap >= 1 or not s1_atok or not t_atok):
            is_match = True

        score = (2.0 if is_match else 0.0) + name_jacc * 0.6 + addr_jacc * 0.4
        scored.append((cid, score, is_match))

    # Sort descending by composite score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Retain top K candidates
    top_candidates = [cid for cid, s, m in scored[:max_candidates]]
    # Matches must be drawn strictly from top candidates
    matched_ids = [cid for cid, s, m in scored[:max_candidates] if m]

    return top_candidates, matched_ids
