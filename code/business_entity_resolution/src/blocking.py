from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    from .preprocess import get_name_tokens, get_clean_name, get_addr_tokens, get_digits
except ImportError:
    from preprocess import get_name_tokens, get_clean_name, get_addr_tokens, get_digits

class BlockingIndex:
    """High-recall, ultra-compact candidate generation blocking index partitioned by country."""

    def __init__(self, max_postings: int = 50):
        self.max_postings = max_postings
        self.records_db: Dict[str, Tuple[str, str, Set[str], str, Set[str], Set[str]]] = {}
        self.clean_name_index = defaultdict(list)
        self.rare_token_index = defaultdict(list)
        self.digit_index = defaultdict(list)

    def add_record(self, eid: str, name: str, addr: str):
        """Preprocess and index a record from Source 2 or Source 3."""
        n_toks = get_name_tokens(name)
        c_name = " ".join(sorted(n_toks))
        a_toks = set(get_addr_tokens(addr))
        digs = get_digits(name + " " + addr)

        self.records_db[eid] = (name, addr, set(n_toks), c_name, a_toks, digs)

        if c_name:
            self.clean_name_index[c_name].append(eid)
        for t in set(n_toks):
            if len(t) >= 4:
                self.rare_token_index[t].append(eid)
        for d in digs:
            self.digit_index[d].append(eid)

    def filter_postings(self):
        """Prune broad posting lists to ensure candidate sets remain compact and high-precision."""
        self.rare_token_index = {
            k: v for k, v in self.rare_token_index.items() if len(v) <= self.max_postings
        }
        self.digit_index = {
            k: v for k, v in self.digit_index.items() if len(v) <= self.max_postings
        }

    def generate_candidates(self, s1_name: str, s1_addr: str) -> Set[str]:
        """Generate high-probability candidate matches for a Source 1 entity."""
        s1_ntok = set(get_name_tokens(s1_name))
        s1_cname = " ".join(sorted(s1_ntok))
        s1_atok = set(get_addr_tokens(s1_addr))
        s1_dig = get_digits(s1_name + " " + s1_addr)

        candidates = set()

        # 1. Exact clean canonical name
        if s1_cname and s1_cname in self.clean_name_index:
            candidates.update(self.clean_name_index[s1_cname][:20])

        # 2. Rare/informative name tokens
        for t in s1_ntok:
            if t in self.rare_token_index:
                candidates.update(self.rare_token_index[t])

        # 3. Address digit match with address token concordance
        for d in s1_dig:
            if d in self.digit_index:
                for cid in self.digit_index[d]:
                    _, _, _, _, catok, _ = self.records_db[cid]
                    if len(s1_atok & catok) >= 2:
                        candidates.add(cid)

        return candidates
