import re
import unicodedata

# Comprehensive Legal and Corporate Form Terms (English, Indian, French, European)
LEGAL_TERMS = {
    'llc', 'inc', 'corp', 'corporation', 'incorporated', 'ltd', 'limited',
    'pvt', 'private', 'pllc', 'co', 'company', 'services', 'enterprises',
    'group', 'holdings', 'industries', 'associates', 'solutions', 'management',
    'consulting', 'technologies', 'international', 'global',
    # French corporate forms
    'sarl', 'sa', 'sas', 'sasu', 'eurl', 'snc', 'sca', 'scop', 'gie', 'ste', 'societe',
    # Other common
    'gmbh', 'bv', 'nv', 'sp', 'spa', 'srl', 'sl', 'pty', 'ag'
}

# Common address words and stopwords (US, India, France)
ADDR_STOPWORDS = {
    'floor', 'fl', 'ground', 'near', 'road', 'rd', 'street', 'st', 'ave', 'avenue',
    'lane', 'dr', 'drive', 'blvd', 'boulevard', 'court', 'ct', 'suite', 'ste',
    'unit', 'apt', 'apartment', 'building', 'bldg', 'block', 'blk', 'shop', 'no',
    'plot', 'behind', 'opp', 'opposite', 'beside', 'city', 'state', 'null', 'c', 'o', 's',
    # French address terms
    'rue', 'bd', 'all', 'allee', 'imp', 'impasse', 'place', 'pl', 'chemin', 'cedex',
    'route', 'rte', 'voie', 'quai', 'cours', 'avenue', 'passage', 'residence'
}

def clean_text(text: str) -> str:
    """Normalize unicode, convert to lower case, and clean punctuation."""
    if not text:
        return ""
    # Strip diacritics / accents using NFKD
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[\(\)\[\]\{\}\<\>\"\'`]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_name_tokens(name: str):
    """Extract clean, meaningful tokens for business names, filtering legal forms."""
    clean = clean_text(name)
    return [t for t in clean.split() if t not in LEGAL_TERMS and len(t) > 1]

def get_clean_name(name: str) -> str:
    """Return canonical sorted name token string for exact canonical matching."""
    tokens = get_name_tokens(name)
    return " ".join(sorted(tokens))

def get_addr_tokens(addr: str):
    """Extract clean structural address tokens, filtering stopwords."""
    clean = clean_text(addr)
    return [t for t in clean.split() if t not in ADDR_STOPWORDS and len(t) > 1]

def get_digits(text: str):
    """Extract all digit sequences (street numbers, shop numbers, pincodes)."""
    return set(re.findall(r'\b\d+\b', text))
