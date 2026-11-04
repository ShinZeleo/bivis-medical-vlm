import re

def parse_yes_no(raw_output: str) -> str:
    """
    Extracts "YES" or "NO" robustly for both English and Indonesian responses.
    Ignores case, punctuation, and prefixes like "Answer: " or "Jawab: ".
    Returns "INVALID" if ambiguous or neither is found.
    """
    if not raw_output or not isinstance(raw_output, str):
        return "INVALID"
        
    text = raw_output.lower().strip()
    
    # Strip common prefixes/suffixes, remove punctuation
    text_clean = re.sub(r'[^a-z\s]', ' ', text)
    tokens = set(text_clean.split())
    
    yes_indicators = {'yes', 'ya', 'true', 'benar'}
    no_indicators = {'no', 'tidak', 'false', 'bukan'}
    
    has_yes = bool(tokens & yes_indicators)
    has_no = bool(tokens & no_indicators) or ('tidak ada' in text)
    
    if has_yes and not has_no:
        return "YES"
    elif has_no and not has_yes:
        return "NO"
    else:
        return "INVALID"
