from topeai_init_bert import init_ner_pipeline
from urllib.parse import urlparse
import pandas as pd
import math
import re

def identify_url_path(url: str, ner_pipeline=None) -> str:
    if ner_pipeline is None:
        ner_pipeline = init_ner_pipeline()

    path = extract_url_parts(url)
    replaced_text = replace_special_tokens(path)

    # Step 1: special tokens
    unique_tokens = set()
    slash_tokens = replaced_text.split('/')
    for token in slash_tokens:
        sub_tokens = re.findall(r'\S+', token)
        for t in sub_tokens:
            if t and t != "param":
                unique_tokens.add(t)

    random_like_tokens = set()
    for token in unique_tokens:
        entropy = shannon_entropy(token)
        if entropy > 3.3 and len(token) >= 8:
            random_like_tokens.add(token)

    # Step 2: final path
    tokens = replaced_text.split('/')
    new_tokens = []
    for token in tokens:
        if token == "param":
            new_tokens.append("param")
        else:
            sub_tokens = token.split(',')
            new_sub_tokens = []
            for t in sub_tokens:
                t_clean = t.strip()
                if t_clean == "param" or t_clean in random_like_tokens:
                    new_sub_tokens.append("param")
                else:
                    new_sub_tokens.append(t_clean)
            rebuilt_token = ', '.join(new_sub_tokens)
            new_tokens.append(rebuilt_token)

    return '/'.join(new_tokens)

def extract_url_parts(url):
    """Extracts path and query string from a URL (excluding domain)"""
    parsed = urlparse(url)
    path = parsed.path
    return path

def replace_special_tokens(text):
    """Replaces UUIDs, IDs, and DateTime strings with placeholders in the entire text"""
    datetime_patterns = [
        r"\d{1,2}[ ]?[A-Za-z]{3}[a-z]?[ ]?\d{4},[ ]?\d{2}:\d{2}",             # 14 Apr 2024, 08:30
        r"[A-Za-z]{3,9}[ ]\d{1,2},[ ]?\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",  # Apr 14, 2024 or Apr 14, 2024 08:30
        r"\d{2}[-/\.]\d{2}[-/\.]\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",        # 14-04-2024 or 14-04-2024 08:30
        r"\d{4}[-/\.]\d{2}[-/\.]\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",        # 2024-04-14 or 2024-04-14 08:30
        r"\d{1,2}[ ]?[A-Za-z]{3}[a-z]?[ ]?\d{4}[ ]?(?:[ T]\d{2}:\d{2})?",     # 14 Apr 2024 with optional time
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",                               # 2024-04-14T08:30:00
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",                              # 2024-04-14T08:30:00Z
    ]
    # Replace datetime patterns
    for pattern in datetime_patterns:
        text = re.sub(pattern, "param", text)
    # Replace UUIDs
    uuid_pattern = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
    text = uuid_pattern.sub("param", text)
    # Replace numbers (only whole tokens)
    tokens = text.split('/')
    new_tokens = []
    for token in tokens:
        sub_tokens = token.split(',')
        new_sub_tokens = []
        for t in sub_tokens:
            if t.isdigit():
                new_sub_tokens.append("param")
            else:
                # Also check if it's a date in text format
                try:
                    date = pd.to_datetime(t, errors='coerce')
                    if pd.notna(date) and (date.day != 1 or date.month != 1 or len(t) > 4):
                        new_sub_tokens.append("param")
                    else:
                        new_sub_tokens.append(t)
                except:
                    new_sub_tokens.append(t)
        new_tokens.append(','.join(new_sub_tokens))
    return '/'.join(new_tokens)

def shannon_entropy(data):
    if not data:
        return 0
    probabilities = [float(data.count(c)) / len(data) for c in set(data)]
    return -sum(p * math.log(p, 2) for p in probabilities)