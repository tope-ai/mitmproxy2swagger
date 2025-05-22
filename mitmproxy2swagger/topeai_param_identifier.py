from .topeai_init_bert import init_ner_pipeline
from urllib.parse import urlparse, unquote_plus
import pandas as pd
import math
import re

class PIIDetector:
    ner_pipeline = None

    def __init__(self):
        if PIIDetector.ner_pipeline is None:
            PIIDetector.ner_pipeline = init_ner_pipeline()
        self.ner_pipeline = PIIDetector.ner_pipeline

    def analyse_url(self, url: str) -> str:
        path = self.extract_url_parts(url)
        replaced_text = self.replace_special_tokens(path)

        unique_tokens = self.extract_unique_tokens(replaced_text)
        random_like_tokens = {t for t in unique_tokens if self.shannon_entropy(t) > 3.3 and len(t) >= 8}

        final_path = self.replace_random_tokens(replaced_text, random_like_tokens)

        # Run NER on remaining non-param tokens
        sentence = final_path.replace('/', ' ')
        ner_results = self.ner_pipeline(sentence)
        ner_tokens = {e['word'] for e in ner_results}

        tokens = final_path.split('/')
        final_tokens = []
        for token in tokens:
            if token == "param":
                final_tokens.append(token)
            else:
                sub_tokens = token.split(',')
                updated = []
                for t in sub_tokens:
                    t_clean = t.strip()
                    if t_clean in ner_tokens:
                        updated.append("param")
                    else:
                        updated.append(t_clean)
                final_tokens.append(', '.join(updated))
                
        final_path = '/'.join(final_tokens)
        final_path = re.sub(r'(param\s*[, ]\s*)+param', 'param', final_path)

        return final_path


    def extract_url_parts(self, url: str) -> str:
        """Extracts decoded path from a URL, removing 'api/' prefix if present"""
        parsed = urlparse(unquote_plus(url))
        path = parsed.path
        if path.startswith("/api/"):
            path = path[4:]  # remove 'api' prefix
        elif path.startswith("api/"):
            path = path[3:]
        return path


    def replace_special_tokens(self, text: str) -> str:
        """Replaces UUIDs, numbers, and datetime strings with 'param'"""
        datetime_patterns = [
            r"\d{1,2}[ ]?[A-Za-z]{3}[a-z]?[ ]?\d{4},[ ]?\d{2}:\d{2}",
            r"[A-Za-z]{3,9}[ ]\d{1,2},[ ]?\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",
            r"\d{2}[-/\.]\d{2}[-/\.]\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",
            r"\d{4}[-/\.]\d{2}[-/\.]\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",
            r"\d{1,2}[ ]?[A-Za-z]{3}[a-z]?[ ]?\d{4}[ ]?(?:[ T]\d{2}:\d{2})?",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?",
        ]
        for pattern in datetime_patterns:
            text = re.sub(pattern, "param", text)

        # UUIDs
        uuid_pattern = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
        text = uuid_pattern.sub("param", text)

        # Numbers and date-like values
        tokens = text.split('/')
        new_tokens = []
        for token in tokens:
            sub_tokens = token.split(',')
            processed = []
            for t in sub_tokens:
                t_clean = t.strip()
                if t_clean.isdigit():
                    processed.append("param")
                else:
                    try:
                        date = pd.to_datetime(t_clean, errors='coerce')
                        if pd.notna(date) and (date.day != 1 or date.month != 1 or len(t_clean) > 4):
                            processed.append("param")
                        else:
                            processed.append(t_clean)
                    except:
                        processed.append(t_clean)
            new_tokens.append(','.join(processed))
        return '/'.join(new_tokens)


    def extract_unique_tokens(self, text: str) -> set:
        """Extract all non-param tokens split by slashes and non-space patterns"""
        unique_tokens = set()
        for token in text.split('/'):
            sub_tokens = re.findall(r'\S+', token)
            for t in sub_tokens:
                if t and t != "param":
                    unique_tokens.add(t)
        return unique_tokens


    def replace_random_tokens(self, text: str, random_like_tokens: set) -> str:
        """Replaces tokens in text if they are in the random-like set"""
        tokens = text.split('/')
        new_tokens = []
        for token in tokens:
            if token == "param":
                new_tokens.append(token)
            else:
                sub_tokens = token.split(',')
                updated = []
                for t in sub_tokens:
                    t_clean = t.strip()
                    updated.append("param" if t_clean in random_like_tokens else t_clean)
                new_tokens.append(', '.join(updated))
        return '/'.join(new_tokens)


    def shannon_entropy(self, data: str) -> float:
        """Computes Shannon entropy of a string"""
        if not data:
            return 0.0
        probabilities = [float(data.count(c)) / len(data) for c in set(data)]
        return -sum(p * math.log(p, 2) for p in probabilities)
