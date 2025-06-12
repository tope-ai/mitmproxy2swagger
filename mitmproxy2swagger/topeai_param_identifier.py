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
        special_tokens = self.replace_special_tokens(path)

        usual_tokens = self.extract_usual_tokens(special_tokens)
        random_tokens = {t for t in usual_tokens if self.shannon_entropy(t) > 3.3 and len(t) >= 8}

        final_path = self.replace_random_tokens(special_tokens, random_tokens)

        # Run NER on remaining non-param tokens
        sentence = final_path.replace('/', ' ')
        ner_results = self.ner_pipeline(sentence)
        ner_tokens = {e['word'] for e in ner_results} # words the ner detects

        tokens = final_path.split('/')
        final_tokens = []
        for token in tokens:
            token_clean = token.strip()

            if re.fullmatch(r"\{(UUID|number|DateTime|string)_.+\}", token_clean):
            #if token_clean in ("{UUID}", "{number}", "{DateTime}", "{string}"):
                final_tokens.append(token_clean)
            elif token_clean in ner_tokens and not token_clean.startswith(('{UUID_', '{DateTime_', '{number_', '{string_')): #clean the token for OpenApi
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', token_clean)
                clean_name = re.sub(r'_+', '_', clean_name)
                final_tokens.append(f"{{string_{clean_name}}}")
                #final_tokens.append("{string}")
            else:
                final_tokens.append(token_clean)

        final_path = '/'.join(final_tokens)
        return final_path
    
        """
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
        #final_path = re.sub(r'(param\s*[, ]\s*)+param', 'param', final_path)

        return final_path
        """


    def extract_url_parts(self, url: str) -> str:
        """Extracts decoded path from a URL, removing 'api/' prefix if present"""
        parsed = urlparse(unquote_plus(url))
        path = parsed.path
        if path.startswith("/api/"):
            path = path[4:]  # remove 'api' prefix
        elif path.startswith("api/"):
            path = path[3:]
        return path
    

    def is_inside_token(self, m: re.Match, text: str) -> bool:
        start, end = m.start(), m.end()
        before = text[:start]
        open_brace = before.rfind('{')
        close_brace = before.rfind('}')
        return open_brace > close_brace


    def replace_special_tokens(self, text: str) -> str:
        """Replaces UUIDs, numbers, and datetime strings with '{UUID}', '{number}', or '{DateTime}'"""
        datetime_patterns = [
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            r"[A-Za-z]{3,9}[ ]\d{1,2},[ ]?\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",
            r"\d{1,2}[ ]?[A-Za-z]{3}[a-z]?[ ]?\d{4},[ ]?\d{2}:\d{2}",
            r"\d{1,2}[ ]?[A-Za-z]{3}[a-z]?[ ]?\d{4}[ ]?(?:[ T]\d{2}:\d{2})?",
            r"\d{2}[-/\.]\d{2}[-/\.]\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",
            r"\d{4}[-/\.]\d{2}[-/\.]\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",

            """
            r"\d{1,2}[ ]?[A-Za-z]{3}[a-z]?[ ]?\d{4},[ ]?\d{2}:\d{2}",
            r"[A-Za-z]{3,9}[ ]\d{1,2},[ ]?\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",
            r"\d{2}[-/\.]\d{2}[-/\.]\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",
            r"\d{4}[-/\.]\d{2}[-/\.]\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?",
            r"\d{1,2}[ ]?[A-Za-z]{3}[a-z]?[ ]?\d{4}[ ]?(?:[ T]\d{2}:\d{2})?",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", 
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", 
            """
        ]

        # UUIDs
        uuid_pattern = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
        text = uuid_pattern.sub(lambda m: f"{{UUID_{m.group(0).replace('-', '_')}}}", text)

        # DateTime pattern
        for pattern in datetime_patterns:
            text = re.sub(
                pattern,
                lambda m: m.group(0) if self.is_inside_token(m, text)
                else f"{{DateTime_{m.group(0).replace('-', '_').replace(':', '_')}}}",
                text
            )

        # Numbers and date-like values
        tokens = text.split('/')
        new_tokens = []
        for token in tokens:
            sub_tokens = token.split(',')
            processed = []
            for t in sub_tokens:
                t_clean = t.strip()
                if re.fullmatch(r"\{(UUID|DateTime|number)_[^{}]+\}", t_clean): ###
                    processed.append(t_clean) ###
                    continue ###
                if t_clean.isdigit():
                    processed.append(f"{{number_{t_clean}}}")
                else:
                    try:
                        date = pd.to_datetime(t_clean, errors='coerce')
                        if pd.notna(date) and (date.day != 1 or date.month != 1 or len(t_clean) > 4):
                            processed.append(f"{{DateTime_{t_clean}}}")
                        else:
                            processed.append(t_clean)
                    except:
                        processed.append(t_clean)
            new_tokens.append(','.join(processed))
        return '/'.join(new_tokens)
        #testprint = '/'.join(new_tokens) --> to see the tokens
        #print(testprint)
        #return testprint


    def extract_usual_tokens(self, text: str) -> set:
        """Extract all tokens split by '/' ignoring special tokens"""
        tokens = text.split('/')
        usual_tokens = {
            token.strip()
            for token in tokens
            if token.strip() and not re.fullmatch(r"\{(UUID|number|DateTime|string)_.+\}", token.strip())
        }
        return usual_tokens


    def replace_random_tokens(self, text: str, random_tokens: set) -> str:
        """Replaces tokens in text if they are in the random-like set with '{string}'"""
        tokens = text.split('/')
        new_tokens = []
        for token in tokens:
            t = token.strip()
            if re.fullmatch(r"\{(UUID|number|DateTime)\_.+\}", t):
            #if t in ("{number}", "{UUID}", "{DateTime}"):
                new_tokens.append(t)
            elif t in random_tokens:
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', t)
                clean_name = re.sub(r'_+', '_', clean_name)
                new_tokens.append(f"{{string_{clean_name}}}")
                #new_tokens.append("{string}")
            else:
                new_tokens.append(t)
        return '/'.join(new_tokens)


    def shannon_entropy(self, data: str) -> float:
        """Computes Shannon entropy of a string"""
        if not data:
            return 0.0
        probabilities = [float(data.count(c)) / len(data) for c in set(data)]
        return -sum(p * math.log(p, 2) for p in probabilities)
