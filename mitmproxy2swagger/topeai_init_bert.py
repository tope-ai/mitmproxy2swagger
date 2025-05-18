import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def init_ner_pipeline():
    """
    Download and initialize NER pipeline with local caching.
    This ensures the model and tokenizer are downloaded and saved locally
    for future use, so they won't be downloaded again.
    """
    model_id = "dbmdz/bert-large-cased-finetuned-conll03-english"
    local_cache_dir = os.path.abspath("./models/bert_ner")
    os.makedirs(local_cache_dir, exist_ok=True)

    # Explicitly download model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_id, cache_dir=local_cache_dir)

    # Create the pipeline using local model/tokenizer
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    print(f"NER pipeline initialized using cache at: {local_cache_dir}")
    return ner

ner_pipeline = init_ner_pipeline()