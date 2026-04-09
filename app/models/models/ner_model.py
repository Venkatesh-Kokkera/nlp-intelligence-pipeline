import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

class NERModel:
    """
    Hybrid NER model combining spaCy and BERT-NER.
    Extracts: PER, ORG, LOC, DATE, PRODUCT entities.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_trf",
        bert_model: str = "dslim/bert-base-NER"
    ):
        print("Loading NER models...")

        # Load spaCy
        try:
            self.nlp = spacy.load(spacy_model)
            print(f"✅ spaCy model loaded: {spacy_model}")
        except OSError:
            print(f"Downloading spaCy model: {spacy_model}")
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", spacy_model]
            )
            self.nlp = spacy.load(spacy_model)

        # Load BERT NER
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        model = AutoModelForTokenClassification.from_pretrained(
            bert_model
        )
        self.bert_ner = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        print(f"✅ BERT NER model loaded: {bert_model}")

    def extract_spacy(self, text: str) -> list:
        """Extract entities using spaCy."""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy"
            })
        return entities

    def extract_bert(self, text: str) -> list:
        """Extract entities using BERT NER."""
        results = self.bert_ner(text)
        entities = []
        for result in results:
            entities.append({
                "text": result["word"],
                "label": result["entity_group"],
                "score": round(result["score"], 4),
                "start": result["start"],
                "end": result["end"],
                "source": "bert"
            })
        return entities

    def merge_entities(
        self,
        spacy_ents: list,
        bert_ents: list
    ) -> list:
        """Merge and deduplicate entities from both models."""
        seen = set()
        merged = []

        for ent in spacy_ents + bert_ents:
            key = (ent["text"].lower(), ent["label"])
            if key not in seen:
                seen.add(key)
                merged.append(ent)

        # Sort by position
        merged.sort(key=lambda x: x.get("start", 0))
        return merged

    def extract(self, text: str) -> list:
        """
        Extract named entities using hybrid approach.
        Combines spaCy and BERT NER results.
        """
        spacy_ents = self.extract_spacy(text)
        bert_ents = self.extract_bert(text)
        merged = self.merge_entities(spacy_ents, bert_ents)

        print(f"Found {len(merged)} entities in text")
        return merged

    def batch_extract(self, texts: list) -> list:
        """Extract entities from a batch of texts."""
        results = []
        for text in texts:
            entities = self.extract(text)
            results.append({
                "text": text,
                "entities": entities,
                "count": len(entities)
            })
        return results
