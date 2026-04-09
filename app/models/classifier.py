import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class TextClassifier:
    """
    BERT-based multi-class text classifier.
    Classifies text into topic categories.
    """

    CATEGORIES = [
        "technology",
        "business",
        "healthcare",
        "finance",
        "legal",
        "general"
    ]

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli"
    ):
        print(f"Loading classifier: {model_name}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()
        print("✅ Text classifier loaded!")

    def predict(self, text: str) -> dict:
        """
        Classify text into one of the predefined categories.
        Uses zero-shot classification with BART.
        """
        # Tokenize for each category
        scores = {}
        for category in self.CATEGORIES:
            hypothesis = f"This text is about {category}."
            inputs = self.tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                # Index 2 = entailment (text matches category)
                score = float(probs[0][2])
                scores[category] = round(score, 4)

        # Get best category
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        return {
            "category": best_category,
            "confidence": best_score,
            "all_scores": scores
        }

    def batch_predict(self, texts: list) -> list:
        """Classify a batch of texts."""
        results = []
        for i, text in enumerate(texts):
            print(f"Classifying {i+1}/{len(texts)}", end="\r")
            result = self.predict(text)
            results.append({
                "text": text[:100] + "...",
                "category": result["category"],
                "confidence": result["confidence"]
            })
        return results
