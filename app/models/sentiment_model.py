import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class SentimentModel:
    """
    BERT-based sentiment analysis model.
    Fine-tuned for positive/negative/neutral classification.
    """

    LABELS = ["negative", "neutral", "positive"]

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"
    ):
        print(f"Loading sentiment model: {model_name}")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()
        print("✅ Sentiment model loaded!")

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for input text.
        Returns label, confidence and all scores.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Get scores
        scores = probs[0].cpu().numpy()
        pred_idx = scores.argmax()

        return {
            "label": self.LABELS[pred_idx],
            "confidence": round(float(scores[pred_idx]), 4),
            "scores": {
                label: round(float(score), 4)
                for label, score in zip(self.LABELS, scores)
            }
        }

    def batch_predict(self, texts: list) -> list:
        """Predict sentiment for a batch of texts."""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
