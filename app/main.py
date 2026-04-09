from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.sentiment_model import SentimentModel
from models.ner_model import NERModel
from models.classifier import TextClassifier

app = FastAPI(
    title="NLP Intelligence Pipeline API",
    description="Sentiment Analysis, NER and Text Classification using BERT and spaCy",
    version="1.0.0"
)

# Initialize models
sentiment_model = SentimentModel()
ner_model = NERModel()
classifier = TextClassifier()

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    confidence: float
    scores: dict

class NERResponse(BaseModel):
    entities: list
    count: int

class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict

class FullAnalysisResponse(BaseModel):
    sentiment: dict
    entities: list
    category: str

@app.get("/")
def root():
    return {
        "message": "NLP Intelligence Pipeline API",
        "status": "running",
        "endpoints": [
            "/sentiment",
            "/ner",
            "/classify",
            "/analyze"
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(request: TextRequest):
    try:
        result = sentiment_model.predict(request.text)
        return SentimentResponse(
            label=result["label"],
            confidence=result["confidence"],
            scores=result["scores"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ner", response_model=NERResponse)
def ner(request: TextRequest):
    try:
        entities = ner_model.extract(request.text)
        return NERResponse(
            entities=entities,
            count=len(entities)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=ClassifyResponse)
def classify(request: TextRequest):
    try:
        result = classifier.predict(request.text)
        return ClassifyResponse(
            category=result["category"],
            confidence=result["confidence"],
            all_scores=result["all_scores"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=FullAnalysisResponse)
def analyze(request: TextRequest):
    try:
        sentiment = sentiment_model.predict(request.text)
        entities = ner_model.extract(request.text)
        category = classifier.predict(request.text)
        return FullAnalysisResponse(
            sentiment=sentiment,
            entities=entities,
            category=category["category"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
