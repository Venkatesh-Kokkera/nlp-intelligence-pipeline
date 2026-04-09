import argparse
import pandas as pd
from pathlib import Path
from models.sentiment_model import SentimentModel
from models.ner_model import NERModel
from models.classifier import TextClassifier

def load_data(input_path: str) -> list:
    """Load text data from CSV file."""
    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("CSV must have a 'text' column")
    texts = df["text"].dropna().tolist()
    print(f"Loaded {len(texts)} texts from {input_path}")
    return texts

def run_sentiment(texts: list, model: SentimentModel) -> list:
    """Run sentiment analysis on all texts."""
    print(f"\n🎭 Running Sentiment Analysis on {len(texts)} texts...")
    results = []
    for i, text in enumerate(texts):
        print(f"Processing {i+1}/{len(texts)}", end="\r")
        result = model.predict(text)
        results.append({
            "text": text[:100],
            "sentiment": result["label"],
            "confidence": result["confidence"]
        })
    print(f"\n✅ Sentiment Analysis complete!")
    return results

def run_ner(texts: list, model: NERModel) -> list:
    """Run NER on all texts."""
    print(f"\n🏷️  Running NER on {len(texts)} texts...")
    results = []
    for i, text in enumerate(texts):
        print(f"Processing {i+1}/{len(texts)}", end="\r")
        entities = model.extract(text)
        results.append({
            "text": text[:100],
            "entities": entities,
            "entity_count": len(entities)
        })
    print(f"\n✅ NER complete!")
    return results

def run_classify(texts: list, model: TextClassifier) -> list:
    """Run text classification on all texts."""
    print(f"\n📂 Running Classification on {len(texts)} texts...")
    results = []
    for i, text in enumerate(texts):
        print(f"Processing {i+1}/{len(texts)}", end="\r")
        result = model.predict(text)
        results.append({
            "text": text[:100],
            "category": result["category"],
            "confidence": result["confidence"]
        })
    print(f"\n✅ Classification complete!")
    return results

def save_results(results: list, output_path: str):
    """Save results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")

def print_summary(results: list, task: str):
    """Print summary statistics."""
    df = pd.DataFrame(results)
    print(f"\n📊 {task} Summary:")
    print(f"   Total processed: {len(df)}")

    if task == "Sentiment":
        counts = df["sentiment"].value_counts()
        for label, count in counts.items():
            pct = count / len(df) * 100
            print(f"   {label}: {count} ({pct:.1f}%)")

    elif task == "NER":
        total_entities = df["entity_count"].sum()
        avg_entities = df["entity_count"].mean()
        print(f"   Total entities: {total_entities}")
        print(f"   Avg per text: {avg_entities:.1f}")

    elif task == "Classification":
        counts = df["category"].value_counts()
        for cat, count in counts.items():
            pct = count / len(df) * 100
            print(f"   {cat}: {count} ({pct:.1f}%)")

def main(args):
    """Main pipeline runner."""
    print("🚀 Starting NLP Intelligence Pipeline...")

    # Load data
    texts = load_data(args.input)

    # Initialize models
    sentiment_model = SentimentModel()
    ner_model = NERModel()
    classifier = TextClassifier()

    # Run tasks
    if args.task in ["sentiment", "all"]:
        results = run_sentiment(texts, sentiment_model)
        print_summary(results, "Sentiment")
        save_results(
            results,
            f"{args.output}/sentiment_results.csv"
        )

    if args.task in ["ner", "all"]:
        results = run_ner(texts, ner_model)
        print_summary(results, "NER")
        save_results(
            results,
            f"{args.output}/ner_results.csv"
        )

    if args.task in ["classify", "all"]:
        results = run_classify(texts, classifier)
        print_summary(results, "Classification")
        save_results(
            results,
            f"{args.output}/classification_results.csv"
        )

    print("\n🎉 NLP Pipeline complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NLP Intelligence Pipeline"
    )
    parser.add_argument(
        "--task",
        choices=["sentiment", "ner", "classify", "all"],
        default="all",
        help="Task to run"
    )
    parser.add_argument(
        "--input",
        default="data/sample.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory"
    )
    args = parser.parse_args()
    main(args)
