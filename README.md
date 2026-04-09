💬 NLP Intelligence Pipeline — Sentiment · NER · Text Classification

Scalable NLP pipeline using HuggingFace Transformers, BERT, and spaCy for sentiment analysis, named entity recognition, and text classification — processing 50K+ records/hour.

Show Image
Show Image
Show Image
Show Image
Show Image
Show Image

🎯 Problem
Organizations generate massive volumes of unstructured text — customer feedback, support tickets, contracts, reviews — that go unanalyzed. This pipeline extracts structured intelligence from raw text at scale: sentiment signals, named entities, and topic categories.

✨ Features

Sentiment Analysis — Fine-tuned BERT — positive / negative / neutral with confidence scores
Named Entity Recognition — spaCy + BERT-NER for PER, ORG, LOC, DATE, PRODUCT
Text Classification — Multi-label BERT classifier for N topic categories
Batch + Streaming — Processes millions of records via batch or real-time API
Azure Integration — Azure OpenAI + Azure ML for model hosting
REST API — FastAPI /sentiment, /ner, /classify endpoints
Dashboard — Streamlit visualization for exploring results


📊 Results
TaskModelScoreSentiment AnalysisFine-tuned BERTF1: 91.3%Named Entity RecognitionBERT-NER + spaCyF1: 88.7%Text ClassificationBERT ClassifierAccuracy: 87.4%Pipeline ThroughputBatch mode50K+ records/hour

🛠️ Tech Stack
ComponentTechnologyTransformersBERT · DistilBERTNERspaCy 3.x + HuggingFace NERFrameworkPyTorch · TransformersAPIFastAPIVisualizationStreamlit · PlotlyCloudAzure OpenAI · Azure MLDatabasePostgreSQL · MongoDB

🚀 Quick Start
bashgit clone https://github.com/Venkatesh-Kokkera/nlp-intelligence-pipeline.git
cd nlp-intelligence-pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_trf
python run_pipeline.py --task all --input data/sample.csv
uvicorn app.main:app --host 0.0.0.0 --port 8000
streamlit run dashboard/app.py

Venkatesh Kokkera · 📧 vkokkeravk@gmail.com · 💼 LinkedIn:https://www.linkedin.com/in/venkatesh-ko/ · 📞 +1 (203) 479-2974 . 📍 Lowell, MA 
