from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
data = ["Tesla earnings tanked by 3%", "I hate you"]
sentiment_pipeline(data)
