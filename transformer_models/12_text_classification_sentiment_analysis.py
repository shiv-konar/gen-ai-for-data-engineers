"""
Sentiment Analysis

This model predicts if the sentence is positive, negative, neutral or sentiments such as happiness or anger
"""

from transformers import pipeline

sentiment_analysis_pipeline = pipeline(task="sentiment-analysis", model_kwargs={"cache_dir": "../models"})

statement = "I loved Jurassic Park so much!"

result = sentiment_analysis_pipeline(statement)

print(result)
