"""
Grammatical Correctness(Corpus of Linguistic Acceptability CoLA)

This model assess the grammatical correctness of the statement
"""

from transformers import pipeline

cola_pipeline = pipeline(task="text-classification"
                         , model="textattack/distilbert-base-uncased-CoLA"
                         , model_kwargs={"cache_dir": "../models"})

statement1 = "Feeling well I am not."
statement2 = "I am not feeling well."

result1 = cola_pipeline(statement1)
result2 = cola_pipeline(statement2)

print(result1)
print(result2)
