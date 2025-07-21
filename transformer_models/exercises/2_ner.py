"""Q1: Some english texts are given below. Identify named entities such as persons and places.

Use ner task
Apply bert-large-cased-finetuned-conll03-english model from huggingface
"""

from transformers import pipeline
import pandas as pd

ner_pipeline = pipeline(model="dbmdz/bert-large-cased-finetuned-conll03-english", model_kwargs={"cache_dir": "../../models"})

text_1 = """
My name is Prashant Kumar Pandey and I work at ScholarNest in Bangalore.
"""
text_2 = """
India vs Australia, 2nd Day Warm UP Match Live Cricket Score: Continuous drizzle in Canberra on Saturday robbed India of much needed pink-ball practice on day one of their two-day warm-up fixture at the Manuka Oval against Australia.
"""

results = ner_pipeline([text_1, text_2])

print(results)