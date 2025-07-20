"""
Named Entity Recognition(NER)

NER models are trained to identify specific entities in a text, such as entities, individuals and places.

Input: My name is Chris and I live in Switzerland
Output: Chris(I-PER, 0.99937123), Switzerland(I-LOC, 0.9998325)
"""

from transformers import pipeline

ner_pipeline = pipeline(task="ner", model_kwargs={"cache_dir": "../models"})

text = "My name is Chris and I live in Switzerland"

result = ner_pipeline(text)

print(result)
