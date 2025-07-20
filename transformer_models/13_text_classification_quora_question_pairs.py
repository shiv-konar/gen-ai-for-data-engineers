"""
Quora Question Pairs(QQP)

This model assess if the two provided questions are paraphrases of each other
"""

from transformers import pipeline

qqp_pipeline = pipeline(task="text-classification"
                        , model="textattack/bert-base-uncased-QQP"
                        , model_kwargs={"cache_dir": "../models"})

question = "Which city is the capital of France?, Where is the capital of France?"

result = qqp_pipeline(question)

print(result)
