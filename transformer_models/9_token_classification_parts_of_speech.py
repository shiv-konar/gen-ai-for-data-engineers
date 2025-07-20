"""
Parts of speech(POS)

The model recognizes parts of speech such as nouns, pronouns, adjectives or verbs
https://huggingface.co/vblagoje/bert-english-uncased-finetuned-pos
"""

from transformers import pipeline

pos_pipeline = pipeline(task="token-classification"
                        , model="vblagoje/bert-english-uncased-finetuned-pos"
                        , model_kwargs={"cache_dir": "../models"})

sentence = "My name is Bart and I work at Meta in San Fransisco"

result = pos_pipeline(sentence)

print(result)