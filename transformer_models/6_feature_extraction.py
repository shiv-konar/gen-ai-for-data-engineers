from transformers import pipeline

feature_extractor = pipeline(task="feature-extraction"
                             , model="facebook/bart-base"
                             , model_kwargs={"cache_dir": "../models"})

text = "Transformers is an awesome library"

result = feature_extractor(text, return_tensors="pt")

print(result)

