from transformers import pipeline

translator = pipeline(task="translation"
                      , model="Helsinki-NLP/opus-mt-en-fr"
                      , model_kwargs={"cache_dir": "../models"})

statement = "My name is Shiv. I live in Aylesbury"

result = translator(statement)

print(result)