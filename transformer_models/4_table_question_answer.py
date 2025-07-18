import pandas as pd
from transformers import pipeline

table_qa = pipeline(model="google/tapas-large-finetuned-wtq"
                    , task="table-question-answering"
                    , model_kwargs={"cache_dir": "../models"})

question = "how many movies does Leonardo Di Caprio have?"
data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney", "Leonardo Di Caprio"], "Number of movies": ["87", "53", "69", "100"]}
table = pd.DataFrame.from_dict(data)
print(table)

result = table_qa(table=table, query=question)
print(result)