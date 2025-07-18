from transformers import pipeline

classifier = pipeline(task="zero-shot-classification"
                      , model="facebook/bart-large-mnli"
                      , model_kwargs={"cache_dir": "../models"})

sentence = "I have a problem with my phone and that needs to be resolved asap!"
candidate_labels = ["urgent", "not urgent", "phone", "tablet", "computer"]

result = classifier(sentence, candidate_labels)

print(result)