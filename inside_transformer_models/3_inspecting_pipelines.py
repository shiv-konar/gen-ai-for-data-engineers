from transformers import pipeline
import torch
import pandas as pd

classifier = pipeline(task="sentiment-analysis"
                      , model="distilbert-base-uncased-finetuned-sst-2-english"
                      , model_kwargs={"cache_dir":"../models"})

print(classifier)
print(classifier.tokenizer)
print(classifier.model)
print(classifier.model.config)

comment_1 = "This is an awesome course for beginners"
comment_2 = "I am tired of calling the customer support. I am not getting any resolution"
comment_3 = "i am neither happy nor sad"

tokenizer = classifier.tokenizer
inputs = tokenizer([comment_1, comment_2, comment_3], padding=True, return_tensors="pt")
print(inputs)

print(inputs.input_ids)
print(inputs.attention_mask)

device = torch.device("cpu")
model = classifier.model.to(device)
raw_outputs = model(inputs.input_ids, inputs.attention_mask)
print(raw_outputs)

predictions = torch.nn.functional.softmax(raw_outputs.logits, dim=1)
print(predictions)

pd_result_df = pd.DataFrame(predictions.tolist()) \
                        .rename({0: model.config.id2label[0].title(),
                                1: model.config.id2label[1].title()}, axis=1)

pdf_comments = pd.DataFrame([comment_1, comment_2, comment_3], columns=['Comments'])
pd_final_result_df = pdf_comments.join(pd_result_df)
print(pd_final_result_df)

