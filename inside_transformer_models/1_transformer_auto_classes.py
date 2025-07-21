from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd

comment_1 = "This is an awesome course for beginners"
comment_2 = "I am tired of calling the customer support. I am not getting any resolution"
comment_3 = "The customer support has done good job to resolve my problem"

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokens = tokenizer.tokenize(comment_1)

print(tokens)

ids = tokenizer.encode(comment_1)

print(ids)

tokens_decoded = tokenizer.decode(ids)

print(tokens_decoded)

inputs = tokenizer([comment_1, comment_2, comment_3], padding=True, return_tensors="pt")
print(inputs)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
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