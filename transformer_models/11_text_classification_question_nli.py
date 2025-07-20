"""
Question Natural Language Inference(QNLI)

QNLI is the task of determining if the answer to a certain question can be found(entailment) in the given document or not(not entailment)

"""

from transformers import pipeline

qnli_pipeline = pipeline(task="text-classification"
                         , model="cross-encoder/qnli-electra-base"
                         , model_kwargs={"cache_dir": "../models"})

prompt1 = """Question: What is the capital of India?
             Sentence: Delhi is the capital of India"""

prompt2 = """Question: What is the capital of India?
             Sentence: Tokyo is the capital of Japan"""

result1 = qnli_pipeline(prompt1)
result2 = qnli_pipeline(prompt2)

print(result1)
print(result2)