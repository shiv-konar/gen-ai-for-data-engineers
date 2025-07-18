from transformers import pipeline

fm_pipeline = pipeline(task="fill-mask")

sentence = "Paris is the <mask> of France"

result = fm_pipeline(sentence)

print(result)