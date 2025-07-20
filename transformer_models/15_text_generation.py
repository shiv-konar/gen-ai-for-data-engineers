from transformers import pipeline

text_gen_pipeline = pipeline(task="text-generation"
                             , max_length=30
                             , num_return_sequences=3
                             , model_kwargs={"cache_dir": "../models"})

text2text_gen_pipeline = pipeline(task="text2text-generation"
                             , model_kwargs={"cache_dir": "../models"})

text1 = "Hello, I'm a language model"

results1 = text_gen_pipeline(text1)

print(results1)

prompt1 = "question: What is 42? context: 42 is the answer to life, the universe and everything"

results2 = text2text_gen_pipeline(prompt1)

print(results2)


prompt2 = "translate from english to french: I am very sad"

results3 = text2text_gen_pipeline(prompt2)

print(results3)
