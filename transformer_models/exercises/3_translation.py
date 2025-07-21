from transformers import pipeline

"""
Q1: Some english texts are given below. Translate these texts to Spanish.

Use translation task
Apply Helsinki-NLP/opus-mt-en-es model from huggingface
"""

en_to_sp = pipeline(task="translation"
                    , model="Helsinki-NLP/opus-mt-en-es"
                    , model_kwargs={"cache_dir": "../../models"})
news_1 = "US universities requests Indian and foreign students to rejoin collage"
news_2 = "Hamas representatives will go to Cairo on Saturday for talks on a possible ceasefire in Gaza"
news_3 = "Staying updated with global news is essential in today’s rapidly changing world"
news_4 = "President Trump hosted a lavish dinner at his club and also invited Elon Musk"

results1 = en_to_sp([news_1, news_2, news_3, news_4])

print(results1)

en_to_hi = pipeline(task="translation"
                    , model="Helsinki-NLP/opus-mt-en-hi"
                    , model_kwargs={"cache_dir": "../../models"})

results2 = en_to_hi([news_1, news_2, news_3, news_4])

print(results2)