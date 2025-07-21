"""
Q1: Some english texts are given below. Translate these texts to Spanish.

Apply cross-encoder/nli-deberta-v3-small model from huggingface
Use the following lables:
politics, finance, sports, science and technology, pop culture, breaking news.
"""

from transformers import pipeline

zs_pipeline = pipeline(task="zero-shot-classification"
                       , model="cross-encoder/nli-deberta-v3-small"
                       , model_kwargs={"cache_dir": "../../models"})

article_1 = """
Maharashtra's political landscape shifts Shiv Sena's Eknath Shinde signaling withdrawal from the Chief Minister race, speculation grows around Devendra Fadnavis as a frontrunner. The coalition's dynamics complicate the selection process ahead of the speculated December 5 oath-taking ceremony.
"""
article_2 = """
India vs Australia PM XI, 2nd Day Warm UP Match Live Cricket Score: Continuous drizzle in Canberra on Saturday robbed India of much needed pink-ball practice on day one of their two-day warm-up fixture at the Manuka Oval against Australia Prime Minister's XI, ahead of the day-night second Test in Adelaide.
"""

candidate_labels=[
            "politics",
            "finance",
            "sports",
            "science and technology",
            "pop culture",
            "breaking news",
        ]

result = zs_pipeline([article_1, article_2], candidate_labels)

print(result)