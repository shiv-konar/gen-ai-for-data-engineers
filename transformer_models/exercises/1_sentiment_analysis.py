from transformers import pipeline
import pandas as pd

"""
Q1: A couple of tweets are given below. Perform sentiment analysis on these tweets.

Use sentiment-analysis task
Apply default model from huggingface hub
"""

sentiment_analysis_pipeline_1 = pipeline(task="sentiment-analysis"
                                         , model_kwargs={"cache_dir": "../../models"})

news_1 = "China's factory activity expanded modestly for a second straight month in November, an official survey showed, adding to a string of recent data suggesting a blitz of stimulus is finally trickling through the world's second-largest economy just as Donald Trump ramps up his trade threats.";

news_2 = "The number of #coffee shops in the US is declining for the first time in 9 years largely from #Covid_19. This is assisting Corporations like #Starbucks, Dunkin', & McDonald's to gain momentum at the price of independent businesses struggling to stay open!"

news_3 = "The current weakness in German industry is sapping demand in Switzerland's manufacturing sector, Swiss National Bank Chairman Martin Schlegel said on Saturday."

news_4 = "India hopes that the incoming administration of U.S. President-elect Donald Trump could help keep global crude oil prices low, which would reduce the South Asian country's import bill and support its faltering economic growth."


results1 = sentiment_analysis_pipeline_1([news_1, news_2, news_3, news_4])

print(pd.DataFrame(results1))

"""

Q2: Perform sentiment analysis on the given tweets.

Use sentiment-analysis task
Apply ahmedrachid/FinancialBERT-Sentiment-Analysis model from huggingface hub
"""

sentiment_analysis_pipeline_2 = pipeline(task="sentiment-analysis"
                                         , model="ahmedrachid/FinancialBERT-Sentiment-Analysis"
                                         , model_kwargs={"cache_dir": "../../models"})


results2 = sentiment_analysis_pipeline_2([news_1, news_2, news_3, news_4])

print(pd.DataFrame(results2))

"""
Q3: Perform sentiment analysis on the given tweets.

Use sentiment-analysis task
Apply ProsusAI/finbert model from huggingface hub
"""

sentiment_analysis_pipeline_3 = pipeline(task="sentiment-analysis"
                                         , model="ProsusAI/finbert"
                                         , model_kwargs={"cache_dir": "../../models"})


results3 = sentiment_analysis_pipeline_3([news_1, news_2, news_3, news_4])

print(pd.DataFrame(results3))