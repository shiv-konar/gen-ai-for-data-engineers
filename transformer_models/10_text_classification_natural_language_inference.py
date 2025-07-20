"""
Natural Language Inference(NLI)

The model determines the relationship between two given texts. The model takes a premise and a hypothesis and returns
a result that can be:
- entailment - which means hypothesis is true
- contraction - which means hypothesis is false
- neutral - no relation between the premise and hypothesis
"""

from transformers import pipeline

nli_pipeline = pipeline(task="text-classification"
                        , model="roberta-large-mnli"
                        , model_kwargs={"cache_dir": "../models"})

prompt1 = """Premise: Cricket is a game with multiple males playing.
             Hypothesis: Some men are playing a sport."""

prompt2 = """Premise: Cricket is a game with multiple males playing.
             Hypothesis: Cricket is only played by men."""

prompt3 = """Premise: Cricket is a game with multiple males playing.
             Hypothesis: I live in Paris"""

result1 = nli_pipeline(prompt1)
result2 = nli_pipeline(prompt2)
result3 = nli_pipeline(prompt3)

print(result1)
print(result2)
print(result3)