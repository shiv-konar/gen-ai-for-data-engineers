from transformers import pipeline

qa_pipeline = pipeline(task="question-answering")

context = "My name is Shiv. I live in Aylesbury"

question_1 = "where do I live?"
question_2 = "what is my name?"
question_3 = "what have i eaten?"

result1 = qa_pipeline(question=question_1, context=context)
result2 = qa_pipeline(question=question_2, context=context)
result3 = qa_pipeline(question=question_3, context=context)

print(result1)
print(result2)
print(result3)