from dotenv import load_dotenv
from openai import OpenAI

MODEL_GPT = "gpt-4o-mini"
MODEL_LLAMA = "llama3.2"

dotenv_path = "/Users/shiv.konar/Training/llm_engineering/.env"
load_dotenv(dotenv_path, override=True)

openai = OpenAI()

user_question = input("Please enter your question: ")

question = """What does this code do and why: """ +user_question

system_prompt = """You are an expert in the field of programming, LLMs, data science, data engineering and computer science.
You are a helpful technical tutor who can answer questions in depth for the topics mentioned earlier."""

user_prompt = """Please give detailed explanation of the following question: """ +question

response = openai.chat.completions.create(
    model=MODEL_GPT,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

print(response.choices[0].message.content)

