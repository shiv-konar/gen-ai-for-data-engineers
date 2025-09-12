from openai import OpenAI
from dotenv import load_dotenv
import os

dotenv_path = "/Users/shiv.konar/Training/llm_engineering/.env"
load_dotenv(dotenv_path, override=True)

system_prompt = """
You are a financial analyst. You are able to analyse financial transactions for a customer.
You are able to give insights on the current financial health of the customer, things like where has customer spent most money on, 
spend by different categories, if the customer keeps spending like this, what is the projected savings.

You are also then able to advise on where to cutback spending on. What will be projected savings if the customer saves more and more.
"""

user_prompt = """
You are a financial analyst. These are my transactions. Can you provide some insight on my financial health?
data = [
    {"transaction_id": 1, "date": "2025-01-05", "merchant": "Amazon", "category": "Shopping", "amount": -120.50, "currency": "GBP"},
    {"transaction_id": 2, "date": "2025-01-07", "merchant": "Starbucks", "category": "Food & Drink", "amount": -4.75, "currency": "GBP"},
    {"transaction_id": 3, "date": "2025-01-09", "merchant": "Tesco", "category": "Groceries", "amount": -56.20, "currency": "GBP"},
    {"transaction_id": 4, "date": "2025-01-10", "merchant": "Uber", "category": "Transport", "amount": -15.80, "currency": "GBP"},
    {"transaction_id": 5, "date": "2025-01-15", "merchant": "Apple", "category": "Electronics", "amount": -899.00, "currency": "GBP"},
    {"transaction_id": 6, "date": "2025-01-18", "merchant": "Netflix", "category": "Subscription", "amount": -9.99, "currency": "GBP"},
    {"transaction_id": 7, "date": "2025-01-20", "merchant": "Salary", "category": "Income", "amount": 2500.00, "currency": "GBP"},
    {"transaction_id": 8, "date": "2025-01-22", "merchant": "British Airways", "category": "Travel", "amount": -450.00, "currency": "GBP"},
    {"transaction_id": 9, "date": "2025-01-25", "merchant": "Marks & Spencer", "category": "Shopping", "amount": -75.30, "currency": "GBP"},
    {"transaction_id": 10, "date": "2025-01-30", "merchant": "HMRC", "category": "Tax", "amount": -320.00, "currency": "GBP"},
]
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

openai = OpenAI()
MODEL = "gpt-4o-mini"

response = openai.chat.completions.create(
    model=MODEL,
    messages=messages
)

print(response.choices[0].message.content)