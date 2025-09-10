import os
from dotenv import load_dotenv
from openai import OpenAI

dotenv_path = "/Users/shiv.konar/Training/llm_engineering/.env"

load_dotenv(dotenv_path, override=True)
api_key = os.getenv('OPENAI_API_KEY')

openai = OpenAI()

system_prompt = """
You are an analyst that analyzes the financial transactions data and provides summary of where the money has been spent,
where money can be cut back so savings be increased
"""

user_prompt = """
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

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)
