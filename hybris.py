from groq import Groq
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()
print("API KEY:", os.getenv("groq_api_key"))

client = Groq(api_key=os.getenv("groq_api_key"))

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "stop":
        break
    messages.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.13,
        max_completion_tokens=681,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    print("Assistant:", response)
    messages.append({"role": "assistant", "content": response})