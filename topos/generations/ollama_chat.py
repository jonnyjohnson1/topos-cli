# summaries.py

import ollama

def stream_chat(message_history, model = "solar", temperature=0):
    stream = ollama.chat(
        model=model,
        messages=message_history,
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

def generate_response(context, prompt, model="solar", temperature=0):
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": context}, {"role": "user", "content": prompt}],
        stream=False
    )
    return response["message"]["content"]

def generate_response_messages(message_history, model="solar", temperature=0):
    response = ollama.chat(
        model=model,
        messages=message_history,
        stream=False
    )
    return response["message"]["content"]
