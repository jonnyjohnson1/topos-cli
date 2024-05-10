
import ollama

def stream_chat(message_history, model = "solar", temperature=0):
    stream = ollama.chat(
        model=model,
        messages=message_history,
        stream=True,
    )

    for chunk in stream:
        yield chunk["message"]["content"]