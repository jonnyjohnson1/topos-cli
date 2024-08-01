from openai import OpenAI


client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

def stream_chat(message_history, model="solar", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=message_history,
        temperature=temperature,
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def generate_response(context, prompt, model="solar", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content

def generate_response_messages(message_history, model="solar", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=message_history,
        stream=False
    )
    return response.choices[0].message.content