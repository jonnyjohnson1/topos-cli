
from langchain_community.llms import Ollama

def stream_chat(message_history, model = "solar", temperature=0):
    prompt = "You are an AI agent. Continue from the following transcript.\n"
    if len(message_history) > 0:
        # add the message history prior to the message
        for msg in message_history:
            # add the role and content of message if the message has context to reference
            prompt += msg['role'] + ": " + msg['content'] + "\n"

        
    print("PROMPT\n", prompt)
    llm = Ollama(model=model,temperature=temperature)
    for chunks in llm.stream(prompt):
        yield chunks