# summaries.py

from langchain_community.llms import Ollama


def stream_chat(message_history, model="solar", temperature=0, current_topic="Unknown", has_topic=False):
    has_topic = False
    if current_topic != "Unknown":
        has_topic = True
        prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is {current_topic}.\n"

    prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is unknown, so try not to make any judgements thus far - only re-express the input words in your own style:\n"
    if len(message_history) > 0:
        # add the message history prior to the message
        for msg in message_history:
            # add the role and content of message if the message has context to reference
            prompt += msg['role'] + ": " + msg['content'] + "\n"
        
    print("PROMPT\n", prompt)
    llm = Ollama(model=model,temperature=temperature)
    for chunks in llm.stream(prompt):
        yield chunks