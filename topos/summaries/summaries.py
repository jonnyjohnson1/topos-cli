# summaries.py

from langchain_community.llms import Ollama


def stream_chat(message_history, model="solar", temperature=0, current_topic="Unknown", has_topic=False):
    has_topic = False
    if current_topic != "Unknown":
        has_topic = True
        prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is {current_topic}.\n"

    system_prompt = f"You are a smooth talking, eloquent, poignant, insightful AI moderator. The current topic is unknown, so try not to make any judgements thus far - only re-express the input words in your own style:"
    user_prompt = ""
    if message_history:
        # Add the message history prior to the message
        user_prompt += '\n'.join(msg['role'] + ": " + msg['content'] for msg in message_history)

    print(f"\t[ system prompt :: {system_prompt} ]")
    print(f"\t[ user prompt :: {user_prompt} ]")

    llm = Ollama(model=model,temperature=temperature,system=system_prompt)
    for chunks in llm.stream(user_prompt):
        yield chunks