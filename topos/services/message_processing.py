# message_processing.py

# from ..generations.ollama_chat import stream_chat
# from topos.FC.semantic_compression import SemanticCompression
# from ..config import get_openai_api_key


# def handle_chat_stream(model, message, message_history, temperature):
#     semantic_compression = SemanticCompression(model=f"ollama:{model}", api_key=get_openai_api_key())
#     # Prepare message history for chat
#     simp_msg_history = [{'role': i['role'], 'content': i['content']} for i in message_history]
#     simp_msg_history.append({'role': 'USER', 'content': message})

#     # Process chat using the summarization model
#     text = []
#     for chunk in stream_chat(simp_msg_history, model=model, temperature=temperature):
#         text.append(chunk)
#     output_combined = ''.join(text)

#     # Fetch semantic category from the output
#     semantic_category = semantic_compression.fetch_semantic_category(output_combined)
#     return output_combined, semantic_category
