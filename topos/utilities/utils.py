import random
# utils.py
import os

def get_root_directory():
    # Get the current file's directory
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Find the first occurrence of "monster" from the right
    monster_index = current_file_directory.rfind("topos")
    
    if monster_index != -1:
        # Get the path up to the first "monster" directory
        base_monster_directory = current_file_directory[:monster_index + len("topos")]
        return base_monster_directory
    else:
        raise ValueError("The 'topos' directory was not found in the path.")
    
def parse_json(data):
    import json
    return json.loads(data)

# convert to a prompt
def create_conversation_string(conversation_data, last_n_messages):
    conversation_string = ""
    for conv_id, messages in conversation_data.items():
        last_messages = list(messages.items())[-last_n_messages:]
        for msg_id, message_info in last_messages:
            role = message_info['role']
            message = message_info['message']
            conversation_string += f"{role}: {message}\n"
    return conversation_string.strip()


# checks if computer is connected to internet
def is_connected(host="8.8.8.8", port=53, timeout=3):
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        return False

def generate_hex_code(n_digits):
    return ''.join(random.choice('0123456789ABCDEF') for _ in range(n_digits))
