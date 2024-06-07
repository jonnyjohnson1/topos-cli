# utils.py
import os

def get_root_directory():
    def find_setup_py_upwards(start_path):
        current_path = start_path
        while True:
            if 'setup.py' in os.listdir(current_path):
                return current_path
            parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
            if parent_path == current_path:  # Reached the root directory
                return None
            current_path = parent_path

    # Starting from the current directory
    current_directory = os.path.abspath('.')
    setup_py_dir = find_setup_py_upwards(current_directory)

    if setup_py_dir is None:
        raise FileNotFoundError("setup.py not found in the directory tree.")

    return setup_py_dir

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