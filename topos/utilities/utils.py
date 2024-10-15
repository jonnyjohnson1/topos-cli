import random
# utils.py
import os
import shutil

def get_python_command():
    if shutil.which("python"):
        return "python"
    elif shutil.which("python3"):
        return "python3"
    else:
        raise EnvironmentError("No Python interpreter found")

def get_config_path():
    config_path = os.getenv('TOPOS_CONFIG_PATH')
    if not config_path:
        raise EnvironmentError("TOPOS_CONFIG_PATH environment variable is not set")
    return config_path

def get_root_directory():
    # Get the current file's directory
    current_file_directory = os.path.dirname(os.path.abspath(__file__))

    # Find the first occurrence of "topos" from the right
    topos_index = current_file_directory.rfind("topos")

    if topos_index != -1:
        # Get the path up to the first "topos" directory
        base_topos_directory = current_file_directory[:topos_index + len("topos")]
        return base_topos_directory
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

def generate_deci_code(n_digits):
    return ''.join(random.choice('0123456789') for _ in range(n_digits))

def generate_group_name() -> str:
    return 'GRP-'.join(random.choices(string.ascii_uppercase + string.digits, k=8))
