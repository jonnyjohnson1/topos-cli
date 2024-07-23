import subprocess


def start_chat():
    """Function to start the API in local mode."""
    # print("\033[92mINFO:\033[0m     API docs available at: \033[1mhttp://127.0.0.1:13394/docs\033[0m")
    # subprocess.run(["python", "topos/chat_api/chat_server.py"]) # A barebones chat server 
    subprocess.run(["uvicorn", "topos.chat_api.server:app", "--host", "0.0.0.0", "--port", "13394", "--workers", "1"])

# start through zrok
# uvicorn main:app --host 127.0.0.1 --port 13394 & zrok expose http://localhost:13394
