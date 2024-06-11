import subprocess


def start_chat():
    """Function to start the API in local mode."""
    print("\033[92mINFO:\033[0m     API docs available at: \033[1mhttp://127.0.0.1:13349/docs\033[0m")
    subprocess.run(["uvicorn", "topos.chat_api.fastapi_server:app", "--host", "127.0.0.1", "--port", "13349", "--workers", "4"])

# start through zrok
# uvicorn main:app --host 127.0.0.1 --port 13349 & zrok expose http://localhost:13349
