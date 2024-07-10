import socket
import threading

HOST = "127.0.0.1"
PORT = 13394
LISTENER_LIMIT = 5
active_clients = [] # list of all currently conencted users


# function to listen for incoming messages from client
def listen_for_message(client, username):
    while 1:
        max_msg_size = 2048
        message = client.recv(max_msg_size).decode('utf-8')
        if message != '':
            final_msg = username + "~" + message
            send_message_to_all(final_msg)
        else:
            print(f"[ Message from client {username} is empty ]")

# function to send message to a single client
def send_message_to_client(client, message):
    client.sendall(message.encode())


# function to send any new message to all the clients
# currently connected to the server.
def send_message_to_all(message):
    for user in active_clients:
        send_message_to_client(user[1], message)


# function to handle client
def client_handler(client):
    # server will listen for client message that will 
    # contain the username
    while 1:
        max_msg_size = 2048
        username = client.recv(max_msg_size).decode('utf-8')
        if username != '':
            # actions to take when user joins server
            active_clients.append((username, client))
            prompt_message = "SERVER~" + f"{username} joined the chat"
            send_message_to_all(prompt_message)
            break
        else:
            print("[ Client username is empty ]")
    
    threading.Thread(target=listen_for_message, args=(client, username, )).start()


# main function
def main():
    # creating the socket class object
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        server.bind((HOST, PORT))
        print(f"[ Running the server on {HOST} {PORT} ]")
    except:
        print(f"[ Unable to bind to host {HOST} and port {PORT} ]")

    # Set server limit
    server.listen(LISTENER_LIMIT)

    # while loop will keep listening to client connections
    num_connected_clients = 0
    while 1:
        client, address = server.accept()
        num_connected_clients += 1
        print(f"New connected client {address[0]} {address[1]} :: {num_connected_clients} of {LISTENER_LIMIT} available")

        threading.Thread(target=client_handler, args=(client, )).start()

if __name__ == "__main__":
    main()