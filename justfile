build:
    poetry build
    pip install .
    
run:
    topos run

zrok:
    zrok share public http://0.0.0.0:13341

zrok_chat:
    zrok share public http://0.0.0.0:13394

cert:
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
