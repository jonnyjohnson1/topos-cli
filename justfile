build:
    poetry build
    pip3 install .
    
run:
    topos run

cert:
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
