build:
    poetry build
    pip install .

run:
    nix run .

zrok:
    zrok share public http://0.0.0.0:13341

zrok_chat:
    zrok share public http://0.0.0.0:13394

cert:
    openssl req -x509 -newkey rsa:4096 -nodes -out topos/cert.pem -keyout topos/key.pem -days 365

python:
	pyi-makespec --onefile main.py
	# add ('topos/config.yaml', 'topos/')
	pyinstaller main.spec
	create-dmg 'dist/main' --overwrite

dmg:
    create-dmg topos.app --volicon "topos/assets/topos_blk_rounded.png" --icon "topos/assets/topos_blk_rounded.png"

stoppg:
    export PGDATA=$(pwd)/pgdata
    echo "Stopping any existing PostgreSQL server..."
    pg_ctl -D "$PGDATA" stop || echo "No existing server to stop."
    