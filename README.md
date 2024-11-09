<p align="center">
  <img src="https://github.com/jonnyjohnson1/topos-cli/blob/main/topos/assets/topos_blk_rounded.png" style="width: 70px; height: 70px;" alt="Private LLMs" />
</p>
<p align="center">
  <em>Private, Personal AI Backend Service</em>
</p>

---

# Topos
Topos is a private, personal AI and database management service.
It comes as an easy-to-install server for your AI apps to connect to.

It is a personal intelligence service, using your own computer to power private conversations with friends, family, and coworkers, collect/store your own private data, and use AI privately. 

Apps Using Topos:
- [chat arena](https://github.com/jonnyjohnson1/chat-arena) #desktop #mobile

Tech: nixOS, ollama, postgres, FastAPI, huggingface-transformers

<p align="center">
  <img src="https://github.com/jonnyjohnson1/topos-cli/blob/main/topos/assets/tui.png" style="zoom:67%;" alt="Terminal User Interface" />
</p>
<p align="center">
  <em>Runs the Terminal User Interface provided by [F1bonacc1](https://github.com/F1bonacc1/process-compose)</em>
</p>

---

## (MacOS) Easy Install With .dmg
*(Experimental)*: This is new, and should work on most MacOS machines!
Simply double click the topos.dmg file, and drag the app into your Applications directory.
You should be able to launch the Topos service anywhere from your machine.

## Install with nix (Recommended)
If nix is not installed:
1. Install nix:
    macos/linux: `sh <(curl -L https://nixos.org/nix/install)`
    windows: `sh <(curl -L https://nixos.org/nix/install) --daemon`
2. Run Topos and all its dependencies:
   ```
   nix run github:jonnyjohnson1/topos-cli/v0.2.3
   ```
   This will start all services including Topos, Postgres, Kafka, and Ollama.

(More nix run information https://determinate.systems/posts/nix-run/)

## Development
Clone the repository:
```
git clone https://github.com/jonnyjohnson1/topos-cli
cd topos-cli
```

For development, you have several options:
### Build Binary
First build topos binary (only usable on machines with nix installed)
```
nix build .#topos
```
run built binary
```
./result/bin/topos
```

(You might also try this)
```
nix build --extra-experimental-features nix-command --extra-experimental-features flakes --show-trace
```
```
./result/bin/services-flake-topos
```

### Dev Shell
```
nix develop
topos run
```

### Poetry Shell
```
nix develop .#poetry
```

## Install Tips

### POSTGRES 
- If postgres is already running, running the bin fails, shut it down first.
- Nix will manage postgres' start/stop function itself when you use it, but if you have started the database elsewhere, it won't be able to manage it, and will fail to start up.


## Install Instructions
requires `brew install just`
requires `brew install poetry`

## Graph Database - Install Neo4j

### Install Neo4j Database on Osx
brew install neo4j
brew services start neo4j

## Topos

### Step 1: Install Topos
install the topos package with the command `just build`

### Step 2: Set the Spacy Model Size
Set the size of the spacy model you wish to use on your system.
There are 'small', 'med', 'large', and 'trf'.

Use the tag like this.
`topos set --spacy small`
`topos set --spacy trf`

### Step 3: Start Topos on your local machine

`topos run`
`topos run --local`

### Step 4a (zrok): Set up web proxy
We are going to expose our backend service to a public network so our phone/tablet can use it. In this case, we use zrok. Below is are the guides to set up ngrok.

zrok is opensourced and free.
ngrok has a gated requests/month under its free tier, then requires you pay for it.

1. Be sure you have the `topos` server running already in another terminal.
2. [Install zrok command](https://docs.zrok.io/docs/getting-started/?_gl=1*1yet1eb*_ga*MTQ1MDc2ODAyNi4xNzE3MDE3MTE3*_ga_V2KMEXWJ10*MTcxNzAxNzExNi4xLjAuMTcxNzAxNzExNi42MC4wLjA.*_gcl_au*NDk3NjM1MzEyLjE3MTcwMTcxMTc.#installing-the-zrok-command)
3. `zrok enable <given_key>`
4. `zrok status` should show you information
5. Route local path through zrok: `zrok share public http://0.0.0.0:13341`
This will take you to a new screen with an https://<url> at the top.
Insert this url into the field under settings-> "Api Endpoints" -> "Custom API"
6. After you've insert it into the field, press the test button, and "hello world" should appear next to the button.

[ ] Enable permanent sharing of zrok url [docs](https://docs.zrok.io/docs/guides/docker-share/#permanent-public-share) (requires Docker)

### Step 4b (ngrok): Set up web proxy



## Future Setup
[ ] Theme the TUI [docs](https://f1bonacc1.github.io/process-compose/tui/)
[ ] Remotely connect to TUI [docs](https://f1bonacc1.github.io/process-compose/client/)
