<p align="center">
  <img src="https://github.com/jonnyjohnson1/topos-cli/blob/main/topos/assets/topos_blk_rounded.png" style="max-width: 100%; height: auto; max-height: 65px;" alt="Private LLMs" />
</p>
<p align="center">
  <em>Private AI Backend Service</em>
</p>

# Topos
A simple server running on your machine to use your local machine to power private conversations with friends, family, and coworkers. Runs with the [chat arena](https://github.com/jonnyjohnson1/chat-arena) app available on desktop and mobile.

## Install with nix (Recommended)
If nix is not installed:
1. Install nix:   
    macos/linux: `sh <(curl -L https://nixos.org/nix/install)`  
    windows: `sh <(curl -L https://nixos.org/nix/install) --daemon` 
Run the software with nix:
1. Download this repo `git clone https://github.com/jonnyjohnson1/topos-cli`
2. `cd topos-cli`
3. build the backend service (only run the topos set --spacy trf line if it is your first time setting up)
```
nix-shell
topos set --spacy trf
topos run
```


## Install Instructions
requires `brew install just`
requires `brew install poetry`
requires `brew install python-tk`

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
