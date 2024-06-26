# Topos
A simple api to use your local machine to play chat games over in the [chat arena](https://github.com/jonnyjohnson1/chat-arena).

Game options are:
1. Debate

## Install Instructions
requires `brew install just`
requires `brew install poetry`

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
We are going to expose our backend service to a public network so our phone/tablet can use it. In this case, we use zrok. Below, is an ngrok setup version.
zrok is opensourced and free.
ngrok has a gated requests/month under its free tier, then requires you pay for it.

1. [Install zrok command](https://docs.zrok.io/docs/getting-started/?_gl=1*1yet1eb*_ga*MTQ1MDc2ODAyNi4xNzE3MDE3MTE3*_ga_V2KMEXWJ10*MTcxNzAxNzExNi4xLjAuMTcxNzAxNzExNi42MC4wLjA.*_gcl_au*NDk3NjM1MzEyLjE3MTcwMTcxMTc.#installing-the-zrok-command) 
2. `zrok enable <given_key>`
3. `zrok status` should show you information
4. Route local path through zrok: `zrok share public http://0.0.0.0:13341`
This will take you to a new screen with an https://<url> at the top.
Insert this url into the field under settings-> "Api Endpoints" -> "Custom API" 
5. After you've insert it into the field, press the test button, and "hello world" should appear next to the button. 

### Step 4b (ngrok): Set up web proxy
