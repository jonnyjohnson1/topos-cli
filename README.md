# Topos
A simple api to use your local machine to play chat games over in the [chat arena](https://github.com/jonnyjohnson1/chat-arena).

Game options are:
1. Debate

## Install Instructions
requires `brew install just`
requires `brew install poetry`

### Step 1: Install Topos
install the topos package with the command `just build`

### Step 2: Create an SSL certificate for your local host (only necessary if app is deployed to web and needs ssl cert)
install with the command `just cert`

This will prompt you to answer a few questions about your location and name. 
I think they can be left empty by typing `.`

### Step 3: Start Topos on your local machine

`topos run`
`topos run --local`
`topos run --dialogues`


## TODOs

[ ] Write curl tests to make sure the api is working