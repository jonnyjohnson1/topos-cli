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