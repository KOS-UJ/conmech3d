#!/bin/sh

screen -XS three quit
screen -dmS three bash -c 'cd ..; python3 -m http.server 3003; exec bash'
