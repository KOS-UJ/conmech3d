#!/bin/sh
# Using .venv

screen -XS tensorboard quit
screen -dmS tensorboard bash -c 'cd ..; source .venv/bin/activate; tensorboard --bind_all --logdir=.; exec bash'
