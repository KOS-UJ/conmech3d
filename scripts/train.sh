#!/bin/sh
# Using .venv

screen -XS train quit
screen -dmS train bash -c 'cd ..; source .venv/bin/activate; clear; PYTHONPATH=. python deep_conmech/run_model.py --mode=train --shell; exec bash'
